use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};

#[derive(Clone)]
pub struct Binarizer {
    sorted_cutpoints: Vec<Series>,
    pub threshold: f64,
    numeric_as_nominal_upper_threshold: usize,
    max_cutpoints_per_column: usize,
    search_depth: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct BinSaver {
    pub threshold: f64,
    nominal_size: usize,
    max_cutpoints: usize,
    depth: usize,
    num_cutpoints: usize,
}

impl Binarizer {
    #[must_use]
    pub fn new(
        threshold: f64,
        nominal_size: usize,
        max_cutpoints_per_column: usize,
        depth: usize,
    ) -> Self {
        Self {
            sorted_cutpoints: Vec::new(),
            threshold,
            numeric_as_nominal_upper_threshold: nominal_size,
            max_cutpoints_per_column,
            search_depth: depth,
        }
    }

    pub fn save(self, json_path: &str, csv_dir: &str) -> Result<(), Box<dyn Error>> {
        // Save each Series as a CSV file named "cutpoints_{i}.csv"
        for (i, series) in self.sorted_cutpoints.iter().enumerate() {
            let file_path = format!("{}/cutpoints_{}.csv", csv_dir, i);
            let mut file = File::create(&file_path)?;
            // Wrap the Series in a DataFrame so that we can write it as CSV.
            let mut df = DataFrame::new(vec![series.clone()])?;
            CsvWriter::new(&mut file).finish(&mut df)?;
        }

        // Create a dummy struct to hold the other fields.
        let saver = BinSaver {
            threshold: self.threshold,
            nominal_size: self.numeric_as_nominal_upper_threshold,
            max_cutpoints: self.max_cutpoints_per_column,
            depth: self.search_depth,
            num_cutpoints: self.sorted_cutpoints.len(),
        };

        let file = File::create(json_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &saver)?;
        Ok(())
    }

    // Load function:
    // - Reads the BinSaver from the JSON file to recover non-Series fields.
    // - Loads each CSV file (assumed named "cutpoints_{i}.csv") to reconstruct the Series vector.
    pub fn load(json_path: &str, csv_dir: &str) -> Result<Binarizer, Box<dyn Error>> {
        // Load the dummy struct from JSON.
        let file = File::open(json_path)?;
        let reader = BufReader::new(file);
        let saver: BinSaver = serde_json::from_reader(reader)?;

        // Read the Series CSV files.
        let mut cutpoints = Vec::with_capacity(saver.num_cutpoints);
        for i in 0..saver.num_cutpoints {
            let file_path = format!("{}/cutpoints_{}.csv", csv_dir, i);
            let file = File::open(file_path)?;
            let df = CsvReader::new(file).finish()?;
            // Assuming each CSV file contains one column, we extract it.
            let series = df
                .select_at_idx(0)
                .ok_or_else(|| format!("No column found in CSV cutpoints_{}.csv", i))?
                .clone();
            cutpoints.push(series);
        }

        // Reconstruct the full Binarizer.
        Ok(Binarizer {
            sorted_cutpoints: cutpoints,
            threshold: saver.threshold,
            numeric_as_nominal_upper_threshold: saver.nominal_size,
            max_cutpoints_per_column: saver.max_cutpoints,
            search_depth: saver.depth,
        })
    }

    #[must_use]
    pub fn get_cutpoints(&self) -> Vec<Series> {
        self.sorted_cutpoints.clone()
    }

    pub fn generate_cutpoints(
        &mut self,
        data: &DataFrame,
        label: &Series,
    ) -> Result<(), PolarsError> {
        if data.shape().0 != label.len() {
            println!(
                "Lengths of data {} and label {} do not match",
                data.shape().0,
                label.len()
            );
            return Ok(());
        }
        let schema = data.schema();
        self.sorted_cutpoints = Vec::new();
        let unique_classes = label.unique_stable()?;
        println!("{unique_classes:?}");

        for (idx, (feature_name, data_type)) in schema.iter().enumerate() {
            let column = data[idx].clone();
            let a = column.n_unique().unwrap_or_default();
            if a == 2 || data_type.is_bool() {
                let unique_values = column.unique_stable()?;

                self.sorted_cutpoints.push(Series::new(
                    format!("Bool#{}", feature_name).into(),
                    unique_values,
                ));

                continue;
            }
            if a <= self.numeric_as_nominal_upper_threshold || data_type.is_string() {
                let unique_values = column.unique_stable()?;

                self.sorted_cutpoints.push(Series::new(
                    format!("Nominal#{}", feature_name).into(),
                    unique_values,
                ));

                continue;
            }

            if data_type.is_numeric() {
                let mut column_and_label = DataFrame::new(vec![label.clone(), column])?;
                column_and_label = column_and_label
                    .sort([feature_name.to_string()], SortMultipleOptions::default())?;
                let mut cutpoints = Vec::new();
                let sorted = column_and_label.drop_in_place(feature_name.as_ref())?;
                let labels = column_and_label.drop_in_place(label.name().as_ref())?;

                let mut segments = vec![(0, sorted.len())]; // Start with the full range

                for _ in 0..self.search_depth {
                    let mut new_segments = Vec::new();

                    for &(start, end) in &segments {
                        // Find the best split point in the current segment
                        if let Some(split_idx) = Self::find_best(
                            self,
                            start,
                            end,
                            &sorted,
                            &labels,
                            &unique_classes,
                            data_type,
                            &mut cutpoints,
                        )? {
                            // Create two new segments based on the split point
                            new_segments.push((start, split_idx));
                            new_segments.push((split_idx, end));
                        }
                    }

                    // Update segments for the next depth level
                    segments = new_segments;
                }

                let cps = cutpoints
                    .iter()
                    //.rev()
                    //.take(self.max_cutpoints)
                    .map(|(x, _s)| x.to_owned())
                    .collect::<Vec<_>>();
                self.sorted_cutpoints
                    .push(Series::new(format!("Numeric#{}", feature_name).into(), cps));
            }
        }
        Ok(())
    }

    pub fn transform(&self, df: &DataFrame) -> PolarsResult<DataFrame> {
        let mut out = DataFrame::default();

        let names = self
            .sorted_cutpoints
            .iter()
            .map(|x| (unsafe { x.name().split_once('#').unwrap_unchecked() }, x));

        for ((dtype, feature_name), col) in names {
            let column = df.column(feature_name)?;
            if dtype == "Bool" {
                let value = col.get(0)?;
                out.hstack_mut(&[Series::new(
                    format!(
                        "{feature_name} = {}",
                        match value {
                            AnyValue::String(a) => a.to_string(),
                            _ => value.to_string(),
                        }
                    )
                    .into(),
                    column.iter().map(|x| x == value).collect::<Vec<_>>(),
                )])?;
                continue;
            }

            if dtype == "Nominal" {
                let unique_values: Vec<_> = col.iter().collect();
                let n = unique_values.len();
                let num_bits = (n as f64).log2().ceil() as usize;

                // Step 2: Map each unique value to a binary representation
                let mut binary_map = HashMap::new();
                for (i, value) in unique_values.iter().enumerate() {
                    let binary_code: Vec<bool> =
                        (0..num_bits).rev().map(|bit| (i >> bit) & 1 == 1).collect();
                    binary_map.insert(value.to_string(), binary_code);
                }

                // Step 3: Create columns based on binary representation
                for bit_pos in 0..num_bits {
                    let column_name = format!("{}_bit_{}", feature_name, bit_pos);
                    let bit_col: Vec<_> = column
                        .iter()
                        .map(|val| {
                            //val.map_or(false, |v| {
                            binary_map
                                .get(&val.to_string())
                                .map_or(false, |bits| bits[bit_pos])
                            //})
                        })
                        .collect();
                    //println!("{}, {}", column.len(), bit_col.len());
                    out.hstack_mut(&[Series::new(column_name.into(), bit_col)])?;
                }
            }
            if dtype == "Numeric" {
                if col.len() == 0 {
                    continue;
                }

                let mut nominal_classes: Vec<usize> = Vec::new();
                let col = col.sort(SortOptions::default())?;
                for val in column.iter() {
                    let a = col.iter().position(|cut| val <= cut).unwrap_or(col.len());
                    nominal_classes.push(a);
                }

                // Step 2: Determine bit length needed for encoding classes
                let n_classes = col.len() + 1; // One class for each cutpoint range, plus one for values above the last cutpoint
                let num_bits = (n_classes as f64).log2().ceil() as usize;
                // Step 3: Map each class to a binary code
                let mut binary_map = HashMap::new();
                for i in 0..n_classes {
                    let binary_code: Vec<bool> =
                        (0..num_bits).rev().map(|bit| (i >> bit) & 1 == 1).collect();
                    binary_map.insert(i, binary_code);
                }

                // Step 4: Create binary columns for each bit position
                for bit_pos in 0..num_bits {
                    let column_name = format!("{}_bit_{}", feature_name, bit_pos);
                    let bit_col: Vec<bool> = nominal_classes
                        .iter()
                        .map(|&class_idx| binary_map[&class_idx][bit_pos])
                        .collect();

                    // Add the bit column to the DataFrame
                    out.hstack_mut(&[Series::new(column_name.into(), bit_col)])?;
                }
            } else {
                //println!("{data_type} not supported yet. Skipping");
            }
        }
        Ok(out)
    }

    fn find_best<'a>(
        &self,
        start: usize,
        end: usize,
        sorted: &Series,
        labels: &Series,
        unique_labels: &Series,
        data_type: &'a DataType,
        cps: &mut Vec<(AnyValue<'a>, f64)>,
    ) -> PolarsResult<Option<usize>> {
        let mut sample_count_until_currently_considered_cutpoint_per_class =
            vec![0u128; unique_labels.len()];
        let mut sample_count_per_class = vec![0u128; unique_labels.len()];
        for l in start..end {
            for (j, lj) in unique_labels.iter().enumerate() {
                if lj == labels.get(l)? {
                    sample_count_per_class[j] += 1;
                    break;
                }
            }
        }

        if start == end {
            return Ok(None);
        }

        let sample_count_per_class = sample_count_per_class;
        let mut prev_label = labels.get(start)?;
        let mut prev_value = sorted.get(start)?;
        sample_count_until_currently_considered_cutpoint_per_class[unsafe {
            unique_labels
                .iter()
                .position(|x| x == prev_label)
                .unwrap_unchecked()
        }] += 1;

        let mut best_score = 0.0;
        let mut best_scoring_value = None;
        let mut index_of_best_scoring_value = None;

        for (idx, (s, l)) in sorted
            .iter()
            .skip(start)
            .take(end - start)
            .zip(labels.iter().skip(start).take(end - start))
            .skip(1)
            .enumerate()
        {
            if prev_label != l {
                let score = Self::score(
                    &sample_count_until_currently_considered_cutpoint_per_class,
                    &sample_count_per_class,
                );
                sample_count_until_currently_considered_cutpoint_per_class
                    [unsafe { unique_labels.iter().position(|x| x == l).unwrap_unchecked() }] += 1;

                if prev_value != s {
                    if score >= self.threshold && score > best_score {
                        best_score = score;
                        best_scoring_value = Some(unsafe {
                            Series::new("tmp".into(), [s.clone(), prev_value])
                                .mean()
                                .unwrap_unchecked()
                        });
                        index_of_best_scoring_value = Some(idx + start + 1);
                    }
                }
            }
            prev_value = s;
            prev_label = l;
        }

        if let Some(value) = best_scoring_value {
            cps.push((AnyValue::from(value).cast(data_type), best_score));
        }

        Ok(index_of_best_scoring_value)
    }

    pub fn score(runner: &[u128], total: &[u128]) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let classwise_coverage_before_cutpoint = runner
            .iter()
            .zip(total.iter())
            .map(|(&r, &t)| r as f64 / t as f64)
            .collect::<Vec<_>>();

        let number_of_classes = classwise_coverage_before_cutpoint.len();

        let mut out = 0.0;

        for i in 0..number_of_classes {
            for j in (i + 1)..number_of_classes {
                let tmp =
                    classwise_coverage_before_cutpoint[i] - classwise_coverage_before_cutpoint[j];
                out += tmp * tmp;
            }
        }

        let max = (number_of_classes * number_of_classes - (number_of_classes % 2)) / 4;

        f64::sqrt(out / max as f64)
    }
}
