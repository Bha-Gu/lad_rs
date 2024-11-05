use std::{cmp::Ordering, collections::HashMap};

use polars::prelude::*;

#[derive(Clone)]
pub struct Binarizer {
    cutpoints: Vec<Series>,
    threshold: f64,
    nominal_size: usize,
    max_cutpoints: usize,
    depth: usize,
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
            cutpoints: Vec::new(),
            threshold,
            nominal_size: nominal_size.max(2usize.saturating_pow(depth as u32)),
            max_cutpoints: max_cutpoints_per_column,
            depth,
        }
    }

    #[must_use]
    pub fn get_cutpoints(&self) -> Vec<Series> {
        self.cutpoints.clone()
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
        self.cutpoints = Vec::new();
        let unique_labels = label.unique_stable()?;
        let mut label_counts = vec![0u128; unique_labels.len()];
        for l in label.iter() {
            for (j, lj) in unique_labels.iter().enumerate() {
                if lj == l {
                    label_counts[j] += 1;
                    break;
                }
            }
        }

        let label_counts = label_counts;
        for (idx, (feature_name, data_type)) in schema.iter().enumerate() {
            let column = data[idx].clone();
            let a = column.n_unique().unwrap_or_default();
            if a == 2 || data_type.is_bool() {
                let unique_values = column.unique_stable()?;

                self.cutpoints.push(Series::new(
                    format!("Bool#{}", feature_name).into(),
                    unique_values,
                ));

                continue;
            }
            if a <= self.nominal_size || data_type.is_string() {
                let unique_values = column.unique_stable()?;

                self.cutpoints.push(Series::new(
                    format!("Nominal#{}", feature_name).into(),
                    unique_values,
                ));

                continue;
            }

            if data_type.is_numeric() {
                let mut column_and_label = DataFrame::new(vec![label.clone(), column])?;
                let mut running_counts = vec![0u128; unique_labels.len()];
                column_and_label = column_and_label
                    .sort([feature_name.to_string()], SortMultipleOptions::default())?;
                let mut cps = Vec::new();
                let sorted = column_and_label.drop_in_place(feature_name.as_ref())?;
                let labels = column_and_label.drop_in_place(label.name().as_ref())?;
                let mut prev_label = labels.get(0)?;
                let mut prev_value = sorted.get(0)?;
                running_counts[unsafe {
                    unique_labels
                        .iter()
                        .position(|x| x == prev_label)
                        .unwrap_unchecked()
                }] += 1;

                let mut segments = vec![(0, sorted.len())]; // Start with the full range

                for _ in 0..self.depth {
                    let mut new_segments = Vec::new();

                    for &(start, end) in &segments {
                        // Find the best split point in the current segment
                        if let Some(split_idx) = Self::find_best(
                            self,
                            start,
                            end,
                            &sorted,
                            &labels,
                            &unique_labels,
                            data_type,
                            &mut cps,
                        )? {
                            // Create two new segments based on the split point
                            new_segments.push((start, split_idx));
                            new_segments.push((split_idx + 1, end));
                        }
                    }

                    // Update segments for the next depth level
                    segments = new_segments;
                }

                //for (s, l) in sorted.iter().zip(labels.iter()).skip(1) {
                //    let score = Self::score(&running_counts, &label_counts);
                //    running_counts[unsafe {
                //        unique_labels.iter().position(|x| x == l).unwrap_unchecked()
                //    }] += 1;
                //    if prev_label != l && prev_value != s {
                //        if score >= self.threshold {
                //            cps.push((
                //                AnyValue::from(unsafe {
                //                    Series::new("tmp".into(), [s.clone(), prev_value])
                //                        .mean()
                //                        .unwrap_unchecked()
                //                })
                //                .cast(data_type),
                //                score,
                //            ));
                //        }
                //        prev_value = s;
                //        prev_label = l;
                //    }
                //}
                //
                //cps = cps
                //    .windows(3)
                //    .filter_map(|window| {
                //        let (_x1, s1) = &window[0];
                //        let (x2, s2) = &window[1];
                //        let (_x3, s3) = &window[2];
                //
                //        if s2 > s1 && s2 > s3 {
                //            Some((x2.clone(), *s2))
                //        } else {
                //            None
                //        }
                //    })
                //    .collect();
                //
                //cps.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let cps = cps
                    .iter()
                    //.rev()
                    //.take(self.max_cutpoints)
                    .map(|(x, s)| {
                        print!("{s} ");
                        x.to_owned()
                    })
                    .collect::<Vec<_>>();
                println!();
                self.cutpoints
                    .push(Series::new(format!("Numeric#{}", feature_name).into(), cps));
            }
        }
        Ok(())
    }

    pub fn transform(&self, df: &DataFrame) -> PolarsResult<DataFrame> {
        let mut out = DataFrame::default();

        let names = self
            .cutpoints
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
            // TODO: Add binary encoding using '''self.depth''' so instead of making n columns for n
            // nominal values it makes ceil(log_2(n)) columns for n nominal values
            if dtype == "Nominal" {
                //for value in col.iter() {
                //    out.hstack_mut(&[Series::new(
                //        format!(
                //            "{feature_name} = {}",
                //            match value {
                //                AnyValue::String(a) => a.to_string(),
                //                _ => value.to_string(),
                //            }
                //        )
                //        .into(),
                //        column.iter().map(|x| x == value).collect::<Vec<_>>(),
                //    )])?;
                //}
                // Step 1: Determine unique values and required binary width
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
                for cutpoint in col.iter() {
                    out.hstack_mut(&[Series::new(
                        format!("{feature_name} > {cutpoint}").into(),
                        column.iter().map(|x| x > cutpoint).collect::<Vec<_>>(),
                    )])?;
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
        let mut running_counts = vec![0u128; unique_labels.len()];
        let mut label_counts = vec![0u128; unique_labels.len()];
        for l in start..end {
            for (j, lj) in unique_labels.iter().enumerate() {
                if lj == labels.get(l)? {
                    label_counts[j] += 1;
                    break;
                }
            }
        }

        let label_counts = label_counts;
        let mut prev_label = labels.get(start)?;
        let mut prev_value = sorted.get(start)?;
        running_counts[unsafe {
            unique_labels
                .iter()
                .position(|x| x == prev_label)
                .unwrap_unchecked()
        }] += 1;

        let mut best_score = 0.0;
        let mut best_value = None;
        let mut best_index = None;

        for (idx, (s, l)) in sorted
            .iter()
            .skip(start)
            .take(end - start)
            .zip(labels.iter().skip(start).take(end - start))
            .skip(1)
            .enumerate()
        {
            let score = Self::score(&running_counts, &label_counts);
            running_counts
                [unsafe { unique_labels.iter().position(|x| x == l).unwrap_unchecked() }] += 1;

            if prev_label != l && prev_value != s {
                if score >= self.threshold && score > best_score {
                    best_score = score;
                    best_value = Some(unsafe {
                        Series::new("tmp".into(), [s.clone(), prev_value])
                            .mean()
                            .unwrap_unchecked()
                    });
                    best_index = Some(idx + start + 1);
                }
                prev_value = s;
                prev_label = l;
            }
        }

        if let Some(value) = best_value {
            cps.push((AnyValue::from(value).cast(data_type), best_score));
        }

        Ok(best_index)
    }

    fn score(runner: &[u128], total: &[u128]) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let rates = runner
            .iter()
            .zip(total.iter())
            .map(|(&r, &t)| r as f64 / t as f64)
            .collect::<Vec<_>>();

        let len = rates.len();

        let mut out = 0.0;

        for i in 0..len {
            for j in (i + 1)..len {
                let tmp = rates[i] - rates[j];
                out += tmp * tmp;
            }
        }

        let len = len as f64;

        let max = (len * len - (rates.len() % 2) as f64) / 4.;

        f64::sqrt(out / max)

        //let sum = rates.iter().sum::<f64>();
        //
        //
        //
        //#[allow(clippy::cast_precision_loss)]
        //let score = rates.iter().map(|x| x * (sum - x)).sum::<f64>() / (runner.len() - 1) as f64;
        //
        //let x = sum - score;
        //
        //let k = -2.0 + f64::from((2u32).pow(runner.len() as u32 - 1));
        //
        //x / k.mul_add(1.0 - x, 1.0)
    }

    //fn entropy(runner: &[u128], total: &[u128]) -> f64 {
    //    // Calculate total counts
    //    let total_count: u128 = total.iter().sum();
    //
    //    // Calculate probabilities
    //    let probabilities: Vec<f64> = runner
    //        .iter()
    //        .map(|&r| {
    //            if total_count > 0 {
    //                r as f64 / total_count as f64
    //            } else {
    //                0.0
    //            }
    //        })
    //        .collect();
    //
    //    let entropy_value: f64 = probabilities
    //        .iter()
    //        .filter_map(|&p| {
    //            if p > 0.0 {
    //                Some(-p * p.log2()) // Using natural logarithm
    //            } else {
    //                None
    //            }
    //        })
    //        .sum();
    //
    //    entropy_value
    //}
}
