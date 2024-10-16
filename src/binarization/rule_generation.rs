use std::collections::HashSet;

use super::binarize::Binarizer;
use polars::prelude::*;
use std::time::Instant;

type Pattern = HashSet<(bool, String)>;

#[derive(Clone)]
pub struct RuleGenerator {
    bin: Binarizer,
    max: usize,
    rules: Vec<(usize, HashSet<(bool, String)>)>,
    labels: Series,
    fallback_label: Option<usize>,
}

impl RuleGenerator {
    pub fn new(bin: &Binarizer, max: usize) -> Self {
        Self {
            bin: bin.clone(),
            max,
            rules: Vec::new(),
            labels: Series::new("tmp".into(), [0]),
            fallback_label: None,
        }
    }

    pub fn get_rules(&self) -> Vec<(usize, HashSet<(bool, String)>)> {
        self.rules.clone()
    }

    pub fn predict(&self, data: &DataFrame) -> PolarsResult<Series> {
        let data: DataFrame = self.bin.transform(data).unwrap();
        let mut predictions: Vec<Option<usize>> = vec![None; data.height()];

        for (label, pattern) in &self.rules {
            let coverage = self.coverage(&data, pattern);

            // Iterate over each index in the coverage vector (a)
            for (&is_covered, prediction) in coverage.iter().zip(predictions.iter_mut()) {
                // If the current value is true and the result is None for this index, set the label
                if is_covered && prediction.is_none() {
                    *prediction = Some(label.clone()); // Use .clone() since label is a reference
                }
            }
        }
        //todo!();
        Ok(Series::new(
            "Predictions".into(),
            predictions
                .iter()
                .map(|x| {
                    x.as_ref().map_or_else(
                        || {
                            self.labels
                                .get(self.fallback_label.unwrap())
                                .unwrap()
                                .clone()
                        },
                        |x| self.labels.get(*x).unwrap().clone(),
                    )
                })
                .collect::<Vec<_>>(),
        ))
    }

    pub fn fit(&mut self, data: &DataFrame, labels: &Series) -> PolarsResult<()> {
        let features = data.get_column_names();
        let unique_y = labels.unique_stable()?;
        self.labels = unique_y.clone();

        // Divide data into groups based on the labels
        let mut grouped_dfs: Vec<DataFrame> = self.divide_data(data, labels);

        self.fallback_label = grouped_dfs
            .iter()
            .enumerate()
            .max_by_key(|(_, df)| df.shape().0)
            .map(|(i, _)| i)
            .clone();

        let mut prime_patterns: Vec<(usize, Pattern)> = Vec::new();
        let mut prev_degree_patterns: Vec<Pattern> = vec![HashSet::new()];

        let max_features = if self.max > features.len() || self.max == 0 {
            features.len()
        } else {
            self.max
        };

        for d in 1..=max_features {
            println!("{d}");
            let start_time = Instant::now();
            let mut curr_degree_patterns = Vec::new();

            let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
            println!("{remaining_shapes:?}");

            for curr_pattern in &prev_degree_patterns {
                for feature in &features {
                    for term in [true, false] {
                        let mut next_pattern = curr_pattern.clone();
                        if !next_pattern.insert((term, feature.to_string())) {
                            continue;
                        }

                        let valid_pattern = next_pattern.iter().all(|t| {
                            let mut test_pattern = next_pattern.clone();
                            test_pattern.remove(t);
                            prev_degree_patterns.contains(&test_pattern)
                        });

                        if !valid_pattern {
                            continue;
                        }

                        let counts: Vec<_> = grouped_dfs
                            .iter()
                            .map(|df| {
                                self.coverage(df, &next_pattern)
                                    .into_iter()
                                    .filter(|&x| x)
                                    .count()
                            })
                            .collect();

                        let tmp = counts.iter().filter(|&&x| x >= 1).count();

                        if tmp == 1 {
                            for (i, &count) in counts.iter().enumerate() {
                                if count == 0 || grouped_dfs[i].shape().0 == 0 {
                                    continue;
                                }

                                // Update grouped_dfs by filtering out covered rows
                                grouped_dfs[i] = grouped_dfs[i]
                                    .filter(
                                        &self
                                            .coverage(&grouped_dfs[i], &next_pattern)
                                            .into_iter()
                                            .map(|x| !x)
                                            .collect(),
                                    )
                                    .unwrap();

                                prime_patterns.push((i, next_pattern.clone()));
                                let remaining_shapes: Vec<_> =
                                    grouped_dfs.iter().map(|df| df.shape().0).collect();
                                println!(
                                    "from pattern {:?}: {remaining_shapes:?}",
                                    next_pattern.clone()
                                );

                                break;
                            }
                        } else if tmp > 1 {
                            curr_degree_patterns.push(next_pattern);
                        }
                    }
                }
            }

            let duration = start_time.elapsed();
            println!("Time taken: {:.2} milliseconds", duration.as_millis());

            let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
            println!("{remaining_shapes:?}");

            if remaining_shapes.iter().sum::<usize>() == 0 {
                break;
            }

            if d == max_features {
                self.fallback_label = grouped_dfs
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, df)| df.shape().0)
                    .map(|(i, _)| i);
            }

            prev_degree_patterns = curr_degree_patterns;
        }

        self.rules = prime_patterns;
        Ok(())
    }
    //
    //pub fn fit_old(&mut self, data: &DataFrame, labels: &Series) -> PolarsResult<()> {
    //    //println!("Debug0");
    //    let features = data.get_column_names();
    //    // Ensure y is categorical or can be grouped
    //    let unique_y = labels.unique_stable()?;
    //    self.labels = unique_y.iter().map(|x| x.to_string()).collect();
    //    // Initialize a Vec to hold the resulting DataFrames
    //    let mut grouped_dfs: Vec<DataFrame> = self.divide_data(data, labels);
    //    //println!("Debug2");
    //
    //    self.fallback_label = grouped_dfs
    //        .iter()
    //        .enumerate()
    //        .max_by(|(_, a), (_, b)| a.shape().0.cmp(&b.shape().0))
    //        .map(|(i, _)| self.labels[i].clone());
    //
    //    // Perform further operations with grouped_dfs if necessary
    //    let mut prime_patterns: Vec<(String, Pattern)> = Vec::new();
    //    let mut prev_degree_patterns: Vec<Pattern> = vec![HashSet::new()];
    //
    //    if self.max > features.len() || self.max == 0 {
    //        self.max = features.len();
    //    }
    //
    //    for d in 1..=self.max {
    //        println!("{d}");
    //
    //        let start_time = Instant::now();
    //
    //        let mut curr_degree_patterns = Vec::new();
    //        //let curr_degree_patterns = Arc::new(Mutex::new(Vec::new()));
    //
    //        for curr_pattern in prev_degree_patterns.clone() {
    //            for feature in features.clone() {
    //                for term in [true, false] {
    //                    let mut next_pattern = curr_pattern.clone();
    //                    let mut should_break = !next_pattern.insert((term, feature.to_string()));
    //                    if should_break {
    //                        continue;
    //                    }
    //                    for t in next_pattern.clone() {
    //                        let mut test_pattern = next_pattern.clone();
    //                        test_pattern.remove(&t);
    //                        if !prev_degree_patterns.contains(&test_pattern) {
    //                            should_break = true;
    //                            break;
    //                        }
    //                    }
    //                    if should_break {
    //                        continue;
    //                    }
    //                    let shapes = grouped_dfs.iter().map(DataFrame::shape).collect::<Vec<_>>();
    //                    //println!("{shapes:?}");
    //                    let counts: Vec<_> = grouped_dfs
    //                        .iter()
    //                        .map(|x| {
    //                            x.filter(&self.coverage(x, &next_pattern).into_iter().collect())
    //                                .unwrap()
    //                                .shape()
    //                                .0
    //                        })
    //                        .collect();
    //                    let tmp = counts.iter().map(|x| usize::from(*x >= 1)).sum::<usize>();
    //
    //                    //println!("0: {counts:?} {tmp}");
    //                    if tmp == 1 {
    //                        //println!("1: {counts:?} {tmp}");
    //                        for i in 0..counts.len() {
    //                            if counts[i] == 0 || shapes[i].0 == 0 {
    //                                continue;
    //                            }
    //                            //println!("{i}");
    //                            grouped_dfs[i] = grouped_dfs[i]
    //                                .filter(
    //                                    &self
    //                                        .coverage(&grouped_dfs[i], &next_pattern)
    //                                        .into_iter()
    //                                        .map(|x| !x)
    //                                        .collect(),
    //                                )
    //                                .unwrap();
    //
    //                            prime_patterns.push((self.labels[i].clone(), next_pattern));
    //                            break;
    //                        }
    //                    } else if tmp == 0 {
    //                        continue;
    //                    } else {
    //                        curr_degree_patterns.push(next_pattern);
    //                    }
    //                }
    //            }
    //        }
    //
    //        let duration = start_time.elapsed();
    //        println!("Time taken: {:.2} milliseconds", duration.as_millis());
    //
    //        let shapes = grouped_dfs
    //            //.lock()
    //            //.unwrap()
    //            .iter()
    //            .map(|x| x.shape().0)
    //            .collect::<Vec<_>>();
    //        println!("{shapes:?}");
    //
    //        if shapes.iter().sum::<usize>() == 0 {
    //            break;
    //        }
    //        if d == self.max {
    //            self.fallback_label = grouped_dfs
    //                .iter()
    //                .enumerate()
    //                .max_by(|(_, a), (_, b)| a.shape().0.cmp(&b.shape().0))
    //                .map(|(i, _)| self.labels[i].clone());
    //        }
    //        prev_degree_patterns = curr_degree_patterns;
    //    }
    //
    //    self.rules = prime_patterns;
    //
    //    Ok(())
    //}
}

impl RuleGenerator {
    fn coverage(&self, data: &DataFrame, pattern: &Pattern) -> Vec<bool> {
        let pattern_iter = pattern.iter();

        // Initialize a boolean mask for coverage with all true values
        let mut mask: Option<Vec<bool>> = None;

        // Iterate over each pattern element (value and column name)
        for (term, col_name) in pattern_iter {
            let col = data.column(col_name).unwrap();

            // Create a boolean mask for the current column by comparing its values with the pattern term
            let current_mask = col
                .bool()
                .unwrap() // Assuming all columns are boolean. Adjust if necessary.
                .into_iter()
                .map(|val| val.unwrap_or(false) == *term) // Replace any None with false
                .collect::<Vec<bool>>();

            // If it's the first iteration, set the mask to the current one
            if mask.is_none() {
                mask = Some(current_mask);
            } else {
                // Combine with previous mask (element-wise AND operation)
                mask = Some(
                    mask.unwrap()
                        .iter()
                        .zip(current_mask.iter())
                        .map(|(a, b)| *a && *b)
                        .collect(),
                );
            }
        }

        mask.unwrap_or_else(|| vec![true; data.height()]) // If no patterns, return full coverage (all true)
    }

    //fn coverage(&self, data: &DataFrame, pattern: &Pattern) -> Vec<bool> {
    //    let a = pattern
    //        .iter()
    //        .map(|(v, c)| {
    //            data.column(c)
    //                .unwrap()
    //                .iter()
    //                .map(|val| {
    //                    //println!("{val} = {v}");
    //                    val.cast(&DataType::Boolean).eq(&AnyValue::Boolean(*v))
    //                })
    //                .collect::<Vec<bool>>()
    //        })
    //        .collect::<Vec<_>>();
    //    let inner_length = a[0].len(); // Assuming all inner Vecs have the same length
    //    (0..inner_length)
    //        .map(|i| a.iter().all(|inner| inner[i]))
    //        .collect()
    //}

    fn divide_data(&self, data: &DataFrame, labels: &Series) -> Vec<DataFrame> {
        let mut grouped_dfs: Vec<DataFrame> = Vec::new();

        let unique_y = self.labels.clone();

        //println!("Debug1");
        // Iterate through unique y values
        for value in unique_y.iter() {
            // Filter the DataFrame rows where y equals the current unique value
            let mask = labels.iter().map(|x| x == value).collect();
            let sub_df = match data.filter(&mask) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("{e:?}");
                    continue;
                }
            };
            // Add the resulting sub DataFrame to the Vec
            grouped_dfs.push(sub_df);
        }
        grouped_dfs
    }
}
