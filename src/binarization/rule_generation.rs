use std::collections::HashSet;

use super::binarize::Binarizer;
use polars::prelude::*;
use std::time::Instant;

use rayon::prelude::*;

use notify_rust::Notification;

type Pattern = HashSet<(bool, usize)>;

#[derive(Clone)]
pub struct RuleGenerator {
    bin: Binarizer,
    max: usize,
    rules: Vec<(usize, HashSet<(bool, usize)>, f64)>,
    labels: Series,
    fallback_label: usize,
    features: Vec<String>,
    remaining: Vec<usize>,
}

impl RuleGenerator {
    #[must_use]
    pub fn new(bin: &Binarizer, max: usize) -> Self {
        Self {
            bin: bin.clone(),
            max,
            rules: Vec::new(),
            labels: Series::new("tmp".into(), [0]),
            fallback_label: 0,
            features: Vec::new(),
            remaining: Vec::new(),
        }
    }

    #[must_use]
    pub fn get_rules(&self) -> Vec<(usize, Vec<(bool, String)>)> {
        self.rules
            .iter()
            .map(|(a, x, _)| {
                (
                    *a,
                    x.iter()
                        .map(|(b, i)| (*b, self.features[*i].clone()))
                        .collect(),
                )
            })
            .collect()
    }

    pub fn predict(&self, data: &DataFrame) -> PolarsResult<Series> {
        let data: DataFrame = self.bin.transform(data)?;
        let mut predictions: Vec<Option<usize>> = vec![None; data.height()];

        for (label, pattern, _size) in &self.rules {
            let coverage = self.par_coverage(&data, pattern);

            // Iterate over each index in the coverage vector (a)
            for (&is_covered, prediction) in coverage?.iter().zip(predictions.iter_mut()) {
                if prediction.is_none() {
                    if is_covered == 1. {
                        *prediction = Some(*label);
                    }
                }
            }
        }

        Ok(Series::new(
            "Predictions".into(),
            predictions
                .iter()
                .map(|x| {
                    x.as_ref().map_or_else(
                        || {
                            self.labels
                                .get(self.fallback_label)
                                .unwrap_or(AnyValue::Null)
                        },
                        |x| self.labels.get(*x).unwrap_or(AnyValue::Null),
                    )
                })
                .collect::<Vec<_>>(),
        ))
    }

    pub fn predict_exact(&self, data: &DataFrame) -> PolarsResult<Series> {
        let data: DataFrame = self.bin.transform(data)?;
        let mut predictions: Vec<Option<usize>> = vec![None; data.height()];

        for (label, pattern, _size) in &self.rules {
            let coverage = self.par_coverage(&data, pattern);

            // Iterate over each index in the coverage vector (a)
            for (&is_covered, prediction) in coverage?.iter().zip(predictions.iter_mut()) {
                if prediction.is_none() {
                    if is_covered == 1. {
                        *prediction = Some(*label);
                    }
                }
            }
        }

        Ok(Series::new(
            "Predictions".into(),
            predictions
                .iter()
                .map(|x| {
                    x.as_ref().map_or_else(
                        || AnyValue::Null,
                        |x| self.labels.get(*x).unwrap_or(AnyValue::Null),
                    )
                })
                .collect::<Vec<_>>(),
        ))
    }

    pub fn predict_fuzzy(&self, data: &DataFrame) -> PolarsResult<Series> {
        let data: DataFrame = self.bin.transform(data)?;
        let mut predictions: Vec<Option<usize>> = vec![None; data.height()];

        let mut confidence = vec![vec![0.0_f64; self.labels.len()]; data.height()];

        for (label, pattern, size) in &self.rules {
            let coverage = self.par_coverage(&data, pattern);

            // Iterate over each index in the coverage vector (a)
            for (&is_covered, (prediction, conf)) in coverage?
                .iter()
                .zip(predictions.iter_mut().zip(confidence.iter_mut()))
            {
                if prediction.is_none() {
                    if is_covered == 1. {
                        *prediction = Some(*label);
                    }
                    conf[*label] = (conf[*label]) + is_covered * (*size);
                }
            }
        }

        predictions
            .iter_mut()
            .zip(confidence.iter())
            .map(|(a, b)| {
                let arg_max = b
                    .iter()
                    .enumerate()
                    .max_by(|(_, val1), (_, val2)| val1.partial_cmp(val2).unwrap())
                    .map(|(index, _)| index);
                (a, arg_max)
            })
            .for_each(|(a, b)| {
                if a.is_none() {
                    *a = b; // Update with Some(index of max value)
                }
            });

        Ok(Series::new(
            "Predictions".into(),
            predictions
                .iter()
                .map(|x| {
                    x.as_ref().map_or_else(
                        || {
                            self.labels
                                .get(self.fallback_label)
                                .unwrap_or(AnyValue::Null)
                        },
                        |x| self.labels.get(*x).unwrap_or(AnyValue::Null),
                    )
                })
                .collect::<Vec<_>>(),
        ))
    }

    pub fn fit(&mut self, data: &DataFrame, labels: &Series) -> PolarsResult<()> {
        let features = data.get_column_names();
        self.features = features.iter().map(|&x| x.to_string()).collect();
        self.labels = labels.unique_stable()?;

        // Divide data into groups based on the labels
        let mut grouped_dfs: Vec<DataFrame> = self.divide_data(data, labels);
        self.fallback_label = grouped_dfs
            .iter()
            .enumerate()
            .max_by_key(|(_, df)| df.shape().0)
            .map(|(i, _)| i)
            .unwrap_or_default();

        let max_features = if self.max > features.len() || self.max == 0 {
            features.len()
        } else {
            self.max
        };

        let mut handle = Notification::new()
            .summary("Task Progress")
            .body("Task is starting")
            .show()
            .unwrap();

        let mut prev_degree_patterns: Vec<Pattern> = vec![HashSet::new()];

        let mut prime_patterns: Vec<(usize, Pattern, f64)> = Vec::new();

        for d in 1..=max_features {
            println!("{d}");
            let start_time = Instant::now();
            let mut curr_degree_patterns = Vec::new();

            let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
            println!("{remaining_shapes:?}");

            let length = prev_degree_patterns.len();

            for (pattern_idx, curr_pattern) in prev_degree_patterns.into_iter().enumerate() {
                handle.body(&format!(
                    "processing pattern: {}/{} at depth {}",
                    pattern_idx + 1,
                    length,
                    d
                ));
                handle.update();

                for idx in 0..features.len() {
                    for &term in &[true, false] {
                        let mut next_pattern = curr_pattern.clone();
                        if !next_pattern.insert((term, idx)) {
                            continue; // Skip if pattern already exists
                        }
                        //
                        //// Check if the next pattern is a valid extension
                        //if next_pattern.par_iter().any(|t| {
                        //    let test_pattern: HashSet<_> =
                        //        next_pattern.iter().filter(|&&x| x != *t).cloned().collect();
                        //    !prev_degree_patterns.contains(&test_pattern)
                        //}) {
                        //    continue;
                        //}

                        // Compute counts in parallel
                        let counts: Vec<usize> = grouped_dfs
                            .par_iter()
                            .map(|df| {
                                self.coverage(df, &next_pattern)
                                    .map(|mask| mask.into_iter().filter(|&x| x).count())
                            })
                            .collect::<PolarsResult<Vec<_>>>()?;

                        let tmp = counts.iter().filter(|&&x| x >= 1).count();

                        if tmp == 1 {
                            for (i, count) in counts.into_iter().enumerate() {
                                if count == 0 || grouped_dfs[i].shape().0 == 0 {
                                    continue;
                                }

                                let len = grouped_dfs[i].shape().0;

                                // Filter the DataFrame based on the coverage mask
                                let mask = self.coverage(&grouped_dfs[i], &next_pattern)?;
                                grouped_dfs[i] = grouped_dfs[i]
                                    .filter(&mask.into_iter().map(|x| !x).collect())?;

                                prime_patterns.push((
                                    i,
                                    next_pattern,
                                    if len > 0 {
                                        count as f64 / len as f64
                                    } else {
                                        0.0
                                    },
                                ));
                                break; // Break after first match
                            }
                        } else if tmp != 0 {
                            curr_degree_patterns.push(next_pattern);
                        }
                    }
                }
            }

            loop {
                let mut max_score_at = ((0, 0), 0);

                for i in (0..curr_degree_patterns.len()).rev() {
                    let pattern = &curr_degree_patterns[i];
                    handle.body(&format!(
                        "processing best pattern: {}/{} at depth {}, max: {}",
                        curr_degree_patterns.len() - i,
                        curr_degree_patterns.len(),
                        d,
                        max_score_at.0 .1
                    ));
                    handle.update();

                    let counts: Vec<usize> = grouped_dfs
                        .par_iter()
                        .map(|df| {
                            self.coverage(df, &pattern)
                                .map(|mask| mask.into_iter().filter(|&x| x).count())
                        })
                        .collect::<PolarsResult<Vec<_>>>()?;

                    let tmp = counts.iter().filter(|&&x| x >= 1).count();

                    if tmp == 1 {
                        let score1 = counts.iter().max().unwrap();

                        max_score_at = (
                            (counts.iter().position(|&x| x == *score1).unwrap(), *score1),
                            i,
                        );
                        let len = grouped_dfs[max_score_at.0 .0].shape().0;
                        let mask = self.coverage(&grouped_dfs[max_score_at.0 .0], &pattern)?;

                        grouped_dfs[max_score_at.0 .0] = grouped_dfs[max_score_at.0 .0]
                            .filter(&mask.into_iter().map(|x| !x).collect())?;

                        prime_patterns.push((
                            max_score_at.0 .0,
                            curr_degree_patterns[max_score_at.1].clone(),
                            *score1 as f64 / len as f64,
                        ));
                        curr_degree_patterns.swap_remove(i); // Remove the current pattern
                    }
                }

                handle.body(&format!("Max Score: {} at depth {}", max_score_at.0 .1, d));
                handle.update();

                if max_score_at.0 .1 == 0 {
                    break;
                }
                //let tmp = curr_degree_patterns[max_score_at.1].clone();
                //
                //let mask = self.coverage(&grouped_dfs[max_score_at.0 .0], &tmp)?;
                //grouped_dfs[max_score_at.0 .0] = grouped_dfs[max_score_at.0 .0]
                //    .filter(&mask.into_iter().map(|x| !x).collect())?;
                //
                //prime_patterns.push((
                //    max_score_at.0 .0,
                //    curr_degree_patterns[max_score_at.1].clone(),
                //));
                //
                //curr_degree_patterns.retain(|idx| *idx != tmp);
            }

            let duration = start_time.elapsed();
            println!("Time taken: {:.2} milliseconds", duration.as_millis());

            // Check remaining shapes
            let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
            println!("{remaining_shapes:?}");

            self.remaining = remaining_shapes.clone();

            if remaining_shapes.into_iter().sum::<usize>() == 0 {
                break;
            }

            if d == max_features {
                self.fallback_label = grouped_dfs
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, df)| df.shape().0)
                    .map(|(i, _)| i)
                    .unwrap_or_default();
            }

            prev_degree_patterns = curr_degree_patterns;
        }

        self.rules = prime_patterns;
        Ok(())
    }

    //pub fn fit(&mut self, data: &DataFrame, labels: &Series) -> PolarsResult<()> {
    //    let features = data.get_column_names();
    //    self.features = features.iter().map(|&x| (*x).to_string()).collect();
    //    let unique_y = labels.unique_stable()?;
    //    self.labels = unique_y;
    //
    //    // Divide data into groups based on the labels
    //    let mut grouped_dfs: Vec<DataFrame> = self.divide_data(data, labels);
    //
    //    self.fallback_label = grouped_dfs
    //        .iter()
    //        .enumerate()
    //        .max_by_key(|(_, df)| df.shape().0)
    //        .map(|(i, _)| i)
    //        .unwrap_or_default();
    //
    //    let mut prime_patterns: Vec<(usize, Pattern)> = Vec::new();
    //    let mut prev_degree_patterns: Vec<Pattern> = vec![HashSet::new()];
    //
    //    let max_features = if self.max > features.len() || self.max == 0 {
    //        features.len()
    //    } else {
    //        self.max
    //    };
    //
    //    let mut handle = Notification::new()
    //        .summary("Task Progress")
    //        .body("Task is starting")
    //        .show()
    //        .unwrap();
    //
    //    for d in 1..=max_features {
    //        println!("{d}");
    //        let start_time = Instant::now();
    //        let mut curr_degree_patterns = Vec::new();
    //
    //        let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
    //        println!("{remaining_shapes:?}");
    //
    //        for (pattern_idx, curr_pattern) in prev_degree_patterns.iter().enumerate() {
    //            // Send notification for pattern progress
    //            handle.body(&format!(
    //                "Processing pattern: {}/{} at depth {}",
    //                pattern_idx + 1,
    //                prev_degree_patterns.len(),
    //                d
    //            ));
    //
    //            handle.update();
    //
    //            for idx in 0..features.len() {
    //                for term in [true, false] {
    //                    let mut next_pattern = curr_pattern.clone();
    //                    if !next_pattern.insert((term, idx)) {
    //                        continue;
    //                    }
    //
    //                    if next_pattern.par_iter().any(|t| {
    //                        let test_pattern: HashSet<_> =
    //                            next_pattern.iter().filter(|&x| *x != *t).cloned().collect();
    //                        !prev_degree_patterns.contains(&test_pattern)
    //                    }) {
    //                        continue;
    //                    }
    //
    //                    let counts: Vec<usize> = grouped_dfs
    //                        .par_iter()
    //                        .map(|df| -> PolarsResult<_> {
    //                            Ok(self
    //                                .coverage(df, &next_pattern)?
    //                                .into_iter()
    //                                .filter(|&x| x)
    //                                .count())
    //                        })
    //                        .collect::<PolarsResult<Vec<_>>>()?;
    //
    //                    let tmp = counts.iter().fold(0, |acc, &x| acc + (x >= 1) as usize);
    //
    //                    if tmp == 1 {
    //                        for (i, &count) in counts.iter().enumerate() {
    //                            if count == 0 || grouped_dfs[i].shape().0 == 0 {
    //                                continue;
    //                            }
    //
    //                            grouped_dfs[i] = grouped_dfs[i].filter(
    //                                &self
    //                                    .coverage(&grouped_dfs[i], &next_pattern)?
    //                                    .into_iter()
    //                                    .map(|x| !x)
    //                                    .collect(),
    //                            )?;
    //
    //                            prime_patterns.push((i, next_pattern));
    //
    //                            break;
    //                        }
    //                    } else if tmp != 0 {
    //                        curr_degree_patterns.push(next_pattern);
    //                    }
    //                }
    //            }
    //        }
    //
    //                  let duration = start_time.elapsed();
    //        println!("Time taken: {:.2} milliseconds", duration.as_millis());
    //
    //        let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
    //        println!("{remaining_shapes:?}");
    //
    //        if remaining_shapes.iter().sum::<usize>() == 0 {
    //            break;
    //        }
    //
    //        if d == max_features {
    //            self.fallback_label = grouped_dfs
    //                .iter()
    //                .enumerate()
    //                .max_by_key(|(_, df)| df.shape().0)
    //                .map(|(i, _)| i)
    //                .unwrap_or_default();
    //        }
    //
    //        prev_degree_patterns = curr_degree_patterns;
    //    }
    //
    //    self.rules = prime_patterns;
    //    Ok(())
    //}
}

impl RuleGenerator {
    fn coverage(&self, data: &DataFrame, pattern: &Pattern) -> PolarsResult<Vec<bool>> {
        let pattern_iter = pattern.iter();

        // Initialize a boolean mask for coverage with all true values
        let mut mask = vec![true; data.height()]; // Start with all true values

        // Iterate over each pattern element (value and column name)
        for (term, col_name) in pattern_iter {
            let col = data.column(&self.features[*col_name])?;

            // Create a boolean mask for the current column by comparing its values with the pattern term
            let current_mask = col
                .bool()?
                .into_iter()
                .map(|val| val.unwrap_or(false) == *term);

            // If it's the first iteration, set the mask to the current one
            mask.iter_mut()
                .zip(current_mask)
                .for_each(|(a, b)| *a = *a && b);
        }

        Ok(mask) // If no patterns, return full coverage (all true)
    }

    fn par_coverage(&self, data: &DataFrame, pattern: &Pattern) -> PolarsResult<Vec<f64>> {
        let pattern_iter = pattern.into_iter();

        // Initialize a boolean mask for coverage with all true values
        let mut mask = vec![0.; data.height()]; // Start with all true values

        let mut len = 0.;
        // Iterate over each pattern element (value and column name)
        for (term, col_name) in pattern_iter {
            len += 1.;

            let col = data.column(&self.features[*col_name])?;

            // Create a boolean mask for the current column by comparing its values with the pattern term
            let current_mask = col
                .bool()?
                .into_iter()
                .map(|val| val.unwrap_or(false) == *term);

            // If it's the first iteration, set the mask to the current one
            mask.iter_mut().zip(current_mask).for_each(|(a, b)| {
                if b {
                    *a = *a + 1.
                }
            });
        }

        mask.iter_mut().for_each(|a| {
            if len != 0. {
                *a = *a / len
            }
        });

        Ok(mask)
    }

    fn divide_data(&self, data: &DataFrame, labels: &Series) -> Vec<DataFrame> {
        let mut grouped_dfs: Vec<DataFrame> = Vec::new();

        // Iterate through unique y values
        for value in self.labels.iter() {
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
