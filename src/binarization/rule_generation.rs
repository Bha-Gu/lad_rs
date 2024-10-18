use std::collections::HashSet;

use super::binarize::Binarizer;
use polars::prelude::*;
use std::time::Instant;

use rayon::prelude::*;

use notify_rust::Notification;

type Pattern = HashSet<(bool, String)>;

#[derive(Clone)]
pub struct RuleGenerator {
    bin: Binarizer,
    max: usize,
    rules: Vec<(usize, HashSet<(bool, String)>)>,
    labels: Series,
    fallback_label: usize,
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
        }
    }

    #[must_use]
    pub fn get_rules(&self) -> Vec<(usize, HashSet<(bool, String)>)> {
        self.rules.clone()
    }

    pub fn predict(&self, data: &DataFrame) -> PolarsResult<Series> {
        let data: DataFrame = self.bin.transform(data)?;
        let mut predictions: Vec<Option<usize>> = vec![None; data.height()];

        for (label, pattern) in &self.rules {
            let coverage = self.coverage(&data, pattern);

            // Iterate over each index in the coverage vector (a)
            for (&is_covered, prediction) in coverage?.iter().zip(predictions.iter_mut()) {
                // If the current value is true and the result is None for this index, set the label
                if is_covered && prediction.is_none() {
                    *prediction = Some(*label); // Use .clone() since label is a reference
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
        let unique_y = labels.unique_stable()?;
        self.labels = unique_y;

        // Divide data into groups based on the labels
        let mut grouped_dfs: Vec<DataFrame> = self.divide_data(data, labels);

        self.fallback_label = grouped_dfs
            .iter()
            .enumerate()
            .max_by_key(|(_, df)| df.shape().0)
            .map(|(i, _)| i)
            .unwrap_or_default();

        let mut prime_patterns: Vec<(usize, Pattern)> = Vec::new();
        let mut prev_degree_patterns: Vec<Pattern> = vec![HashSet::new()];

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

        for d in 1..=max_features {
            println!("{d}");
            let start_time = Instant::now();
            let mut curr_degree_patterns = Vec::new();

            let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
            println!("{remaining_shapes:?}");

            for (pattern_idx, curr_pattern) in prev_degree_patterns.iter().enumerate() {
                // Send notification for pattern progress
                handle.body(&format!(
                    "Processing pattern: {}/{} at depth {}",
                    pattern_idx + 1,
                    prev_degree_patterns.len(),
                    d
                ));

                handle.update();

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

                        let counts: Vec<usize> = grouped_dfs
                            .par_iter()
                            .map(|df| -> PolarsResult<_> {
                                Ok(self
                                    .coverage(df, &next_pattern)?
                                    .into_iter()
                                    .filter(|&x| x)
                                    .count())
                            })
                            .collect::<PolarsResult<Vec<_>>>()?;

                        let tmp = counts.iter().filter(|&&x| x >= 1).count();

                        if tmp == 1 {
                            for (i, &count) in counts.iter().enumerate() {
                                if count == 0 || grouped_dfs[i].shape().0 == 0 {
                                    continue;
                                }

                                grouped_dfs[i] = grouped_dfs[i].filter(
                                    &self
                                        .coverage(&grouped_dfs[i], &next_pattern)?
                                        .into_iter()
                                        .map(|x| !x)
                                        .collect(),
                                )?;

                                prime_patterns.push((i, next_pattern));

                                break;
                            }
                        } else if tmp != 0 {
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
                    .map(|(i, _)| i)
                    .unwrap_or_default();
            }

            prev_degree_patterns = curr_degree_patterns;
        }

        self.rules = prime_patterns;
        Ok(())
    }
}

impl RuleGenerator {
    fn coverage(&self, data: &DataFrame, pattern: &Pattern) -> PolarsResult<Vec<bool>> {
        let pattern_iter = pattern.iter();

        // Initialize a boolean mask for coverage with all true values
        let mut mask = vec![true; data.height()]; // Start with all true values

        // Iterate over each pattern element (value and column name)
        for (term, col_name) in pattern_iter {
            let col = data.column(col_name)?;

            // Create a boolean mask for the current column by comparing its values with the pattern term
            let current_mask = col
                .bool()?
                .into_iter()
                .map(|val| val.unwrap_or(false) == *term) // Replace any None with false
                .collect::<Vec<bool>>();

            // If it's the first iteration, set the mask to the current one
            mask.iter_mut()
                .zip(current_mask)
                .for_each(|(a, b)| *a = *a && b);
        }

        Ok(mask) // If no patterns, return full coverage (all true)
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
