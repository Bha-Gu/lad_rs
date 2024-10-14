use std::collections::HashSet;

use super::binarize::Binarizer;
use polars::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type Pattern = HashSet<(bool, String)>;

pub struct RuleGenerator {
    bin: Binarizer,
    max: usize,
    rules: Vec<(String, HashSet<(bool, String)>)>,
    labels: Vec<String>,
    fallback_label: Option<String>,
}

impl RuleGenerator {
    pub fn new(bin: &Binarizer, max: usize) -> Self {
        Self {
            bin: bin.clone(),
            max,
            rules: Vec::new(),
            labels: Vec::new(),
            fallback_label: None,
        }
    }

    pub fn get_rules(&self) -> Vec<(String, HashSet<(bool, String)>)> {
        self.rules.clone()
    }

    pub fn predict(&self, data: &DataFrame) -> PolarsResult<Vec<String>> {
        let data: DataFrame = self.bin.transform(data).unwrap();
        let mut predictions: Vec<Option<String>> = vec![None; data.height()];

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
        Ok(predictions
            .iter()
            .map(|x| {
                x.as_ref().map_or_else(
                    || self.fallback_label.clone().expect("fallback not found"),
                    std::clone::Clone::clone,
                )
            })
            .collect())
    }

    pub fn fit(&mut self, data: &DataFrame, labels: &Series) -> PolarsResult<()> {
        //println!("Debug0");
        let features = data.get_column_names();
        // Ensure y is categorical or can be grouped
        let unique_y = labels.unique_stable()?;
        self.labels = unique_y.iter().map(|x| x.to_string()).collect();
        // Initialize a Vec to hold the resulting DataFrames
        let mut grouped_dfs: Vec<DataFrame> = self.divide_data(data, labels);
        //println!("Debug2");

        self.fallback_label = grouped_dfs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.shape().0.cmp(&b.shape().0))
            .map(|(i, _)| self.labels[i].clone());

        // Perform further operations with grouped_dfs if necessary
        let mut prime_patterns: Vec<(String, Pattern)> = Vec::new();
        let mut prev_degree_patterns: Vec<Pattern> = vec![HashSet::new()];

        if self.max > features.len() || self.max == 0 {
            self.max = features.len();
        }
        //let grouped_dfs = Arc::new(Mutex::new(grouped_dfs));
        //let prime_patterns = Arc::new(Mutex::new(prime_patterns));

        for d in 1..=self.max {
            println!("{d}");

            let start_time = Instant::now();

            let mut curr_degree_patterns = Vec::new();
            //let curr_degree_patterns = Arc::new(Mutex::new(Vec::new()));

            //prev_degree_patterns
            //    .par_iter()
            //    .cloned()
            //    .for_each(|curr_pattern| {
            //        features.par_iter().for_each(|feature| {
            //            [true, false].par_iter().for_each(|&term| {
            //                let mut next_pattern = curr_pattern.clone();
            //                let should_break = !next_pattern.insert((term, feature.to_string()));
            //                if should_break {
            //                    return;
            //                }
            //
            //                // Check if subpatterns exist in prev_degree_patterns
            //                let any_break = next_pattern.iter().any(|t| {
            //                    let mut test_pattern = next_pattern.clone();
            //                    test_pattern.remove(t);
            //                    !prev_degree_patterns.contains(&test_pattern)
            //                });
            //
            //                if any_break {
            //                    return;
            //                }
            //
            //                // Lock for access to grouped_dfs
            //                let shapes: Vec<_> = grouped_dfs
            //                    .lock()
            //                    .unwrap()
            //                    .iter()
            //                    .map(DataFrame::shape)
            //                    .collect();
            //
            //                let counts: Vec<_> = grouped_dfs
            //                    .lock()
            //                    .unwrap()
            //                    .iter()
            //                    .map(|df| {
            //                        df.filter(
            //                            &self.coverage(df, &next_pattern).into_iter().collect(),
            //                        )
            //                        .unwrap()
            //                        .shape()
            //                        .0
            //                    })
            //                    .collect();
            //
            //                let tmp = counts
            //                    .par_iter()
            //                    .map(|&x| usize::from(x >= 1))
            //                    .sum::<usize>();
            //
            //                if tmp == 1 {
            //                    for (i, count) in counts.iter().enumerate() {
            //                        if *count == 0 || shapes[i].0 == 0 {
            //                            continue;
            //                        }
            //
            //                        let mut grouped_dfs_locked = grouped_dfs.lock().unwrap();
            //
            //                        grouped_dfs_locked[i] = grouped_dfs_locked[i]
            //                            .filter(
            //                                &self
            //                                    .coverage(&grouped_dfs_locked[i], &next_pattern)
            //                                    .into_iter()
            //                                    .map(|x| !x)
            //                                    .collect(),
            //                            )
            //                            .unwrap();
            //                        drop(grouped_dfs_locked);
            //
            //                        prime_patterns
            //                            .lock()
            //                            .unwrap()
            //                            .push((self.labels[i].clone(), next_pattern.clone()));
            //
            //                        break;
            //                    }
            //                } else if tmp == 0 {
            //                    return;
            //                } else {
            //                    curr_degree_patterns.lock().unwrap().push(next_pattern);
            //                }
            //            });
            //        });
            //    });

            for curr_pattern in prev_degree_patterns.clone() {
                for feature in features.clone() {
                    for term in [true, false] {
                        let mut next_pattern = curr_pattern.clone();
                        let mut should_break = !next_pattern.insert((term, feature.to_string()));
                        if should_break {
                            continue;
                        }
                        for t in next_pattern.clone() {
                            let mut test_pattern = next_pattern.clone();
                            test_pattern.remove(&t);
                            if !prev_degree_patterns.contains(&test_pattern) {
                                should_break = true;
                                break;
                            }
                        }
                        if should_break {
                            continue;
                        }
                        let shapes = grouped_dfs.iter().map(DataFrame::shape).collect::<Vec<_>>();
                        //println!("{shapes:?}");
                        let counts: Vec<_> = grouped_dfs
                            .iter()
                            .map(|x| {
                                x.filter(&self.coverage(x, &next_pattern).into_iter().collect())
                                    .unwrap()
                                    .shape()
                                    .0
                            })
                            .collect();
                        let tmp = counts.iter().map(|x| usize::from(*x >= 1)).sum::<usize>();

                        //println!("0: {counts:?} {tmp}");
                        if tmp == 1 {
                            //println!("1: {counts:?} {tmp}");
                            for i in 0..counts.len() {
                                if counts[i] == 0 || shapes[i].0 == 0 {
                                    continue;
                                }
                                //println!("{i}");
                                grouped_dfs[i] = grouped_dfs[i]
                                    .filter(
                                        &self
                                            .coverage(&grouped_dfs[i], &next_pattern)
                                            .into_iter()
                                            .map(|x| !x)
                                            .collect(),
                                    )
                                    .unwrap();

                                prime_patterns.push((self.labels[i].clone(), next_pattern));
                                break;
                            }
                        } else if tmp == 0 {
                            continue;
                        } else {
                            curr_degree_patterns.push(next_pattern);
                        }
                    }
                }
            }

            let duration = start_time.elapsed();
            println!("Time taken: {:.2} milliseconds", duration.as_millis());

            let shapes = grouped_dfs
                //.lock()
                //.unwrap()
                .iter()
                .map(|x| x.shape().0)
                .collect::<Vec<_>>();
            println!("{shapes:?}");

            if shapes.iter().sum::<usize>() == 0 {
                break;
            }
            if d == self.max {
                self.fallback_label = grouped_dfs
                    //.lock()
                    //.unwrap()
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.shape().0.cmp(&b.shape().0))
                    .map(|(i, _)| self.labels[i].clone());
            }
            prev_degree_patterns = curr_degree_patterns
            //prev_degree_patterns = Arc::try_unwrap(curr_degree_patterns)
            //    .unwrap()
            //    .into_inner()
            //    .unwrap();
        }

        self.rules = prime_patterns;
        //self.rules.clone_from(&prime_patterns.lock().unwrap());

        Ok(())
    }
}

impl RuleGenerator {
    fn coverage(&self, data: &DataFrame, pattern: &Pattern) -> Vec<bool> {
        let a = pattern
            .iter()
            .map(|(v, c)| {
                data.column(c)
                    .unwrap()
                    .iter()
                    .map(|val| {
                        //println!("{val} = {v}");
                        val.cast(&DataType::Boolean).eq(&AnyValue::Boolean(*v))
                    })
                    .collect::<Vec<bool>>()
            })
            .collect::<Vec<_>>();
        let inner_length = a[0].len(); // Assuming all inner Vecs have the same length
        (0..inner_length)
            .map(|i| a.iter().all(|inner| inner[i]))
            .collect()
    }

    fn divide_data(&self, data: &DataFrame, labels: &Series) -> Vec<DataFrame> {
        let mut grouped_dfs: Vec<DataFrame> = Vec::new();

        let unique_y = self.labels.clone();

        //println!("Debug1");
        // Iterate through unique y values
        for value in &unique_y {
            // Filter the DataFrame rows where y equals the current unique value
            let mask = labels.iter().map(|x| x.to_string() == *value).collect();
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
