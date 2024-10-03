use std::collections::HashSet;

use polars::prelude::*;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use pyo3_polars::{PyDataFrame, PySeries};

use super::binarize::Binarizer;

#[pyclass]
pub struct RuleGenerator {
    bin: Binarizer,
    max: usize,
    rules: Vec<Vec<HashSet<(bool, String)>>>,
}

#[pymethods]
impl RuleGenerator {
    #[new]
    pub fn new(bin: &Binarizer, max: usize) -> Self {
        Self {
            bin: bin.clone(),
            max,
            rules: Vec::new(),
        }
    }

    pub fn get_rules(&self) -> Vec<Vec<HashSet<(bool, String)>>> {
        return self.rules.clone();
    }

    pub fn fit(&mut self, X: PyDataFrame, y: PySeries) -> PyResult<()> {
        // Convert PyDataFrame and PySeries to Polars DataFrame and Series
        println!("Debug0");
        let df: DataFrame = X.into(); // Extract the Polars DataFrame
        let y_series: Series = y.into(); // Extract the Polars Series
        let features = df.get_column_names();
        // Ensure y is categorical or can be grouped
        let unique_y = y_series.unique_stable().unwrap();
        let unique_y = unique_y.iter().collect::<Vec<_>>();

        // Initialize a Vec to hold the resulting DataFrames
        let mut grouped_dfs: Vec<DataFrame> = Vec::new();

        println!("Debug1");
        // Iterate through unique y values
        for value in unique_y {
            // Filter the DataFrame rows where y equals the current unique value
            println!("{value:?}");
            let mask = y_series.iter().map(|x| x == value).collect();
            let sub_df = match df.filter(&mask) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("{e:?}");
                    continue;
                }
            };

            // Add the resulting sub DataFrame to the Vec
            grouped_dfs.push(sub_df);
        }

        println!("Debug2");
        type Pattern = HashSet<(bool, String)>;

        // Perform further operations with grouped_dfs if necessary
        let mut prime_patterns = vec![Vec::<Pattern>::new(); grouped_dfs.len()];
        let mut prev_degree_patterns: Vec<Pattern> = vec![HashSet::new()];

        if self.max > features.len() || self.max == 0 {
            self.max = features.len();
        }

        for _ in 1..=self.max {
            let mut curr_degree_patterns: Vec<Pattern> = vec![];
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
                        let mask = |x: &DataFrame| -> Vec<bool> {
                            let a = next_pattern
                                .iter()
                                .map(|(v, c)| {
                                    x.column(&c)
                                        .unwrap()
                                        .iter()
                                        .map(|val| {
                                            val.cast(&DataType::Boolean).eq(&AnyValue::Boolean(*v))
                                        })
                                        .collect::<Vec<bool>>()
                                })
                                .collect::<Vec<_>>();
                            let inner_length = a[0].len(); // Assuming all inner Vecs have the same length
                            (0..inner_length)
                                .map(|i| a.iter().all(|inner| inner[i]))
                                .collect()
                        };
                        let counts: Vec<_> = grouped_dfs
                            .iter()
                            .map(|x| x.filter(&mask(x).into_iter().collect()).unwrap().shape().0)
                            .collect();
                        println!("{counts:?}");
                        let tmp = counts
                            .iter()
                            .map(|x| if *x > 1 { 1 } else { 0 })
                            .sum::<usize>();
                        if tmp == 1 {
                            for i in 0..counts.len() {
                                if counts[i] == 0 {
                                    continue;
                                }
                                grouped_dfs[i] = grouped_dfs[i]
                                    .filter(&mask(&grouped_dfs[i]).into_iter().collect())
                                    .unwrap();

                                prime_patterns[i].push(next_pattern);
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

            prev_degree_patterns = curr_degree_patterns;
        }

        self.rules = prime_patterns;
        Ok(())
    }
}
