use std::collections::HashSet;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};

use super::binarize::Binarizer;

#[pyclass]
pub struct RuleGenerator {
    bin: Binarizer,
    max: usize,
    rules: Vec<(String, HashSet<(bool, String)>)>,
    labels: Vec<String>,
}

#[pymethods]
impl RuleGenerator {
    #[new]
    pub fn new(bin: &Binarizer, max: usize) -> Self {
        Self {
            bin: bin.clone(),
            max,
            rules: Vec::new(),
            labels: Vec::new(),
        }
    }

    pub fn get_rules(&self) -> Vec<(String, HashSet<(bool, String)>)> {
        self.rules.clone()
    }

    pub fn fit(&mut self, data: PyDataFrame, labels: PySeries) {
        //println!("Debug0");
        let df: DataFrame = data.into(); // Extract the Polars DataFrame
        let y_series: Series = labels.into(); // Extract the Polars Series
        let features = df.get_column_names();
        // Ensure y is categorical or can be grouped
        let unique_y = y_series.unique_stable().unwrap();
        self.labels = unique_y.iter().map(|x| format!("{x}")).collect();
        // Initialize a Vec to hold the resulting DataFrames
        let mut grouped_dfs: Vec<DataFrame> = Vec::new();

        //println!("Debug1");
        // Iterate through unique y values
        for value in unique_y.iter() {
            // Filter the DataFrame rows where y equals the current unique value
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

        //println!("Debug2");
        type Pattern = HashSet<(bool, String)>;

        // Perform further operations with grouped_dfs if necessary
        let mut prime_patterns: Vec<(String, HashSet<(bool, String)>)> = Vec::new();
        let mut prev_degree_patterns: Vec<Pattern> = vec![HashSet::new()];

        if self.max > features.len() || self.max == 0 {
            self.max = features.len();
        }

        for d in 1..=self.max {
            println!("{d}");

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
                                    x.column(c)
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
                        };
                        let shapes = grouped_dfs.iter().map(DataFrame::shape).collect::<Vec<_>>();
                        //println!("{shapes:?}");
                        let counts: Vec<_> = grouped_dfs
                            .iter()
                            .map(|x| x.filter(&mask(x).into_iter().collect()).unwrap().shape().0)
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
                                        &mask(&grouped_dfs[i]).into_iter().map(|x| !x).collect(),
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

            let shapes = grouped_dfs.iter().map(|x| x.shape().0).collect::<Vec<_>>();

            println!("{shapes:?}");

            if shapes.iter().sum::<usize>() == 0 {
                break;
            }

            prev_degree_patterns = curr_degree_patterns;
        }

        self.rules = prime_patterns;
    }
}
