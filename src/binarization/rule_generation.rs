use std::{collections::HashSet, f64::consts::E};

use super::binarize::Binarizer;
use itertools::Itertools;
use polars::prelude::*;
use std::time::Instant;

use rayon::prelude::*;

use notify_rust::Notification;

type Pattern = HashSet<(bool, usize)>;

const STEP_SIZE: usize = 100000;

#[derive(Clone)]
pub struct RuleGenerator {
    bin: Binarizer,
    max: usize,
    rules: Vec<(usize, Pattern, f64, usize)>,
    labels: Series,
    fallback_label: usize,
    features: Vec<String>,
    remaining: Vec<usize>,
    deep: usize,
    decay: Vec<f64>,
    retention: Vec<f64>,
}

impl RuleGenerator {
    #[must_use]
    pub fn new(
        bin: &Binarizer,
        max: usize,
        deep: usize,
        decay: Vec<f64>,
        retention: Vec<f64>,
    ) -> Self {
        Self {
            bin: bin.clone(),
            max,
            rules: Vec::new(),
            labels: Series::new("tmp".into(), [0]),
            fallback_label: 0,
            features: Vec::new(),
            remaining: Vec::new(),
            deep,
            decay,
            retention,
        }
    }

    #[must_use]
    pub fn get_rules(&self) -> Vec<(usize, Vec<(bool, String)>, usize)> {
        self.rules
            .iter()
            .map(|(a, x, _, t)| {
                (
                    *a,
                    x.iter()
                        .map(|(b, i)| (*b, self.features[*i].clone()))
                        .collect(),
                    *t,
                )
            })
            .collect()
    }

    pub fn predict(&self, data: &DataFrame) -> PolarsResult<Series> {
        let data: DataFrame = self.bin.transform(data)?;
        let mut predictions: Vec<Option<usize>> = vec![None; data.height()];

        for (label, pattern, _size, _) in &self.rules {
            let coverage = self.par_coverage(&data, pattern);

            // Iterate over each index in the coverage vector (a)
            for (&is_covered, prediction) in coverage?.iter().zip(predictions.iter_mut()) {
                if prediction.is_none() && is_covered == 1. {
                    *prediction = Some(*label);
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

    pub fn predict_exact(&self, data: &DataFrame) -> PolarsResult<(Series, Series, DataFrame)> {
        let data: DataFrame = self.bin.transform(data)?;
        let mut predictions: Vec<Option<usize>> = vec![None; data.height()];
        let mut rule_index: Vec<Option<usize>> = vec![None; data.height()];

        for (idx, (label, pattern, _size, _)) in self.rules.iter().enumerate() {
            let coverage = self.par_coverage(&data, pattern);

            coverage?
                .iter()
                .zip(predictions.iter_mut())
                .zip(rule_index.iter_mut())
                .filter(|((&is_covered, prediction), _)| prediction.is_none() && is_covered == 1.)
                .map(|((_, prediction), rule_idx)| (prediction, rule_idx))
                .for_each(|(prediction, rile_idx)| {
                    *prediction = Some(*label);
                    *rile_idx = Some(idx)
                });
        }

        Ok((
            Series::new(
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
            ),
            Series::new(
                "RuleIndex".into(),
                rule_index
                    .iter()
                    .map(|x| {
                        x.as_ref()
                            .map_or_else(|| AnyValue::Null, |x| AnyValue::UInt64(*x as u64))
                    })
                    .collect::<Vec<_>>(),
            ),
            data,
        ))
    }

    pub fn predict_fuzzy(&self, data: &DataFrame) -> PolarsResult<Series> {
        let data: DataFrame = self.bin.transform(data)?;
        let mut predictions: Vec<Option<usize>> = vec![None; data.height()];

        let mut confidence = vec![vec![0.0_f64; self.labels.len()]; data.height()];

        for (label, pattern, size, _) in &self.rules {
            let coverage = self.par_coverage(&data, pattern);

            for (&is_covered, (prediction, conf)) in coverage?
                .iter()
                .zip(predictions.iter_mut().zip(confidence.iter_mut()))
            {
                if prediction.is_none() {
                    if is_covered == 1. {
                        *prediction = Some(*label);
                    }
                    conf[*label] += is_covered * (*size);
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
                    *a = b;
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

    pub fn fit2(&mut self, data: &DataFrame, labels: &Series) -> PolarsResult<DataFrame> {
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

        let mut prime_patterns: Vec<(usize, Pattern, f64, usize)> = Vec::new();

        let mut flag = false;

        loop {
            let mut base_score = 0;
            let mut prev_degree_patterns: Vec<(Pattern, usize, usize)> =
                vec![(HashSet::new(), 0, 2)];
            let mut found_at = max_features;

            for d in 1..=max_features {
                let mut curr_degree_patterns = Vec::new();
                //let mut curr_patterns_score = Vec::new();
                let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();

                let length = prev_degree_patterns.len();
                let step = (length / STEP_SIZE).max(1);
                let mut max_score = 0;
                for (pattern_idx, (curr_pattern, score0, tmp0)) in
                    prev_degree_patterns.into_iter().enumerate()
                {
                    if pattern_idx % step == 0 {
                        handle.body(&format!(
                            "processing pattern: {}/{} at depth {} with base {} {:?} {}",
                            pattern_idx + 1,
                            length,
                            d,
                            base_score,
                            remaining_shapes,
                            max_score
                        ));
                        handle.update();
                    }
                    if tmp0 == 0 || score0 < base_score {
                        continue;
                    }

                    curr_degree_patterns.push((curr_pattern.clone(), score0, tmp0));

                    if tmp0 == 1 {
                        continue;
                    }
                    if curr_pattern.len() == d - 1 {
                        for idx in 0..features.len() {
                            for &term in &[true, false] {
                                let mut next_pattern = curr_pattern.clone();
                                if !next_pattern.insert((term, idx)) {
                                    continue;
                                }

                                let (counts, tmp) = self.count(&grouped_dfs, &next_pattern)?;
                                if tmp == 0 {
                                    continue;
                                }
                                let max_value = counts.iter().cloned().max().unwrap_or_default();
                                // let max = vec![max_value; counts.len()];
                                // let score = Binarizer::score(
                                //     &counts.iter().map(|x| *x as u128).collect::<Vec<_>>(),
                                //     &max.iter().map(|x| *x as u128).collect::<Vec<_>>(),
                                // );
                                // if score < self.bin.threshold {
                                //     continue;
                                // }
                                let condition = {
                                    if self.deep == 9 {
                                        base_score
                                            .checked_mul(193)
                                            .map(|p| p / 122)
                                            .unwrap_or_else(|| (base_score / 122) * 193)
                                    } else {
                                        base_score
                                    }
                                };

                                let score1 = counts.iter().max().unwrap_or(&0).to_owned();
                                if score1 < condition && tmp != 1 || score1 < base_score && tmp == 1
                                {
                                    continue;
                                }
                                if score1 > base_score && tmp == 1 {
                                    found_at = d;
                                    base_score = score1;
                                }
                                if max_value > max_score {
                                    max_score = max_value;
                                }

                                curr_degree_patterns.push((next_pattern, score1, tmp));
                            }
                        }
                    }
                }

                prev_degree_patterns = curr_degree_patterns
                    .into_iter()
                    .filter(|(_, a, _)| *a >= base_score)
                    .collect();

                if max_score <= base_score || found_at + 2 == d
                // || (base_score as f64 >= max_score as f64 * (1. - f64::exp(-1.))
                //     && self.deep == 9)
                // || d == max_features
                {
                    break;
                }
            }
            let pattern = {
                let prev_degree_patterns: Vec<_> = prev_degree_patterns
                    .iter()
                    .filter(|x| x.2 == 1)
                    .cloned()
                    .collect();

                let len = prev_degree_patterns.len();
                match len {
                    0 => {
                        if flag {
                            break;
                        } else {
                            flag = true;
                            continue;
                        }
                    }
                    1 => prev_degree_patterns[0].0.clone(),
                    _ => {
                        let mut p = prev_degree_patterns[0].clone();
                        for i in prev_degree_patterns.iter().skip(1) {
                            if i.1 > p.1 || i.0.len() < p.0.len() {
                                p = i.clone();
                            }
                        }
                        p.0
                    }
                }
            };

            let (counts, _tmp) = self.count(&grouped_dfs, &pattern)?;
            let max = counts.iter().copied().max().unwrap_or_default();
            let index = counts
                .iter()
                .find_position(|&x| *x == max)
                .map(|x| x.0)
                .unwrap_or_default();
            let len = grouped_dfs[index].shape().0;
            let score = max as f64 / len as f64;
            for i in 0..grouped_dfs.len() {
                let mask = self.coverage(&grouped_dfs[i], &pattern)?;
                grouped_dfs[i] = grouped_dfs[i].filter(&mask.into_iter().map(|x| !x).collect())?;
            }
            prime_patterns.push((index, pattern, score, max));
        }

        let mut remaning_data = DataFrame::empty();

        for (idx, df) in grouped_dfs.into_iter().enumerate() {
            let len = df.shape().0;
            if len == 0 {
                continue;
            }
            let l = Series::new("target".into(), vec![self.labels.get(idx).unwrap(); len]);
            let data = df.hstack(&[l])?;
            remaning_data.vstack_mut(&data)?;
        }

        self.rules = prime_patterns;
        Ok(remaning_data)
    }

    pub fn fit(&mut self, data: &DataFrame, labels: &Series) -> PolarsResult<DataFrame> {
        if self.deep >= 9 {
            return self.fit2(data, labels);
        }
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

        let mut prev_degree_patterns: Vec<(Pattern, usize, usize)> = vec![(HashSet::new(), 0, 2)];

        let mut prime_patterns: Vec<(usize, Pattern, f64, usize)> = Vec::new();

        for d in 1..=max_features {
            println!("{d}");
            let start_time = Instant::now();
            let mut curr_degree_patterns = Vec::new();
            //let mut curr_patterns_score = Vec::new();
            let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
            println!("{remaining_shapes:?}");

            let length = prev_degree_patterns.len();
            let step = (length / STEP_SIZE).max(1);
            for (pattern_idx, (curr_pattern, score0, tmp0)) in
                prev_degree_patterns.into_iter().enumerate()
            {
                if pattern_idx % step == 0 {
                    handle.body(&format!(
                        "processing pattern: {}/{} at depth {}",
                        pattern_idx + 1,
                        length,
                        d
                    ));
                    handle.update();
                }
                if (self.deep >> 2) % 2 == 1 {
                    if tmp0 == 0 {
                        continue;
                    }

                    curr_degree_patterns.push((curr_pattern.clone(), score0, tmp0));

                    if tmp0 == 1 {
                        continue;
                    }
                }
                if curr_pattern.len() == d - 1 {
                    for idx in 0..features.len() {
                        for &term in &[true, false] {
                            let mut next_pattern = curr_pattern.clone();
                            if !next_pattern.insert((term, idx)) {
                                continue;
                            }

                            let (counts, tmp) = self.count(&grouped_dfs, &next_pattern)?;

                            if tmp == 0 {
                                continue;
                            }
                            let max = vec![*counts.iter().max().unwrap(); counts.len()];
                            let score = Binarizer::score(
                                &counts.iter().map(|x| *x as u128).collect::<Vec<_>>(),
                                &max.iter().map(|x| *x as u128).collect::<Vec<_>>(),
                            );
                            if score < self.bin.threshold {
                                continue;
                            }
                            let covered: usize = counts.iter().sum();
                            let score1 = if tmp == 1 {
                                counts
                                    .iter()
                                    .enumerate()
                                    .filter(|(i, &x)| {
                                        x as f64 / grouped_dfs[*i].shape().0 as f64 > self.decay[*i]
                                            && x as f64 / covered as f64 >= self.retention[*i]
                                    })
                                    .max_by_key(|(i, &x)| {
                                        if x as f64 / grouped_dfs[*i].shape().0 as f64
                                            > self.decay[*i]
                                            && x as f64 / covered as f64 >= self.retention[*i]
                                        {
                                            x
                                        } else {
                                            0
                                        }
                                    })
                                    .unwrap_or((0, &0))
                                    .1
                                    .to_owned()
                            } else {
                                0
                            };

                            curr_degree_patterns.push((next_pattern, score1, tmp));
                        }
                    }
                }
            }
            if (self.deep >> 2) % 2 == 0 || d == max_features {
                let mut count = 0u128;
                loop {
                    curr_degree_patterns.sort_by_key(|(b, a, _t)| (*a, std::cmp::Reverse(b.len())));
                    let mut max_score_at = ((0, 0), 0);

                    let mut flags = vec![false; grouped_dfs.len()];
                    let length = curr_degree_patterns.len();
                    let step = (length / STEP_SIZE).max(1);
                    for i in (0..curr_degree_patterns.len()).rev() {
                        let lens = grouped_dfs.iter().map(|x| x.shape().0).collect::<Vec<_>>();
                        let pattern = &curr_degree_patterns[i].0;

                        let (counts, tmp) = self.count(&grouped_dfs, pattern)?;
                        if tmp == 0 {
                            curr_degree_patterns.swap_remove(i);
                            continue;
                        }
                        let max = vec![*counts.iter().max().unwrap(); counts.len()];

                        let score = Binarizer::score(
                            &counts.iter().map(|x| *x as u128).collect::<Vec<_>>(),
                            &max.iter().map(|x| *x as u128).collect::<Vec<_>>(),
                        );
                        if score < self.bin.threshold {
                            count += 1;
                            curr_degree_patterns.swap_remove(i);
                            continue;
                        }
                        if i % step == 0 {
                            handle.body(&format!(
                                "processing best pattern: {}/{} at depth {} {:?}",
                                length - i,
                                length,
                                d,
                                lens
                            ));
                            handle.update();
                        }

                        let covered: usize = counts.iter().sum();
                        let (msa, score1) = counts
                            .iter()
                            .enumerate()
                            .filter(|(i, &x)| {
                                x as f64 / grouped_dfs[*i].shape().0 as f64 > self.decay[*i]
                                    && x as f64 / covered as f64 >= self.retention[*i]
                            })
                            .max_by_key(|(i, &x)| {
                                if x as f64 / grouped_dfs[*i].shape().0 as f64 > self.decay[*i]
                                    && x as f64 / covered as f64 >= self.retention[*i]
                                {
                                    x
                                } else {
                                    0
                                }
                            })
                            .unwrap_or((0, &0));

                        if *score1 == 0 {
                            curr_degree_patterns.swap_remove(i);
                            continue;
                        }

                        max_score_at = ((msa, *score1), i);
                        let o = self.deep % 4 > 1 && flags.iter().any(|&x| x);

                        let len = grouped_dfs[max_score_at.0 .0].shape().0;
                        if tmp == 1 && !o
                        // && (*score1 as f64 / len as f64 >= f64::exp(-4.)
                        //     || len < 50 && *score1 > 0)
                        {
                            for i in 0..grouped_dfs.len() {
                                let mask = self.coverage(&grouped_dfs[i], pattern)?;
                                grouped_dfs[i] = grouped_dfs[i]
                                    .filter(&mask.into_iter().map(|x| !x).collect())?;
                            }
                            prime_patterns.push((
                                max_score_at.0 .0,
                                curr_degree_patterns[max_score_at.1].0.clone(),
                                *score1 as f64 / len as f64,
                                *score1,
                            ));
                            curr_degree_patterns.swap_remove(i);
                            flags[max_score_at.0 .0] = true;
                        } else {
                            curr_degree_patterns[i].1 = max_score_at.0 .1;
                            curr_degree_patterns[i].2 = tmp;
                        }
                        if self.deep % 4 == 2 && flags.iter().any(|&x| x) {
                            break;
                        }
                    }

                    handle.body(&format!("Max Score: {} at depth {}", max_score_at.0 .1, d));
                    handle.update();

                    if flags.iter().all(|x| !*x) {
                        break;
                    }
                    if self.deep % 4 == 0 {
                        break;
                    }
                }

                let duration = start_time.elapsed();
                println!("Time taken: {:.2} milliseconds", duration.as_millis());

                // Check remaining shapes
                let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
                println!("{remaining_shapes:?} {count}");

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
            }
            prev_degree_patterns = curr_degree_patterns
                .into_iter()
                // .map(|(x, _c, t)| x)
                .collect();
        }

        let mut remaning_data = DataFrame::empty();

        for (idx, df) in grouped_dfs.into_iter().enumerate() {
            let len = df.shape().0;
            if len == 0 {
                continue;
            }
            let l = Series::new("target".into(), vec![self.labels.get(idx).unwrap(); len]);
            let data = df.hstack(&[l])?;
            remaning_data.vstack_mut(&data)?;
        }

        self.rules = prime_patterns;
        Ok(remaning_data)
    }
}

impl RuleGenerator {
    fn count(
        &self,
        grouped_dfs: &Vec<DataFrame>,
        pattern: &Pattern,
    ) -> PolarsResult<(Vec<usize>, usize)> {
        let lens: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
        let counts: Vec<usize> = grouped_dfs
            .par_iter()
            .map(|df| {
                self.coverage(df, pattern)
                    .map(|mask| mask.into_iter().filter(|&x| x).count())
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        let tmp = counts
            .iter()
            .zip(lens.iter())
            .zip(self.decay.iter())
            .filter(|((&x, &l), &d)| {
                // print!("{x}, {l}, {d} ");
                x as f64 / l as f64 > d
            })
            .count();
        // println!("{tmp}");
        Ok((counts, tmp))
    }

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
        let pattern_iter = pattern.iter();

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
                    *a += 1.
                }
            });
        }

        mask.iter_mut().for_each(|a| {
            if len != 0. {
                *a /= len
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
