use core::f64;
use std::cmp::Reverse;
use std::collections::HashSet;

use super::binarize::Binarizer;
use itertools::Itertools;

use notify_rust::Notification;

use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use std::time::Duration;

type Pattern = HashSet<(bool, usize)>;

const STEP_SIZE: usize = 100_000;

#[derive(Clone)]
pub struct RuleGenerator {
    bin: Binarizer,
    max: usize,
    rules: Vec<(usize, Pattern, f64, usize)>,
    labels: Series,
    fallback_label: usize,
    features: Vec<String>,
    remaining: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
struct RuleGeneratorSaver {
    max: usize,
    rules: Vec<(usize, Pattern, f64, usize)>,
    fallback_label: usize,
    features: Vec<String>,
    remaining: Vec<usize>,
    // Filenames for non-serializable parts.
    bin_json: String,
    labels_csv: String,
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

    pub fn save(self, json_path: &str, work_dir: &str) -> Result<(), Box<dyn Error>> {
        // Define file paths for the non-serializable fields.
        let bin_json = format!("{work_dir}/binarizer.json");
        let labels_csv = format!("{work_dir}/labels.csv");

        // Save the Binarizer.
        self.bin.save(&bin_json, work_dir)?;

        // Save the labels Series to CSV.
        {
            let mut file = File::create(&labels_csv)?;
            let mut df = DataFrame::new(vec![self.labels.clone().into()])?;
            CsvWriter::new(&mut file).finish(&mut df)?;
        }

        // Build the dummy struct for the rest of RuleGenerator.
        let saver = RuleGeneratorSaver {
            max: self.max,
            rules: self.rules.clone(),
            fallback_label: self.fallback_label,
            features: self.features.clone(),
            remaining: self.remaining.clone(),
            bin_json: bin_json.clone(),
            labels_csv: labels_csv.clone(),
        };

        let file = File::create(json_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &saver)?;
        Ok(())
    }

    // Load RuleGenerator: reads the JSON file, then loads the Binarizer and labels Series.
    pub fn load(
        json_path: &str,
        work_dir: &str,
    ) -> Result<(Binarizer, RuleGenerator), Box<dyn Error>> {
        let file = File::open(json_path)?;
        let reader = BufReader::new(file);
        let saver: RuleGeneratorSaver = serde_json::from_reader(reader)?;

        // Load the Binarizer.
        let bin = Binarizer::load(&saver.bin_json, work_dir)?;

        // Load the labels Series.
        let labels = {
            let file = File::open(&saver.labels_csv)?;
            let df = CsvReader::new(file).finish()?;
            df.select_at_idx(0)
                .ok_or("No column found in labels CSV")?
                .clone()
        };

        Ok((
            bin.clone(),
            RuleGenerator {
                bin,
                max: saver.max,
                rules: saver.rules,
                labels: labels.as_materialized_series_maintain_scalar().rechunk(),
                fallback_label: saver.fallback_label,
                features: saver.features,
                remaining: saver.remaining,
            },
        ))
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

    pub fn predict_exact(
        &self,
        data: &DataFrame,
        until: usize,
    ) -> PolarsResult<(Series, Series, DataFrame)> {
        let data: DataFrame = self.bin.transform(data)?;
        let mut predictions: Vec<Option<usize>> = vec![None; data.height()];
        let mut rule_index: Vec<Option<usize>> = vec![None; data.height()];

        for (idx, (label, pattern, _size, count)) in self.rules.iter().enumerate() {
            if *count < until {
                break;
            }

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

    pub fn predict_fuzzy(
        &self,
        data: &DataFrame,
        until: usize,
    ) -> PolarsResult<(Series, Series, DataFrame)> {
        // Create a channel for sending notification messages
        let (_tx, rx) = mpsc::channel::<String>();

        // Spawn a thread to handle notifications
        let _notifier_handle = thread::spawn(move || {
            // Create the notification instance in the thread
            let mut handle = Notification::new()
                .summary("Task Progress")
                .body("Task is starting")
                .show()
                .unwrap();

            // Loop and update notifications as messages are received.
            //
            if !cfg!(target_os = "windows") {
                for msg in rx {
                    handle.body(&msg);
                    handle.update();
                }
            }
        });

        let data: DataFrame = self.bin.transform(data)?;
        let mut predictions: Vec<Option<usize>> = vec![None; data.height()];
        let mut rule_index: Vec<Option<usize>> = vec![None; data.height()];
        let mut fuzzy = vec![vec![(0, 0); data.width()]; self.labels.len()];

        for (idx, (label, pattern, _size, count)) in self.rules.iter().enumerate() {
            if *count < until {
                break;
            }

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

            for i in pattern {
                if i.0 {
                    fuzzy[*label][i.1].0 += count;
                } else {
                    fuzzy[*label][i.1].1 += count;
                }
            }

            // let coverage = self.par_coverage(&data, pattern);

            mask.iter()
                .zip(predictions.iter_mut())
                .zip(rule_index.iter_mut())
                .filter(|((&is_covered, prediction), _)| prediction.is_none() && is_covered)
                .map(|((_, prediction), rule_idx)| (prediction, rule_idx))
                .for_each(|(prediction, rile_idx)| {
                    *prediction = Some(*label);
                    *rile_idx = Some(idx)
                });
        }

        // For each row with a None prediction, compute the similarity against each class.
        for (row_idx, prediction) in predictions.iter_mut().enumerate() {
            if prediction.is_none() {
                let mut best_label = None;
                let mut best_similarity = 0f64;

                // Iterate over each label's fuzzy vector.
                for (label_idx, fuzzy_values) in fuzzy.iter().enumerate() {
                    let mut similarity = 0f64;

                    // Iterate over each feature.
                    for (col_idx, (true_count, false_count)) in fuzzy_values.iter().enumerate() {
                        let total = true_count + false_count;
                        if total == 0 {
                            continue; // Avoid division by zero if no counts are recorded.
                        }
                        // Normalize the counts to get proportions.
                        let normalized_true = *true_count as f64 / total as f64;
                        let normalized_false = *false_count as f64 / total as f64;

                        // Retrieve the boolean value of the row for this feature.
                        let col = data.column(&self.features[col_idx])?;
                        let value = col.bool()?.get(row_idx);

                        if let Some(row_val) = value {
                            similarity += if row_val {
                                normalized_true
                            } else {
                                normalized_false
                            };
                        }
                    }

                    // Check if this label's similarity is the best so far.
                    if similarity > best_similarity {
                        best_similarity = similarity;
                        best_label = Some(label_idx);
                    }
                }
                *prediction = best_label;
            }
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

    pub fn fit(&mut self, data: &DataFrame, labels: &Series) -> PolarsResult<DataFrame> {
        let features = data.get_column_names();
        self.features = features.iter().map(|&x| x.to_string()).collect();
        self.labels = labels.unique_stable()?;

        // Divide data into groups based on the labels
        let mut grouped_dfs: Vec<DataFrame> = self.divide_data(data, labels);
        // println!("{grouped_dfs:?}");
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

        // Create a channel for sending notification messages
        let (tx, rx) = mpsc::channel::<String>();

        // Spawn a thread to handle notifications
        let notifier_handle = thread::spawn(move || {
            // Create the notification instance in the thread
            let mut handle = Notification::new()
                .summary("Task Progress")
                .body("Task is starting")
                .show()
                .unwrap();

            // Loop and update notifications as messages are received.

            if !cfg!(target_os = "windows") {
                for msg in rx {
                    handle.body(&msg);
                    handle.update();
                }
            }
        });

        let mut prime_patterns: Vec<(usize, Pattern, f64, usize)> = Vec::new();
        let mut flag = false;

        let mut prev_loop_best_patterns: Vec<(Pattern, usize, usize)> =
            vec![(HashSet::new(), 0, grouped_dfs.len())];

        // let mut loop_counter = 0;
        loop {
            // loop_counter += 1;
            // println!("Loop: {loop_counter}");

            let mut base_score = 0;
            let mut prev_degree_patterns: Vec<(Pattern, usize, usize)> =
                vec![(HashSet::new(), usize::MAX, grouped_dfs.len())];
            let mut found_at = max_features;

            let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();
            let length = prev_loop_best_patterns.len();
            let step = (length / STEP_SIZE).max(1);
            prev_degree_patterns
                .sort_by_key(|(pattern, score, tmp)| (pattern.len(), Reverse(*score), *tmp));
            for (pattern_idx, (curr_pattern, score0, _tmp0)) in
                prev_loop_best_patterns.clone().into_iter().enumerate()
            {
                if score0 <= base_score {
                    // println!("{score0} <? {base_score} ");
                    continue;
                }
                if pattern_idx % step == 0 {
                    let msg = format!(
                        "processing pattern: {}/{} at depth -1 with base {} {:?} {}",
                        pattern_idx + 1,
                        length,
                        base_score,
                        remaining_shapes,
                        score0
                    );
                    // Send the message to the notification thread.
                    let _ = tx.send(msg);
                }
                // thread::sleep(Duration::from_secs(1));
                let max_idx = { curr_pattern.iter().map(|(_, a)| *a).max() };
                for idx in max_idx.map(|x| x + 1).unwrap_or_default()..features.len() {
                    for term in [true, false] {
                        let mut next_pattern = curr_pattern.clone();
                        if !next_pattern.insert((term, idx)) {
                            continue;
                        }

                        let (counts, tmp) = self.count(&grouped_dfs, &next_pattern)?;

                        let mut lens = grouped_dfs
                            .iter()
                            .map(|x| x.shape().0 as f64)
                            .zip(counts.iter().map(|&x| x as f64))
                            .map(|(l, c)| c / l)
                            .collect::<Vec<_>>();

                        let max_lens = lens.iter().cloned().reduce(f64::max).unwrap_or(f64::NAN);

                        for len in lens.iter_mut() {
                            *len /= max_lens;
                        }

                        if let Some(pos) = lens.iter().position(|&x| x == 1.0) {
                            lens[pos] = 0.0;
                        }

                        let max_value = counts.iter().cloned().max().unwrap_or_default();

                        if max_value < base_score {
                            continue;
                        }
                        if max_value > base_score && tmp == 1 {
                            base_score = max_value;
                        }
                    }
                }
            }
            for d in 1..=max_features {
                let mut curr_degree_patterns = Vec::new();
                let remaining_shapes: Vec<_> = grouped_dfs.iter().map(|df| df.shape().0).collect();

                let length = prev_degree_patterns.len();
                let step = (length / STEP_SIZE).max(1);
                let mut max_score = 0;
                let mut d1 = None;
                prev_degree_patterns.sort_by_key(|(pattern, score, tmp)| {
                    (pattern.len(), *tmp != 1, Reverse(*score), *tmp)
                });
                for (pattern_idx, (curr_pattern, score0, tmp0)) in
                    prev_degree_patterns.into_iter().enumerate()
                {
                    if tmp0 == 0 || score0 < base_score {
                        // println!("{score0} <? {base_score} ");
                        continue;
                    }

                    curr_degree_patterns.push((curr_pattern.clone(), score0, tmp0));

                    if tmp0 == 1 || (score0 == base_score && base_score != 0) {
                        // println!("{score0} =? {base_score} != 0 ");
                        continue;
                    }
                    if pattern_idx % step == 0 {
                        let msg = format!(
                            "processing pattern: {}/{} at depth {} with base {} {:?} {} {}",
                            pattern_idx + 1,
                            length,
                            d,
                            base_score,
                            remaining_shapes,
                            max_score,
                            score0
                        );
                        // Send the message to the notification thread.
                        let _ = tx.send(msg);
                    }
                    if curr_pattern.len() == d - 1 {
                        if d1.is_none() {
                            d1 = Some(pattern_idx);
                        }
                        let max_idx = { curr_pattern.iter().map(|(_, a)| *a).max() };
                        for idx in max_idx.map(|x| x + 1).unwrap_or_default()..features.len() {
                            for term in [true, false] {
                                let mut next_pattern = curr_pattern.clone();
                                if !next_pattern.insert((term, idx)) {
                                    continue;
                                }

                                let (counts, tmp) = self.count(&grouped_dfs, &next_pattern)?;

                                // println!("{d}:{next_pattern:?}->{counts:?}?{tmp}");

                                if tmp == 0 {
                                    continue;
                                }
                                let mut lens = grouped_dfs
                                    .iter()
                                    .map(|x| x.shape().0 as f64)
                                    .zip(counts.iter().map(|&x| x as f64))
                                    .map(|(l, c)| c / l)
                                    .collect::<Vec<_>>();

                                let max_lens =
                                    lens.iter().cloned().reduce(f64::max).unwrap_or(f64::NAN);

                                for len in lens.iter_mut() {
                                    *len /= max_lens;
                                }

                                if let Some(pos) = lens.iter().position(|&x| x == 1.0) {
                                    lens[pos] = 0.0;
                                }

                                let max_value = counts.iter().cloned().max().unwrap_or_default();

                                if max_value < base_score {
                                    continue;
                                }
                                if max_value > base_score && tmp == 1 {
                                    found_at = d;
                                    base_score = max_value;
                                }
                                if max_value > max_score {
                                    max_score = max_value;
                                }
                                // println!("{next_pattern:?}, {max_value}, {tmp}");
                                curr_degree_patterns.push((next_pattern, max_value, tmp));
                            }
                        }
                    }
                }
                // println!("{d}\n{curr_degree_patterns:?}\n");
                prev_degree_patterns = curr_degree_patterns
                    .into_iter()
                    // .filter(|(_, a, _)| *a >= base_score)
                    .collect();

                if found_at + 2 == d {
                    println!("Early Break");
                    break;
                }
            }
            let pattern = {
                let best_patterns: Vec<_> = prev_degree_patterns
                    .iter()
                    .filter(|x| x.2 == 1)
                    .cloned()
                    .collect();

                prev_loop_best_patterns = best_patterns.clone();

                let best_pattern: Vec<_> = best_patterns
                    .into_iter()
                    .filter(|(_, a, _)| *a >= base_score)
                    .collect();

                let len = best_pattern.len();
                match len {
                    0 => {
                        if flag {
                            break;
                        }
                        flag = true;
                        continue;
                    }
                    1 => best_pattern[0].0.clone(),
                    _ => {
                        let mut p = best_pattern[0].clone();
                        for i in best_pattern.iter().skip(1) {
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
            let data = df.hstack(&[l.into()])?;
            remaning_data.vstack_mut(&data)?;
        }

        self.rules = prime_patterns;

        // Optionally, drop the sender to signal the notification thread to exit.
        drop(tx);
        // Wait for the notification thread to finish.
        let _ = notifier_handle.join();

        Ok(remaning_data)
    }
    pub fn fit_legacy(
        &mut self,
        data: &DataFrame,
        labels: &Series,
        deep: usize,
        decay: Vec<f64>,
        retention: Vec<f64>,
    ) -> PolarsResult<DataFrame> {
        if deep >= 9 {
            return self.fit(data, labels);
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
                if !cfg!(target_os = "windows") {
                    if pattern_idx % step == 0 {
                        handle.body(&format!(
                            "processing pattern: {}/{} at depth {}",
                            pattern_idx + 1,
                            length,
                            d
                        ));
                        handle.update();
                    }
                }
                if (deep >> 2) % 2 == 1 {
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
                                        x as f64 / grouped_dfs[*i].shape().0 as f64 > decay[*i]
                                            && x as f64 / covered as f64 >= retention[*i]
                                    })
                                    .max_by_key(|(i, &x)| {
                                        if x as f64 / grouped_dfs[*i].shape().0 as f64 > decay[*i]
                                            && x as f64 / covered as f64 >= retention[*i]
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
            if (deep >> 2) % 2 == 0 || d == max_features {
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
                        if !cfg!(target_os = "windows") {
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
                        }

                        let covered: usize = counts.iter().sum();
                        let (msa, score1) = counts
                            .iter()
                            .enumerate()
                            .filter(|(i, &x)| {
                                x as f64 / grouped_dfs[*i].shape().0 as f64 > decay[*i]
                                    && x as f64 / covered as f64 >= retention[*i]
                            })
                            .max_by_key(|(i, &x)| {
                                if x as f64 / grouped_dfs[*i].shape().0 as f64 > decay[*i]
                                    && x as f64 / covered as f64 >= retention[*i]
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
                        let o = deep % 4 > 1 && flags.iter().any(|&x| x);

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
                        if deep % 4 == 2 && flags.iter().any(|&x| x) {
                            break;
                        }
                    }

                    if !cfg!(target_os = "windows") {
                        handle.body(&format!("Max Score: {} at depth {}", max_score_at.0 .1, d));
                        handle.update();
                    }
                    if flags.iter().all(|x| !*x) {
                        break;
                    }
                    if deep % 4 == 0 {
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
            let data = df.hstack(&[l.into()])?;
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
        let counts: Vec<usize> = grouped_dfs
            .par_iter()
            .map(|df| {
                self.coverage(df, pattern)
                    .map(|mask| mask.into_iter().filter(|&x| x).count())
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        let tmp = counts.iter().filter(|&x| *x > 0).count();
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
