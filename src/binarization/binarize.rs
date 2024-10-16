use std::cmp::Ordering;

use polars::prelude::*;

#[derive(Clone)]
pub struct Binarizer {
    cutpoints: Vec<Series>,
    threshold: f64,
    nominal_size: usize,
    max_cutpoints: usize,
}

impl Binarizer {
    pub const fn new(threshold: f64, nominal_size: usize, max_cutpoints_per_column: usize) -> Self {
        Self {
            cutpoints: Vec::new(),
            threshold,
            nominal_size,
            max_cutpoints: max_cutpoints_per_column,
        }
    }

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
            if a <= self.nominal_size {
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
                    labels
                        .iter()
                        .position(|x| x == prev_label)
                        .unwrap_unchecked()
                }] += 1;
                for (s, l) in sorted.iter().zip(labels.iter()).skip(1) {
                    let score = Self::score(&running_counts, &label_counts);
                    running_counts[unsafe {
                        unique_labels.iter().position(|x| x == l).unwrap_unchecked()
                    }] += 1;
                    if prev_label != l && prev_value != s {
                        if score >= self.threshold {
                            cps.push((
                                AnyValue::from(unsafe {
                                    Series::new("tmp".into(), [s.clone(), prev_value.clone()])
                                        .mean()
                                        .unwrap_unchecked()
                                })
                                .cast(data_type),
                                score,
                            ));
                        }
                        prev_value = s;
                        prev_label = l;
                    }
                }
                cps.sort_by(|(_, a), (_, b)| {
                    if a.is_nan() && b.is_nan() {
                        Ordering::Equal
                    } else if a.is_nan() {
                        Ordering::Greater
                    } else if b.is_nan() {
                        Ordering::Less
                    } else {
                        a.partial_cmp(b).unwrap_or(Ordering::Equal)
                    }
                });
                let cps = cps
                    .iter()
                    .rev()
                    .map(|(x, s)| {
                        print!("{s} ");
                        x.clone()
                    })
                    .take(self.max_cutpoints)
                    .collect::<Vec<_>>();
                println!();
                self.cutpoints.push(Series::new(feature_name.clone(), cps));
            }
        }
        Ok(())
    }

    pub fn transform(&self, df: &DataFrame) -> PolarsResult<DataFrame> {
        let schema = df.schema();

        let mut out = DataFrame::default();

        for (feature_name, data_type) in schema.iter() {
            let column = df.column(feature_name)?;
            if data_type.is_bool() {
                out.hstack_mut(&[column.clone()])?;
                continue;
            }
            let a = column.n_unique()?;
            if false {
                // set to false to debug meant to be "a == 2"
                for value in column.iter().take(1) {
                    out.hstack_mut(&[Series::new(
                        format!("{feature_name} = {value}").into(),
                        column.iter().map(|x| x == value).collect::<Vec<_>>(),
                    )])?;
                }
            } else if a <= self.nominal_size || data_type.is_string() {
                for value in column.unique_stable()?.iter() {
                    out.hstack_mut(&[Series::new(
                        format!("{feature_name} = {value}").into(),
                        column.iter().map(|x| x == value).collect::<Vec<_>>(),
                    )])?;
                }
            } else if data_type.is_numeric() {
                let Some(cutpoints) = self.cutpoints.iter().find(|x| x.name() == feature_name)
                else {
                    return Err(PolarsError::ColumnNotFound(format!("Cannot find numeric column {feature_name}.\nMake sure schema of input and output data is the same.").into()));
                };
                for cutpoint in cutpoints.iter() {
                    out.hstack_mut(&[Series::new(
                        format!("{feature_name} > {cutpoint}").into(),
                        column.iter().map(|x| x > cutpoint).collect::<Vec<_>>(),
                    )])?;
                }
            } else {
                println!("{data_type} not supported yet. Skipping");
            }
        }

        Ok(out)
    }

    fn score(runner: &[u128], total: &[u128]) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let rates = runner
            .iter()
            .zip(total.iter())
            .map(|(&r, &t)| r as f64 / t as f64)
            .collect::<Vec<_>>();
        let sum = rates.iter().sum::<f64>();

        #[allow(clippy::cast_precision_loss)]
        let score = rates.iter().map(|x| x * (sum - x)).sum::<f64>() / (runner.len() - 1) as f64;

        let x = sum - score;

        let k = -2.0 + f64::from((2u32).pow(runner.len() as u32 - 1));

        x / k.mul_add(1.0 - x, 1.0)
    }

    fn entropy(runner: &[u128], total: &[u128]) -> f64 {
        // Calculate total counts
        let total_count: u128 = total.iter().sum();

        // Calculate probabilities
        let probabilities: Vec<f64> = runner
            .iter()
            .map(|&r| {
                if total_count > 0 {
                    r as f64 / total_count as f64
                } else {
                    0.0
                }
            })
            .collect();

        let entropy_value: f64 = probabilities
            .iter()
            .filter_map(|&p| {
                if p > 0.0 {
                    Some(-p * p.log2()) // Using natural logarithm
                } else {
                    None
                }
            })
            .sum();

        entropy_value
    }
}
