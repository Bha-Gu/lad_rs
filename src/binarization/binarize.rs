use std::cmp::Ordering;

use polars::prelude::*;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use pyo3_polars::{PyDataFrame, PySeries};

#[derive(Clone)]
#[pyclass]
pub struct Binarizer {
    cutpoints: Vec<Series>,
    threshold: f64,
    nominal_size: usize,
}

#[pymethods]
impl Binarizer {
    #[new]
    pub const fn new(threshold: f64, nominal_size: usize) -> Self {
        Self {
            cutpoints: Vec::new(),
            threshold,
            nominal_size,
        }
    }

    pub fn get_cutpoints(&self) -> Vec<PySeries> {
        self.cutpoints.iter().map(|x| PySeries(x.clone())).collect()
    }

    #[pyo3(name = "generate_cutpoints")]
    pub fn generate_cutpoints_py(&mut self, X: PyDataFrame, y: PySeries) -> PyResult<()> {
        match self.generate_cutpoints(&X.into(), &y.into(), self.threshold) {
            Ok(a) => Ok(a),
            Err(e) => Err(PyErr::new::<PyRuntimeError, _>(e.to_string())),
        }
    }

    pub fn transform(&self, X: PyDataFrame) -> PyResult<PyDataFrame> {
        // Convert PyDataFrame into a Polars DataFrame
        println!("Debug0");
        let df: DataFrame = X.into();
        let mut out = DataFrame::default();
        // Check if cutpoints are available
        if self.cutpoints.is_empty() {
            return Err(PyErr::new::<PyRuntimeError, _>(
                "Cutpoints not generated. Call 'generate_cutpoints' first.",
            ));
        }

        println!("Debug1");

        // Iterate over each feature (column) and apply corresponding cutpoints
        for cutpoints in &self.cutpoints {
            println!("Debug2");
            let feature_name = cutpoints.name();
            let dtype = cutpoints.dtype();
            // Ensure the feature exists in the DataFrame
            if let Ok(feature_series) = df.column(feature_name) {
                for cp in cutpoints.iter() {
                    let binarized_column = feature_series
                        .iter()
                        .map(|value| {
                            if dtype.is_numeric() {
                                if value >= cp {
                                    true // Value is less than the cutpoint
                                } else {
                                    false // Value is greater than or equal to the cutpoint
                                }
                            } else {
                                value == cp
                            }
                        })
                        .collect::<Vec<_>>();

                    match out.hstack_mut(&[Series::new(
                        format!("{feature_name} on {cp}").into(),
                        binarized_column,
                    )]) {
                        Ok(_) => (),
                        Err(e) => return Err(PyErr::new::<PyRuntimeError, _>(e.to_string())),
                    };
                }
            } else {
                // If feature is not found, raise an error
                return Err(PyErr::new::<PyRuntimeError, _>(format!(
                    "Feature '{feature_name}' not found in DataFrame"
                )));
            }
        }

        // Convert the DataFrame back to PyDataFrame and return
        Ok(PyDataFrame(out))
    }
}

impl Binarizer {
    pub fn generate_cutpoints(
        &mut self,
        X: &DataFrame,
        y: &Series,
        threshold: f64,
    ) -> Result<(), PolarsError> {
        let schema = X.schema();
        self.cutpoints = Vec::new();
        let features = schema.iter_names();
        let unique_labels = y.unique_stable()?;
        let mut label_counts = vec![0u128; unique_labels.len()];
        for l in y.iter() {
            for (j, lj) in unique_labels.iter().enumerate() {
                if lj == l {
                    label_counts[j] += 1;
                    break;
                }
            }
        }
        let label_counts = label_counts;
        for (idx, feature) in features.enumerate() {
            let s = X[idx].clone();
            let a = s.n_unique()?;
            let mut col_y = DataFrame::new(vec![y.clone(), s])?;
            if let Some((_, dtype)) = schema.get_at_index(idx) {
                if dtype.is_numeric() && a >= self.nominal_size {
                    //let mut cutpoints = Series::new(*feature, []);
                    let mut running_counts = vec![0u128; unique_labels.len()];
                    col_y = col_y.sort([feature.to_string()], SortMultipleOptions::default())?;
                    let mut cps = Vec::new();
                    let sorted = col_y.drop_in_place(feature.as_ref())?;
                    let labels = col_y.drop_in_place(y.name().as_ref())?;
                    let mut prev_label = labels.get(0)?;
                    let mut prev_value = sorted.get(0)?;
                    running_counts[labels.iter().position(|x| x == prev_label).unwrap()] += 1;
                    for (s, l) in sorted.iter().zip(labels.iter()).skip(1) {
                        let score = Self::score(&running_counts, &label_counts);
                        running_counts[unique_labels.iter().position(|x| x == l).unwrap()] += 1;
                        if prev_label != l && prev_value != s {
                            if score >= threshold {
                                cps.push((
                                    Series::new("tmp".into(), [s.clone(), prev_value.clone()])
                                        .mean()
                                        .unwrap(),
                                    score,
                                ));
                            }
                            prev_value = s;
                            prev_label = l;
                        }
                    }
                    cps.sort_by(|(_, a), (_, b)| {
                        if a.is_nan() {
                            Ordering::Greater
                        } else if b.is_nan() {
                            Ordering::Less
                        } else {
                            a.partial_cmp(b).unwrap()
                        }
                    });
                    let cps = cps
                        .iter()
                        .rev()
                        .map(|(x, s)| {
                            print!("{s} ");
                            *x
                        })
                        .collect::<Vec<_>>();
                    println!();
                    self.cutpoints.push(Series::new(feature.clone(), cps));
                } else {
                    self.cutpoints.push(
                        col_y
                            .drop_in_place(feature)?
                            .unique()?
                            .cast(&DataType::String)?,
                    );
                }
            }
        }
        Ok(())
    }

    fn score(runner: &[u128], total: &[u128]) -> f64 {
        let rates = runner
            .iter()
            .zip(total.iter())
            .map(|(&r, &t)| r as f64 / t as f64)
            .collect::<Vec<_>>();
        let sum = rates.iter().sum::<f64>();

        let score = rates.iter().map(|x| x * (sum - x)).sum::<f64>() / (runner.len() - 1) as f64;

        let x = sum - score;

        let k = -2.0 + (2.0_f64).powi(runner.len() as i32 - 1);

        x / k.mul_add(1.0 - x, 1.0)
    }
}
