use polars::prelude::*;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
};
use pyo3_polars::{PyDataFrame, PySeries};

#[pyclass]
pub struct Binarizer {
    cutpoints: Vec<PySeries>,
    threshold: f64,
}

#[pymethods]
impl Binarizer {
    #[new]
    pub fn new(threshold: f64) -> Self {
        Self {
            cutpoints: Vec::new(),
            threshold,
        }
    }

    pub fn get_cutpoints(&self) -> Vec<PySeries> {
        self.cutpoints.clone()
    }

    #[pyo3(name = "generate_cutpoints")]
    pub fn generate_cutpoints_py(&mut self, X: PyDataFrame, y: PySeries) -> PyResult<()> {
        match self.generate_cutpoints(X.into(), y.into(), self.threshold) {
            Ok(a) => Ok(a),
            Err(e) => Err(PyErr::new::<PyRuntimeError, _>(e.to_string())),
        }
    }

    pub fn test_function(&mut self) {
        self.cutpoints
            .push(PySeries(Series::new("test".into(), [1, 2, 3, 4, 5, 6])));
    }
}

impl Binarizer {
    pub fn generate_cutpoints(
        &mut self,
        X: DataFrame,
        y: Series,
        threshold: f64,
    ) -> Result<(), PolarsError> {
        let schema = X.schema();
        self.cutpoints = Vec::new();
        let features = schema.iter_names();
        let labels = y.unique_stable()?;
        let mut label_counts = vec![0u128; labels.len()];
        let mut running_counts = vec![0u128; labels.len()];
        for l in y.iter() {
            for (j, lj) in labels.iter().enumerate() {
                if lj == l {
                    label_counts[j] += 1;
                    break;
                }
            }
        }
        let label_counts = label_counts;
        for (idx, feature) in features.enumerate() {
            let mut col_y = DataFrame::new(vec![X[idx].clone(), y.clone()])?;
            if let Some((_, dtype)) = schema.get_at_index(idx) {
                if dtype.is_numeric() {
                    //let mut cutpoints = Series::new(*feature, []);
                    col_y = col_y.sort([feature.to_string()], Default::default())?;
                    let mut cps = Vec::new();
                    let sorted = col_y.column(&feature.to_string())?;
                    let labels = col_y.column(&y.name().to_string())?;
                    let mut prev_label = labels.get(0)?;
                    let mut prev_value = sorted.get(0)?;
                    running_counts[labels.iter().position(|x| x == prev_label).unwrap()] += 1;
                    for (s, l) in sorted.iter().zip(labels.iter()).skip(1) {
                        let score = Self::score(&running_counts, &label_counts);
                        running_counts[labels.iter().position(|x| x == l).unwrap()] += 1;
                        if prev_label != l {
                            if prev_value != s {
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
                    }
                }
            }
        }
        Ok(())
    }

    fn score(runer: &Vec<u128>, total: &Vec<u128>) -> f64 {
        let sum: f64 = runer
            .iter()
            .zip(total.iter())
            .map(|(&r, &t)| r as f64 / t as f64)
            .sum();

        let score = runer
            .iter()
            .zip(total.iter())
            .map(|(&r, &t)| {
                let rate = r as f64 / t as f64;
                rate * (sum - rate)
            })
            .sum::<f64>()
            / (runer.len() - 1) as f64;

        sum - score
    }
}
