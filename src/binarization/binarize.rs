use polars::prelude::*;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
};
use pyo3_polars::{PyDataFrame, PySeries};

#[pyclass]
pub struct Binarizer {
    cutpoints: Vec<PySeries>,
}

#[pymethods]
impl Binarizer {
    #[new]
    pub fn new() -> Self {
        Self {
            cutpoints: Vec::new(),
        }
    }

    pub fn get_cutpoints(&self) -> Vec<PySeries> {
        self.cutpoints.clone()
    }

    #[pyo3(name = "generate_cutpoints")]
    pub fn generate_cutpoints_py(&mut self, X: PyDataFrame, y: PySeries) -> PyResult<()> {
        match self.generate_cutpoints(X.into(), y.into()) {
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
    pub fn generate_cutpoints(&mut self, X: DataFrame, y: Series) -> Result<(), PolarsError> {
        let schema = X.schema();
        self.cutpoints = Vec::new();
        let features = schema.iter_names();
        let labels = y.unique_stable()?;
        let mut label_counts = vec![0; labels.len()];
        for l in y.iter() {
            for (j, lj) in labels.iter().enumerate() {
                if lj == l {
                    label_counts[j] += 1;
                    break;
                }
            }
        }
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
                    for (s, l) in sorted.iter().zip(labels.iter()).skip(1) {
                        if prev_label != l {
                            if prev_value != s {
                                cps.push(
                                    Series::new("tmp".into(), [s.clone(), prev_value.clone()])
                                        .mean(),
                                );
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
}
