use pyo3::prelude::*;
mod binarization;
use binarization::{binarize::Binarizer, rule_generation::RuleGenerator};

#[pymodule]
pub fn lad_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Binarizer>()?;
    m.add_class::<RuleGenerator>()?;
    Ok(())
}
