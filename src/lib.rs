use pyo3::prelude::*;
mod binarization;
use binarization::{binarize::Binarizer, rule_generation::RuleGenerator};

use pyo3::types::PyModule;

#[pymodule]
pub fn lad_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Binarizer>()?;
    m.add_class::<RuleGenerator>()?;
    Ok(())
}
