pub mod binarization;

pub use crate::binarization::{binarize::Binarizer, rule_generation::RuleGenerator};

//#[pymodule]
//pub fn lad_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
//    m.add_class::<Binarizer>()?;
//    m.add_class::<RuleGenerator>()?;
//    Ok(())
//}
