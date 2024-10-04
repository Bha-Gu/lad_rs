use pyo3::prelude::*;
mod binarization;
use binarization::{binarize::Binarizer, rule_generation::RuleGenerator};

use pyo3::types::PyModule;

use std::io::{self, Write};
use std::thread;
use std::time::Duration;

#[pyfunction]
fn rust_print(py: Python) -> PyResult<()> {
    // Import Python's 'builtins' module
    let builtins = PyModule::import_bound(py, "builtins")?;
    let sys = PyModule::import_bound(py, "sys")?;
    // Call the 'print' function from Python
    builtins
        .getattr("print")?
        .call1(("Hello from Rust via PyO3!",))?;
    //sys.getattr("stdout")?.getattr("flush")?.call0()?;
    println!("Hello");

    //io::stdout().flush().unwrap();

    thread::sleep(Duration::from_secs(5));

    Ok(())
}

#[pymodule]
pub fn lad_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_print, m)?)?;
    m.add_class::<Binarizer>()?;
    m.add_class::<RuleGenerator>()?;
    Ok(())
}
