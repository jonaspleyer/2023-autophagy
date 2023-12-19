#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

//! To go back to the main documentation click [here](https://jonaspleyer.github.io/2023-autophagy/)

/// Contains properties of particles such as interaction and mechanics.
mod particle_properties;

/// Methods and objects to run a full simulation.
mod simulation;

pub use particle_properties::*;
pub use simulation::*;

use pyo3::prelude::*;

#[pymodule]
fn cr_autophagy_pyo3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;

    m.add_class::<SimulationSettings>()?;
    m.add_class::<Species>()?;
    m.add_class::<TypedInteraction>()?;
    m.add_class::<cellular_raza::building_blocks::cell_building_blocks::mechanics::Langevin3D>()?;

    Ok(())
}
