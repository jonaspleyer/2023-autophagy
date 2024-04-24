use super::particle_properties::*;

use cellular_raza::core::backend::chili;
use cellular_raza::core::storage::{StorageBuilder, StorageInterfaceLoad};
use cellular_raza::{building_blocks::*, core::storage::StorageManager};
use pyo3::prelude::*;

use nalgebra::Vector3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// One nanometre in SI units.
pub const NANOMETRE: f64 = 1e-9;

/// One micrometre in SI units.
pub const MICROMETRE: f64 = 1e-6;

/// One second in SI units
pub const SECOND: f64 = 1.0;

/// One minute in SI units
pub const MINUTE: f64 = 60.0 * SECOND;

/// One hour in SI units
pub const HOUR: f64 = 60.0 * MINUTE;

/// One day in SI units
pub const DAY: f64 = 24.0 * HOUR;

/// The Boltzmann-constant in SI units.
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

/// One Kelvin in SI units
pub const KELVIN: f64 = 1.0;

/// Name of the simulation_settings file
const SIM_SETTINGS: &str = "simulation_settings.json";

/// All settings which can be configured by the Python interface.
///
/// We aim to provide access to even the more lower-level settings
/// that can influence the results of our simulation.
/// Not all settings do make sense and some combinations can lead
/// to numerical integration problems.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[pyclass(get_all, set_all)]
pub struct SimulationSettings {
    /// Number of cargo particles in the simulation.
    pub n_cells_cargo: usize,

    /// Number of Atg11/19 particles in the simulation.
    pub n_cells_atg11w19: usize,

    /// Radius of the Cargo particles
    pub cell_radius_cargo: f64,

    /// Radius of the Atg11/19 particles
    pub cell_radius_atg11w19: f64,

    /// Diffusion as given in the [Brownian3D] struct.
    pub diffusion_atg11w19: f64,

    /// Diffusion as given in the [Brownian3D] struct.
    pub diffusion_cargo: f64,

    /// Product of Boltzmann-Constant and temperature of the
    /// Brownian mechanics model for Atg11/19 particles.
    pub temperature_atg11w19: f64,

    /// Product of Boltzmann-Constant and temperature of the
    /// Brownian mechanics model for Atg11/19 particles.
    pub temperature_cargo: f64,

    /// See TypedInteraction
    pub potential_strength_cargo_cargo: f64,

    /// See TypedInteraction
    pub potential_strength_atg11w19_atg11w19: f64,

    /// See TypedInteraction
    pub potential_strength_cargo_atg11w19: f64,

    /// See TypedInteraction
    pub interaction_range_cargo_cargo: f64,

    /// See TypedInteraction
    pub interaction_range_atg11w19_atg11w19: f64,

    /// See TypedInteraction
    pub interaction_range_atg11w19_cargo: f64,

    /// Integration step of the numerical simulation.
    pub dt: f64,

    /// Maximum time until which to solve
    pub t_max: f64,

    /// Specifies the frequency at which results are saved as json files.
    /// Lower the number for more saved results.
    pub save_interval: f64,

    /// Extra iterations at which the simulation should be saved
    pub extra_saves: Vec<usize>,

    /// Number of threads to use in the simulation.
    pub n_threads: usize,

    /// Overall domain size of the simulation. The lower bound starts at `0.0`.
    pub domain_size: f64,

    /// Upper bound of the radius in which the cargo particles will be spawned
    pub domain_cargo_radius_max: f64,

    /// Minimal radius outside of which atg11w19 particles will be spawned
    /// Must be smaller than the domain_size
    pub domain_atg11w19_radius_min: f64,

    /// See [CartesianCuboid3]
    pub domain_n_voxels: Option<usize>,

    /// Name of the folder to store the results in.
    pub storage_name: std::path::PathBuf,

    /// Path where the cargo initial positions will be stored.
    pub cargo_initials_dir: std::path::PathBuf,

    /// Do we want to show a progress bar
    pub show_progressbar: bool,

    /// The seed with which the simulation is initially configured
    pub random_seed: u64,
}

#[pymethods]
impl SimulationSettings {
    #[new]
    fn new() -> Self {
        let cell_radius_atg11w19: f64 = 100.0 * NANOMETRE;
        let cell_radius_cargo: f64 = 1.0 * cell_radius_atg11w19;

        SimulationSettings {
            n_cells_cargo: 200,
            n_cells_atg11w19: 200,
            cell_radius_cargo,
            cell_radius_atg11w19,

            diffusion_atg11w19: 2e-3 * MICROMETRE.powf(2.0) / SECOND,
            diffusion_cargo: 2e-3 * MICROMETRE.powf(2.0) / SECOND,

            temperature_atg11w19: 300.0 * KELVIN,
            temperature_cargo: 300.0 * KELVIN,

            potential_strength_cargo_cargo: 0.03 * NANOMETRE.powf(2.0) / SECOND.powf(2.0),
            potential_strength_atg11w19_atg11w19: 0.01 * NANOMETRE.powf(2.0) / SECOND.powf(2.0),
            potential_strength_cargo_atg11w19: 0.02 * NANOMETRE.powf(2.0) / SECOND.powf(2.0),

            interaction_range_cargo_cargo: 0.4 * (cell_radius_cargo + cell_radius_atg11w19),
            interaction_range_atg11w19_atg11w19: 0.4 * (cell_radius_cargo + cell_radius_atg11w19),
            interaction_range_atg11w19_cargo: 0.4 * (cell_radius_cargo + cell_radius_atg11w19),

            dt: 0.1 * MINUTE,
            t_max: 30.0 * MINUTE,
            save_interval: 10.0 * MINUTE,
            extra_saves: Vec::new(),

            n_threads: 1,

            domain_size: 2000.0 * NANOMETRE,
            domain_cargo_radius_max: 600.0 * NANOMETRE,
            domain_atg11w19_radius_min: 650.0 * NANOMETRE,

            domain_n_voxels: Some(4),

            storage_name: "out/autophagy".into(),
            cargo_initials_dir: "out/cargo_initials".into(),

            show_progressbar: true,

            random_seed: 1,
        }
    }

    /// Formats the object
    pub fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }

    /// Loads the settings from an existing path.
    ///
    /// Will fail if the given path is invalid or the given json file does not match memory layout
    /// of this object.
    #[staticmethod]
    pub fn load_from_file(path: std::path::PathBuf) -> PyResult<Self> {
        let file = std::fs::File::open(&path).or_else(|e| {
            Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("std::fs::File error: {e} while opening file: {:?}", path),
            ))
        })?;
        let reader = std::io::BufReader::new(file);
        let settings: Self = serde_json::from_reader(reader).or_else(|e| {
            Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("serde_json error in loading simulation settings from file: {e}"),
            ))
        })?;
        Ok(settings)
    }

    /// Saves the object to a file at the given path
    pub fn save_to_file(&self, path: std::path::PathBuf) -> Result<(), std::io::Error> {
        // Create dir if not existing
        std::fs::create_dir_all(&path)?;
        // Also save the SimulationSettings into the same folder
        let mut save_path = path.clone();
        save_path.push(SIM_SETTINGS);
        let f = std::fs::File::create(save_path)?;
        let writer = std::io::BufWriter::new(f);
        match serde_json::to_writer_pretty(writer, &self) {
            Err(e) => {
                return Err(e.into());
            }
            _ => (),
        };
        Ok(())
    }
}

fn generate_particle_pos_spherical(
    simulation_settings: &SimulationSettings,
    rng: &mut ChaCha8Rng,
    is_cargo: bool,
) -> [f64; 3] {
    let middle = simulation_settings.domain_size / 2.0;
    let mut generate_position = |lower_radius: f64, upper_radius: f64| -> [f64; 3] {
        let r = rng.gen_range(lower_radius..upper_radius);
        let phi = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let theta = rng.gen_range(0.0..std::f64::consts::PI);
        [
            middle + r * phi.cos() * theta.sin(),
            middle + r * phi.sin() * theta.sin(),
            middle + r * theta.cos(),
        ]
    };
    let pos = if is_cargo {
        generate_position(0.0, simulation_settings.domain_cargo_radius_max)
    } else {
        generate_position(
            simulation_settings.domain_atg11w19_radius_min,
            simulation_settings.domain_size / 2.0,
        )
    };
    pos
}

#[test]
fn test_particle_pos_config() {
    let simulation_settings = SimulationSettings::new();
    let mut rng = ChaCha8Rng::seed_from_u64(simulation_settings.random_seed);

    for n in 0..simulation_settings.n_cells_cargo + simulation_settings.n_cells_atg11w19 {
        let pos = generate_particle_pos_spherical(
            &simulation_settings,
            &mut rng,
            n < simulation_settings.n_cells_cargo,
        );
        for i in 0..3 {
            assert!(0.0 <= pos[i]);
            assert!(pos[i] <= simulation_settings.domain_size);
        }
    }
}

fn calculate_interaction_range_max(simulation_settings: &SimulationSettings) -> f64 {
    // Calculate the maximal interaction range
    let i1 = simulation_settings.interaction_range_cargo_cargo;
    let i2 = simulation_settings.interaction_range_atg11w19_cargo;
    let i3 = simulation_settings.interaction_range_atg11w19_atg11w19;
    let imax = i1.max(i2).max(i3);

    let r1 = simulation_settings.cell_radius_cargo;
    let r2 = simulation_settings.cell_radius_atg11w19;
    let rmax = r1.max(r2);

    2.0 * rmax + imax
}

fn create_particle_mechanics(
    simulation_settings: &SimulationSettings,
    rng: &mut ChaCha8Rng,
    is_cargo: bool,
) -> Brownian3D {
    let pos = generate_particle_pos_spherical(simulation_settings, rng, is_cargo);
    let kb_temperature = if is_cargo {
        simulation_settings.temperature_cargo * BOLTZMANN_CONSTANT
    } else {
        simulation_settings.temperature_atg11w19 * BOLTZMANN_CONSTANT
    };
    let diffusion = if is_cargo {
        simulation_settings.diffusion_cargo
    } else {
        simulation_settings.diffusion_atg11w19
    };
    Brownian3D::new(
        pos,
        diffusion,
        kb_temperature,
    )
}

fn create_particle_interaction(
    simulation_settings: &SimulationSettings,
    is_cargo: bool,
) -> TypedInteraction {
    TypedInteraction::new(
        if is_cargo {
            Species::Cargo
        } else {
            Species::Atg11w19
        },
        if is_cargo {
            simulation_settings.cell_radius_cargo
        } else {
            simulation_settings.cell_radius_atg11w19
        },
        simulation_settings.potential_strength_cargo_cargo, // potential_strength_cargo_cargo
        simulation_settings.potential_strength_atg11w19_atg11w19, // potential_strength_atg11w19_atg11w19
        simulation_settings.potential_strength_cargo_atg11w19, // potential_strength_cargo_atg11w19
        simulation_settings.interaction_range_cargo_cargo,     // interaction_range_cargo_cargo
        simulation_settings.interaction_range_atg11w19_atg11w19, // interaction_range_atg11w19_atg11w19
        simulation_settings.interaction_range_atg11w19_cargo,    // interaction_range_atg11w19_cargo
    )
}

///
#[pyclass]
#[derive(Clone)]
pub struct Storager {
    manager: StorageManager<
        chili::CellIdentifier,
        (
            chili::CellBox<Particle>,
            _CrAuxStorage<Vector3<f64>, Vector3<f64>, Vector3<f64>, 2>,
        ),
    >,
}

#[pymethods]
impl Storager {
    /// Construct a [Storager] from the given path.
    #[staticmethod]
    pub fn from_path(path: std::path::PathBuf, cargo: bool, date: Option<std::path::PathBuf>) -> PyResult<Self> {
        let full_path = match &date {
            Some(date) => path.join(date),
            None => path,
        };
        let simulation_settings = SimulationSettings::load_from_file(full_path.join(SIM_SETTINGS))?;
        let builder = construct_storage_builder(&simulation_settings, cargo).suffix("cells");
        let manager = match date {
            Some(date) => {
                let builder = builder.init_with_date(&date);
                StorageManager::open_or_create(builder, 0)
            }
            None => {
                println!("no date");
                let builder = builder.add_date(false).init();
                StorageManager::open_or_create(builder, 0)
            }
        }
        .or_else(|e| Err(chili::SimulationError::from(e)))?;
        Ok(Storager { manager })
    }

    fn load_all_particles_at_iteration(
        &self,
        iteration: u64,
    ) -> PyResult<std::collections::HashMap<chili::CellIdentifier, Particle>> {
        Ok(self
            .manager
            .load_all_elements_at_iteration(iteration)
            .or_else(|e| Err(chili::SimulationError::from(e)))?
            .into_iter()
            .map(|(x, (y, _))| (x, y.cell))
            .collect())
    }

    fn load_all_particles(
        &self,
    ) -> PyResult<
        std::collections::HashMap<u64, std::collections::HashMap<chili::CellIdentifier, Particle>>,
    > {
        Ok(self
            .manager
            .load_all_elements()
            .or_else(|e| Err(chili::SimulationError::from(e)))?
            .into_iter()
            .map(|(x, particles)| {
                (
                    x,
                    particles
                        .into_iter()
                        .map(|(px, (py, _))| (px, py.cell))
                        .collect(),
                )
            })
            .collect())
    }

    fn get_all_iterations(&self) -> PyResult<Vec<u64>> {
        Ok(self
            .manager
            .get_all_iterations()
            .or_else(|e| Err(chili::SimulationError::from(e)))?)
    }

    fn load_single_element(
        &self,
        iteration: u64,
        identifier: chili::CellIdentifier,
    ) -> PyResult<Option<Particle>> {
        Ok(self
            .manager
            .load_single_element(iteration, &identifier)
            .or_else(|e| Err(chili::SimulationError::from(e)))?
            .and_then(|p| Some(p.0.cell)))
    }
}

/// Returns dates and the settings structs which are currently present
fn get_all_cargo_simulation_settings(
    simulation_settings: &SimulationSettings,
) -> Vec<(std::path::PathBuf, SimulationSettings)> {
    let mut outputs = Vec::new();
    let mut action = || -> Result<(), std::io::Error> {
        for subfolder in std::fs::read_dir(&simulation_settings.cargo_initials_dir)?.into_iter() {
            let path = subfolder?.path();
            let new_path = path.join(SIM_SETTINGS);
            let sim_settings = SimulationSettings::load_from_file(new_path)?;
            outputs.push((path, sim_settings));
        }
        Ok(())
    };
    match action() {
        Err(_) => outputs,
        Ok(()) => outputs,
    }
}

/// Returns true if both simulation settings contain equal parameters specifically for the cargo
/// particles but only interactions between cargo particles (not cargo - atg11w19).
fn compare_cargo_properties(
    settings1: &SimulationSettings,
    settings2: &SimulationSettings,
) -> bool {
    let uncertainty = 1e-5;
    approx::abs_diff_eq!(settings1.n_cells_cargo, settings2.n_cells_cargo)
        && approx::abs_diff_eq!(
            settings1.cell_radius_cargo,
            settings2.cell_radius_cargo,
            epsilon = uncertainty * NANOMETRE
        )
        && approx::abs_diff_eq!(
            settings1.diffusion_cargo,
            settings2.diffusion_cargo,
            epsilon = uncertainty * MICROMETRE.powf(2.0) / SECOND
        )
        && approx::abs_diff_eq!(
            settings1.temperature_cargo,
            settings1.temperature_cargo,
            epsilon = uncertainty * KELVIN
        )
        && approx::abs_diff_eq!(
            settings1.potential_strength_cargo_cargo,
            settings2.potential_strength_cargo_cargo,
            epsilon = uncertainty * NANOMETRE.powf(2.0) / SECOND.powf(2.0)
        )
        && approx::abs_diff_eq!(
            settings1.interaction_range_cargo_cargo,
            settings2.interaction_range_cargo_cargo,
            epsilon = uncertainty * NANOMETRE
        )
        && approx::abs_diff_eq!(settings1.dt, settings2.dt, epsilon = uncertainty * SECOND)
        && approx::abs_diff_eq!(settings1.t_max, settings2.t_max, epsilon = uncertainty * SECOND)
        && settings1.n_threads == settings2.n_threads
        && approx::abs_diff_eq!(
            settings1.domain_size,
            settings2.domain_size,
            epsilon = uncertainty * NANOMETRE
        )
        && approx::abs_diff_eq!(
            settings1.domain_cargo_radius_max,
            settings2.domain_cargo_radius_max,
            epsilon = uncertainty * NANOMETRE
        )
        && settings1.domain_n_voxels == settings2.domain_n_voxels
        && settings1.random_seed == settings2.random_seed
}

/// Returns either loaded or calculated positions of cargo particles.
/// Will also return if these were loaded
fn prepare_cargo_particles(
    simulation_settings: &SimulationSettings,
) -> Result<impl IntoIterator<Item = Particle>, pyo3::PyErr> {
    // Define Rng
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);

    // Get settings for all previous runs
    let cargo_sim_settings = get_all_cargo_simulation_settings(simulation_settings);
    // Check if a previous simulation is present with parameters which match current settings
    let cargo_initials_storager: Storager = if let Some((loaded_path, loaded_settings)) =
        cargo_sim_settings
            .into_iter()
            .find(|(_, previous_run_settings)| {
                compare_cargo_properties(&previous_run_settings, simulation_settings)
            }) {
        // If so load positions from there
        Storager::from_path(
            loaded_settings.cargo_initials_dir.clone(),
            true,
            Some(
                loaded_path
                    .strip_prefix(&loaded_settings.cargo_initials_dir)
                    .or_else(|e| {
                        Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!(
                                "StripPrefixError: {e} for paths: {:?} and {:?}",
                                loaded_path, loaded_settings.cargo_initials_dir
                            ),
                        ))
                    })?
                    .into(),
            ),
        )?
    } else {
        // If not run new simulation
        println!("Calculating cargo initial positions");
        let mut simulation_settings_cargo_initials = simulation_settings.clone();
        simulation_settings_cargo_initials.save_interval = simulation_settings.t_max;
        let cargo_agents = (0..simulation_settings_cargo_initials.n_cells_cargo).map(|_| {
            let mechanics =
                create_particle_mechanics(&simulation_settings_cargo_initials, &mut rng, true);
            let interaction =
                create_particle_interaction(&simulation_settings_cargo_initials, true);
            Particle {
                mechanics,
                interaction,
            }
        });
        run_simulation_single(&simulation_settings_cargo_initials, cargo_agents, true)?
    };

    let iterations = cargo_initials_storager.get_all_iterations()?;
    let max_iter = iterations.into_iter().max().unwrap();
    let cells_last_iter = cargo_initials_storager.load_all_particles_at_iteration(max_iter)?;

    // Return obtained positions
    Ok(cells_last_iter.into_iter().map(|(_, cell)| cell))
}

/// Takes [SimulationSettings], runs the full simulation and returns the string of the output
/// directory.
#[pyfunction]
pub fn run_simulation(simulation_settings: SimulationSettings) -> Result<Storager, pyo3::PyErr> {
    let mut rng = ChaCha8Rng::seed_from_u64(simulation_settings.random_seed);
    let cargo_positions = prepare_cargo_particles(&simulation_settings)?;

    let particles = cargo_positions
        .into_iter()
        .map(|p| {
            // Set the diffusion constant of the cargo particles to zero in order to fix them in
            // space.
            let mut particle = p;
            particle.mechanics.diffusion_constant = 0.0;
            particle
        })
        .chain((0..simulation_settings.n_cells_atg11w19).map(|_| {
            let mechanics = create_particle_mechanics(&simulation_settings, &mut rng, false);
            let interaction = create_particle_interaction(&simulation_settings, false);
            Particle {
                mechanics,
                interaction,
            }
        }));

    println!("Running Simulation");
    let storager = run_simulation_single(&simulation_settings, particles, false)?;
    Ok(storager)
}

fn construct_storage_builder(
    simulation_settings: &SimulationSettings,
    access_cargo: bool,
) -> StorageBuilder {
    StorageBuilder::new()
        .location(if access_cargo {
            simulation_settings.cargo_initials_dir.clone()
        } else {
            simulation_settings.storage_name.clone()
        })
        .priority([cellular_raza::core::storage::StorageOption::SerdeJson])
        .add_date(true)
}

chili::prepare_types!(
    aspects: [Mechanics, Interaction]
);

fn run_simulation_single(
    simulation_settings: &SimulationSettings,
    particles: impl IntoIterator<Item = Particle>,
    calculate_cargo: bool,
) -> Result<Storager, chili::SimulationError> {
    let interaction_range_max = calculate_interaction_range_max(&simulation_settings);

    let domain = match simulation_settings.domain_n_voxels {
        Some(n_voxels) => CartesianCuboid3New::from_boundaries_and_n_voxels(
            [0.0; 3],
            [simulation_settings.domain_size; 3],
            [n_voxels; 3],
        ),
        None => CartesianCuboid3New::from_boundaries_and_interaction_ranges(
            [0.0; 3],
            [simulation_settings.domain_size; 3],
            [interaction_range_max; 3],
        ),
    }?;

    let time = cellular_raza::core::time::FixedStepsize::from_partial_save_interval(
        0.0,
        simulation_settings.dt,
        simulation_settings.t_max,
        simulation_settings.save_interval,
    )?;

    let storage = construct_storage_builder(&simulation_settings, calculate_cargo).init();
    let path = storage.get_full_path();
    simulation_settings.save_to_file(path)?;

    let settings = chili::Settings {
        n_threads: simulation_settings.n_threads.try_into().unwrap(),
        time,
        storage,
        show_progressbar: simulation_settings.show_progressbar,
    };

    chili::test_compatibility!(
        agents: particles,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction]
    );
    let storage_access: chili::StorageAccess<_, _> = chili::run_main!(
        agents: particles,
        domain: domain,
        settings: settings,
        aspects: [Mechanics, Interaction],
    )?;
    let manager = storage_access.cells;
    Ok(Storager { manager })
}
