use cellular_raza::prelude::*;
use nalgebra::Vector3;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Particle Species of type Cargo or Atg11/19
///
/// We currently only distinguish between the cargo itself
/// and freely moving combinations of the receptor and ATG11.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[pyclass(set_all, get_all)]
pub enum Species {
    /// Cargo particle
    Cargo,
    /// Atg11/19 particle which is a combination of receptor and Atg11
    Atg11w19,
}

/// Interaction potential depending on the other cells species.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass(get_all, set_all)]
pub struct TypedInteraction {
    /// Labels particles as Cargo or Atg11/19
    pub species: Species,

    /// Radius of the spherical particle
    pub cell_radius: f64,

    /// Attracting potential strength between individual Cargo particles
    pub potential_strength_cargo_cargo: f64,

    /// Attracting potential strength between individual Atg11/19 particles
    pub potential_strength_atg11w19_atg11w19: f64,

    /// Attracting potential strength between individual Cargo and Atg11/19 particles
    pub potential_strength_cargo_atg11w19: f64,

    /// The interaction range (beyond size of the particles) between Cargo particles
    pub interaction_range_cargo_cargo: f64,

    /// The interaction range (beyond size of the particles) between Atg11/19 particles
    pub interaction_range_atg11w19_atg11w19: f64,

    /// The interaction range (beyond size of the particles) between Cargo and Atg11/19 particles
    pub interaction_range_atg11w19_cargo: f64,

    /// Relative length in units of the diameter of a particle until when to treat nearby particles
    /// of the same species as neighbours
    pub relative_neighbour_distance: f64,

    neighbour_count: usize,
}

impl Interaction<Vector3<f64>, Vector3<f64>, Vector3<f64>, (f64, Species)> for TypedInteraction {
    fn calculate_force_between(
        &self,
        own_pos: &Vector3<f64>,
        _own_vel: &Vector3<f64>,
        ext_pos: &Vector3<f64>,
        _ext_vel: &Vector3<f64>,
        ext_info: &(f64, Species),
    ) -> Result<(Vector3<f64>, Vector3<f64>), CalcError> {
        // Calculate radius and direction
        let min_relative_distance_to_center = 0.3162277660168379;
        let (r, dir) =
            match (own_pos - ext_pos).norm() < self.cell_radius * min_relative_distance_to_center {
                false => {
                    let z = own_pos - ext_pos;
                    let r = z.norm();
                    (r, z.normalize())
                }
                true => {
                    let dir = match own_pos == ext_pos {
                        true => {
                            return Ok((Vector3::zeros(), Vector3::zeros()));
                        }
                        false => (own_pos - ext_pos).normalize(),
                    };
                    let r = self.cell_radius * min_relative_distance_to_center;
                    (r, dir)
                }
            };
        let (ext_radius, ext_species) = ext_info;
        // Introduce Non-dimensional length variable
        let sigma = r / (self.cell_radius + ext_radius);
        let bound = 4.0 + 1.0 / sigma;
        let calculate_cutoff = |interaction_range| {
            if interaction_range + self.cell_radius + ext_radius >= r {
                1.0
            } else {
                0.0
            }
        };

        // Calculate the strength of the interaction with correct bounds
        let strength = ((1.0 / sigma).powf(2.0) - (1.0 / sigma).powf(4.0))
            .min(bound)
            .max(-bound);

        // Calculate only attracting and repelling forces
        let attracting_force = dir * strength.max(0.0);
        let repelling_force = dir * strength.min(0.0);

        match (ext_species, &self.species) {
            // Atg11/19 will bind to cargo
            (Species::Cargo, Species::Atg11w19) | (Species::Atg11w19, Species::Cargo) => {
                let cutoff = calculate_cutoff(self.interaction_range_atg11w19_cargo);
                let force = cutoff
                    * (self.potential_strength_cargo_cargo * repelling_force
                        + self.potential_strength_cargo_atg11w19 * attracting_force);
                Ok((-force, force))
            }

            // Atg11/19 forms clusters
            (Species::Cargo, Species::Cargo) => {
                let cutoff = calculate_cutoff(self.interaction_range_cargo_cargo);
                let force = cutoff
                    * self.potential_strength_cargo_cargo
                    * (repelling_force + attracting_force);
                Ok((-force, force))
            }

            (Species::Atg11w19, Species::Atg11w19) => {
                let cutoff = calculate_cutoff(self.interaction_range_atg11w19_atg11w19);
                let force = cutoff
                    * (self.potential_strength_cargo_cargo * repelling_force
                        + self.potential_strength_atg11w19_atg11w19 * attracting_force);
                Ok((-force, force))
            }
        }
    }

    fn get_interaction_information(&self) -> (f64, Species) {
        (self.cell_radius, self.species.clone())
    }

    fn is_neighbour(
        &self,
        own_pos: &Vector3<f64>,
        ext_pos: &Vector3<f64>,
        ext_inf: &(f64, Species),
    ) -> Result<bool, CalcError> {
        match (&self.species, &ext_inf.1) {
            (Species::Atg11w19, Species::Atg11w19) | (Species::Cargo, Species::Cargo) => {
                Ok((own_pos - ext_pos).norm() <= self.relative_neighbour_distance * (self.cell_radius + ext_inf.0))
            }
            _ => Ok(false),
        }
    }

    fn react_to_neighbours(&mut self, neighbours: usize) -> Result<(), CalcError> {
        Ok(self.neighbour_count = neighbours)
    }
}

impl Volume for Particle {
    fn get_volume(&self) -> f64 {
        1.0
    }
}

#[pymethods]
impl TypedInteraction {
    #[new]
    #[pyo3(signature = (
        species,
        cell_radius,
        potential_strength_cargo_cargo,
        potential_strength_atg11w19_atg11w19,
        potential_strength_cargo_atg11w19,
        interaction_range_cargo_cargo,
        interaction_range_atg11w19_atg11w19,
        interaction_range_atg11w19_cargo,
        relative_neighbour_distance,
    ))]
    /// Constructs a new TypedInteraction
    pub fn new(
        species: Species,
        cell_radius: f64,
        potential_strength_cargo_cargo: f64,
        potential_strength_atg11w19_atg11w19: f64,
        potential_strength_cargo_atg11w19: f64,
        interaction_range_cargo_cargo: f64,
        interaction_range_atg11w19_atg11w19: f64,
        interaction_range_atg11w19_cargo: f64,
        relative_neighbour_distance: f64,
    ) -> Self {
        Self {
            species,
            cell_radius,
            potential_strength_cargo_cargo,
            potential_strength_atg11w19_atg11w19,
            potential_strength_cargo_atg11w19,
            interaction_range_cargo_cargo,
            interaction_range_atg11w19_atg11w19,
            interaction_range_atg11w19_cargo,
            relative_neighbour_distance,
            neighbour_count: 0,
        }
    }
}

/// Cargo or Atg11/19 particle depending on the [Species] of the interaction field.
#[pyclass(get_all, set_all)]
#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
pub struct Particle {
    /// The [Langevin3D] motion was chosen to model mechanics of the particle movement
    #[Mechanics]
    pub mechanics: Brownian3D,

    /// The [TypedInteraction] assigns the [Species] to the particle and handles
    /// calculation of interactions.
    #[Interaction]
    pub interaction: TypedInteraction,
}
