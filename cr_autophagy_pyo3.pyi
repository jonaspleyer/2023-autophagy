from typing import Final
from dataclasses import dataclass
from pathlib import Path

# SI Constants
NANOMETRE: Final[float] = 1e-9
MICROMETRE: Final[float] = 1e-6
SECOND: Final[float] = 1.0
MINUTE: Final[float] = 60.0 * SECOND
HOUR: Final[float] = 60.0 * MINUTE
DAY: Final[float] = 24.0 * HOUR
BOLTZMANN_CONSTANT: Final[float] = 1.380649e-23
KELVIN: Final[float] = 1.0


class Storager:
    @staticmethod
    def from_path(path: Path) -> Storager:
        pass

    def get_output_path(self) -> Path:
        pass


def run_simulation(simulation_settings: SimulationSettings) -> Storager:
    pass


@dataclass
class SimulationSettings:
    # TODO add default values
    n_cells_cargo: int = 200
    n_cells_atg11w19: int = 200
    cell_radius_cargo: float = 100 * NANOMETRE
    cell_radius_atg11w19: float = 100 * NANOMETRE
    diffusion_atg11w19: float = 2e-3 * MICROMETRE**2 / SECOND
    diffusion_cargo: float = 2e-3 * MICROMETRE**2 / SECOND
    temperature_atg11w19: float = 300 * KELVIN
    temperature_cargo: float = 300 * KELVIN
    update_interval: int = 5
    potential_strength_cargo_cargo: float = 6e-4 * MICROMETRE**2 / SECOND**2
    potential_strength_atg11w19_atg11w19: float = 2e-4 * MICROMETRE**2 / SECOND**2
    potential_strength_cargo_atg11w19: float = 1e-4 * MICROMETRE**2 / SECOND**2
    interaction_range_cargo_cargo: float = 0.4 * (cell_radius_cargo + cell_radius_atg11w19)
    interaction_range_atg11w19_atg11w19: float = 0.4 * (cell_radius_cargo + cell_radius_atg11w19)
    interaction_range_atg11w19_cargo: float = 0.4 * (cell_radius_cargo + cell_radius_atg11w19)
    dt: float = 0.001 * MINUTE
    t_max: float = 40 * MINUTE
    save_interval: float = 0.1 * MINUTE
    extra_saves: list[float] = []
    n_threads: int = 1
    domain_size: float = 2000 * NANOMETRE
    domain_cargo_radius_max: float = 600 * NANOMETRE
    domain_atg11w19_radius_min: float = 650 * NANOMETRE
    domain_n_voxels: int | None = 4
    storage_name: Path = Path("out/autophagy")
    substitute_date: str | None = None
    show_progressbar: bool = True
    random_seed: int = 1

    def load_from_file(path: Path) -> SimulationSettings:
        pass

    def approx_eq(self, sim_settings: SimulationSettings) -> bool:
        pass


class Species():
    pass


@dataclass
class TypedInteraction:
    species: Species
    cell_radius: float
    potential_strength_cargo_cargo: float
    potential_strength_atg11w19_atg11w19: float
    potential_strength_cargo_atg11w19: float
    interaction_range_cargo_cargo: float
    interaction_range_atg11w19_atg11w19: float
    interaction_range_atg11w19_cargo: float


@dataclass
class Brownian3D:
    pos: list[float]
    diffusion_constant: float
    kb_temperature: float
    update_interval: int
