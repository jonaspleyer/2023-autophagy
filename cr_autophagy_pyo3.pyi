from dataclasses import dataclass
from pathlib import Path


def run_simulation(SimulationSettings) -> str:
    pass


@dataclass
class SimulationSettings:
    n_cells_cargo: int
    n_cells_r11: int
    cell_radius_cargo: float
    cell_radius_r11: float
    mass_cargo: float
    mass_r11: float
    damping_cargo: float
    damping_r11: float
    kb_temperature_cargo: float
    kb_temperature_r11: float
    update_interval: int
    potential_strength_cargo_cargo: float
    potential_strength_r11_r11: float
    potential_strength_cargo_r11: float
    potential_strength_cargo_r11_avidity: float
    interaction_range_cargo_cargo: float
    interaction_range_r11_r11: float
    interaction_range_r11_cargo: float
    interaction_relative_neighbour_distance: float
    dt: float
    n_times: int
    save_interval: int
    extra_saves: list[int]
    n_threads: int
    domain_size: float
    domain_cargo_radius_max: float
    domain_r11_radius_min: float
    domain_n_voxels: int
    storage_name: str
    storage_name_add_date: bool
    show_progressbar: bool
    random_seed: int

    def load_from_file(path: Path) -> SimulationSettings:
        pass


class Species():
    pass


@dataclass
class TypedInteraction:
    species: Species
    cell_radius: float
    potential_strength_cargo_cargo: float
    potential_strength_r11_r11: float
    potential_strength_cargo_r11: float
    potential_strength_cargo_r11_avidity: float
    interaction_range_cargo_cargo: float
    interaction_range_r11_r11: float
    interaction_range_r11_cargo: float
    relative_neighbour_distance: float


@dataclass
class Langevin3D:
    pos: list[float]
    diffusion_constant: float
    kb_temperature: float
    update_interval: int
