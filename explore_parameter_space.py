from cr_autophagy_pyo3 import SimulationSettings, run_simulation
import cr_autophagy as cra
import numpy as np
import itertools
import multiprocessing as mp
import os
import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import json


def create_default_settings():
    simulation_settings = SimulationSettings()

    # Settings Cargo
    simulation_settings.potential_strength_r11_r11 = 0.002
    simulation_settings.potential_strength_cargo_r11 = 0.0
    simulation_settings.potential_strength_cargo_r11_avidity = 0.02

    simulation_settings.n_times = 45_001
    simulation_settings.dt = 1.5
    simulation_settings.save_interval = 15_000

    simulation_settings.extra_saves = np.arange(44_000, 45_001, 200)

    simulation_settings.n_threads = 1

    simulation_settings.kb_temperature_r11 = 0.001

    # Other settings
    simulation_settings.show_progressbar = False

    return simulation_settings


def run_single_simulation(
        i,
        potential_strength_r11_r11,
        potential_strength_cargo_r11,
        potential_strength_cargo_r11_avidity,
        interaction_relative_neighbour_distance,
        kb_temperature_r11,
        seed,
    ):
    simulation_settings = create_default_settings()
    simulation_settings.potential_strength_r11_r11 = potential_strength_r11_r11
    simulation_settings.potential_strength_cargo_r11 = potential_strength_cargo_r11
    simulation_settings.potential_strength_cargo_r11_avidity = potential_strength_cargo_r11_avidity
    simulation_settings.interaction_relative_neighbour_distance = interaction_relative_neighbour_distance
    simulation_settings.kb_temperature_r11 = kb_temperature_r11

    simulation_settings.show_progressbar = False
    simulation_settings.storage_name = f"out/autophagy/explore_parameter_space_2_{i:08}/"
    simulation_settings.storage_name_add_date = False
    simulation_settings.random_seed = seed
    output_path = Path(simulation_settings.storage_name)

    # Skip if folder already exists
    if os.path.isdir(output_path):
        return output_path
    else:
        output_path = run_simulation(simulation_settings)
    return Path(output_path)


def combine_plots(output_path):
    number = str(output_path).split("/")[-2].split("_")[-1]
    f = open(output_path / "simulation_settings.json")
    simulation_settings = json.load(f)

    potential_strength_r11_r11 = simulation_settings["potential_strength_r11_r11"]
    potential_strength_cargo_r11 = simulation_settings["potential_strength_cargo_r11"]
    potential_strength_cargo_r11_avidity = simulation_settings["potential_strength_cargo_r11_avidity"]
    kb_temperature_r11 = simulation_settings["kb_temperature_r11"]

    if potential_strength_cargo_r11>=1.1:
        return None

    cell_text = []
    cell_text.append(["potential_strength_r11_r11", potential_strength_r11_r11])
    cell_text.append(["potential_strength_cargo_r11", potential_strength_cargo_r11])
    cell_text.append(["potential_strength_cargo_r11_avidity", potential_strength_cargo_r11_avidity])
    cell_text.append(["kb_temperature_r11", kb_temperature_r11])

    max_iter = max(cra.get_all_iterations(output_path))
    im = plt.imread(f"{output_path}/snapshots/snapshot_{max_iter:08}.png")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    ax.table(cell_text)
    ax.set_title(f"Number {number}")
    ax.imshow(im)
    fig.tight_layout()
    plt.savefig(f"param_space/combined_{number}.png")
    plt.close(fig)


def postprocessing(output_path):
    return True


def run_pipeline(args):
    output_path = run_single_simulation(*args)
    return postprocessing(output_path)


def sample_parameter_space():
    potential_strength_r11_r11 = np.linspace(0.00, 0.01, 5)
    print(len(potential_strength_r11_r11))
    potential_strength_cargo_r11 = np.linspace(0.00, 0.01, 5)
    print(len(potential_strength_cargo_r11))
    potential_strength_cargo_r11_avidity = np.linspace(0.00, 0.01, 5)
    print(len(potential_strength_cargo_r11_avidity))
    kb_temperature_r11 = np.linspace(0.00, 0.03, 6)
    print(len(kb_temperature_r11))
    seeds = np.arange(8)
    print(len(seeds))
    interaction_relative_neighbour_distance = np.arange(1.8, 2.3, 0.1)

    entries = [(i, *args) for (i, args) in enumerate(itertools.product(
        potential_strength_r11_r11,
        potential_strength_cargo_r11,
        potential_strength_cargo_r11_avidity,
        interaction_relative_neighbour_distance,
        kb_temperature_r11,
        seeds,
    ))]
    print(len(entries))
    return entries


if __name__ == "__main__":
    parameter_space = sample_parameter_space()

    with mp.Pool(40) as p:
        paths = list(tqdm.tqdm(p.imap(run_pipeline, parameter_space), total=len(parameter_space)))
