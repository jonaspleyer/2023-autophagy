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

    simulation_settings.n_times = 30_001
    simulation_settings.dt = 2.5
    simulation_settings.save_interval = 5_000

    simulation_settings.n_threads = 1

    simulation_settings.kb_temperature_r11 = 0.001

    # Other settings
    simulation_settings.show_progressbar = False

    return simulation_settings


def run_single_simulation(i, potential_strength_r11_r11, potential_strength_cargo_r11, potential_strength_cargo_r11_avidity, kb_temperature_r11):
    simulation_settings = create_default_settings()
    simulation_settings.potential_strength_r11_r11 = potential_strength_r11_r11
    simulation_settings.potential_strength_cargo_r11 = potential_strength_cargo_r11
    simulation_settings.potential_strength_cargo_r11_avidity = potential_strength_cargo_r11_avidity
    simulation_settings.kb_temperature_r11 = kb_temperature_r11

    simulation_settings.show_progressbar = False
    simulation_settings.storage_name = f"out/autophagy/explore_parameter_space_2_{i:08}/"
    simulation_settings.storage_name_add_date = False
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
    # Save scatter snapshots
    # for iteration in cra.get_all_iterations(output_path):
    #     cra.save_scatter_snapshot(output_path, iteration)

    # Also create a movie with ffmpeg
    # bashcmd = f"ffmpeg -hide_banner -loglevel panic -y -r 30 -f image2 -pattern_type glob -i '{output_path}/scatterplots/*.png' -c:v h264 -pix_fmt yuv420p -strict -2 {output_path}/scatter_movie.mp4"
    # os.system(bashcmd)

    # Save all snapshots
    # for iteration in cra.get_all_iterations(output_path):
    #     cra.save_snapshot(output_path, iteration)
    max_iter = max(cra.get_all_iterations(output_path))
    # cra.save_snapshot(output_path, max_iter)
    cra.save_cluster_information_plots(output_path, max_iter)
    cra.save_kernel_density(
        output_path,
        max_iter,
        threshold=0.45,
        overwrite=False,
        discretization_factor=0.5,
        bw_method=0.2
    )

    # combine_plots(output_path)
    
    # Also create a movie with ffmpeg
    # bashcmd = f"ffmpeg -hide_banner -loglevel panic -y -r 30 -f image2 -pattern_type glob -i '{output_path}/snapshots/*.png' -c:v h264 -pix_fmt yuv420p -strict -2 {output_path}/snapshot_movie.mp4"
    # os.system(bashcmd)

    return True


def run_pipeline(args):
    output_path = run_single_simulation(*args)
    return postprocessing(output_path)


def sample_parameter_space():
    potential_strength_r11_r11 = np.linspace(0.00, 0.01, 4)
    print(len(potential_strength_r11_r11))
    potential_strength_cargo_r11 = np.linspace(0.00, 0.01, 4)
    print(len(potential_strength_cargo_r11))
    potential_strength_cargo_r11_avidity = np.linspace(0.00, 0.01, 4)
    print(len(potential_strength_cargo_r11_avidity))
    kb_temperature_r11 = np.linspace(0.00, 0.03, 3)
    print(len(kb_temperature_r11))

    entries = [(i, *args) for (i, args) in enumerate(itertools.product(
        potential_strength_r11_r11,
        potential_strength_cargo_r11,
        potential_strength_cargo_r11_avidity,
        kb_temperature_r11,
    ))]
    print(len(entries))
    return entries


if __name__ == "__main__":
    parameter_space = sample_parameter_space()

    with mp.Pool(10) as p:
        paths = list(tqdm.tqdm(p.imap(run_pipeline, parameter_space), total=len(parameter_space)))
