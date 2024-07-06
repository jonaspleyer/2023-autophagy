from cr_autophagy_pyo3 import SimulationSettings, run_simulation
import cr_autophagy as cra
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
import glob
import multiprocessing as mp
import math
import itertools
import tqdm
import os

OUT_PATH: Path = Path("out/autophagy_param_space")

def get_previous_simulation_run_opath(simulation_settings: SimulationSettings) -> tuple[Path, SimulationSettings] | None:
    for opath in list(glob.glob(str(OUT_PATH) + "/*")):
        opath = Path(opath)
        try:
            sim_settings_prev = cra.get_simulation_settings(opath)
            if sim_settings_prev is not None and sim_settings_prev.approx_eq(simulation_settings):
                return opath, sim_settings_prev
        finally:
            pass
    return None

def generate_results(simulation_settings: SimulationSettings) -> tuple[Path, SimulationSettings]:
    # Check if a previous simulation was run with identical settings
    previous_result = get_previous_simulation_run_opath(simulation_settings)
    if previous_result is None:
        sim_settings, storager = run_simulation(simulation_settings)
        return (Path(storager.get_output_path()), sim_settings)
    else:
        return previous_result

def _run_sim(
    n_run: int,
    ny_pot_aa: tuple[int, float],
    nx_pot_ac: tuple[int, float],
    n_threads:int=1
) -> tuple[int, int, Path, SimulationSettings]:
    simulation_settings = SimulationSettings()
    simulation_settings.random_seed += 3
    simulation_settings.storage_name = OUT_PATH
    simulation_settings.substitute_date = str("{:010}".format(n_run))
    simulation_settings.n_threads = n_threads
    simulation_settings.show_progressbar = False
    simulation_settings.domain_size *= 2
    simulation_settings.n_cells_atg11w19 = round(2 * simulation_settings.n_cells_atg11w19)
    simulation_settings.save_interval *= 10
    simulation_settings.diffusion_atg11w19 *= 0.8

    factor = 1
    simulation_settings.t_max = 80 * cra.MINUTE
    simulation_settings.dt *= 8 / factor

    simulation_settings.potential_strength_cargo_atg11w19 = nx_pot_ac[1]
    simulation_settings.potential_strength_atg11w19_atg11w19 = ny_pot_aa[1]
    simulation_settings.potential_strength_cargo_cargo *= factor

    simulation_settings.interaction_range_atg11w19_cargo *= 0.75
    return (ny_pot_aa[0], nx_pot_ac[0], *generate_results(simulation_settings))

def __run_sim_helper(args: list) -> tuple[int, int, Path, SimulationSettings]:
    return _run_sim(*args)

def plot_with_angle(
        angle: float = 0.0,
        show_progressbar: bool = True,
        parallelize: bool = True,
        custom_suffix: str | None = None,
        headless: bool = False,
        simulation_threads: int = 4,
        total_threads: int = mp.cpu_count(),
    ):
    if headless:
        import pyvista as pv
        pv.start_xvfb()
    units = cra.MICROMETRE**2 / cra.SECOND**2
    values_potential_strength_cargo_atg11w19 = units * np.array([0.0, 0.5e-1, 1e-1, 2e-1, 3e-1, 4e-1, 1e0])# 2e-1, 3e-1, 4e-1, 5e-1, 1e0])# 2.5e-1, 5e-1, 7.5e-1, 1e0, 2.5e0, 5e0, 7.5e0, 1e1])
    values_potential_strength_atg11w19_atg11w19 = units * np.array([0.2, 0.3, 0.4, 0.5, 0.55])# , 0.56, 0.57, 0.58, 0.59, 0.6])# 0.5, 0.6, 0.7, 0.8, 0.9])

    n_workers = max(1, math.floor(total_threads / simulation_threads))

    n_prev_runs = max([
        int(str(Path(p)).split("/")[-1])
        for p in glob.glob(str(OUT_PATH) + "/*")
    ] + [0]) + 1
    values = list(map(lambda x: (x[0], *x[1], *x[2:]),
        zip(
            itertools.count(n_prev_runs),
            itertools.product(
                enumerate(values_potential_strength_atg11w19_atg11w19),
                enumerate(values_potential_strength_cargo_atg11w19),
            ),
            itertools.repeat(simulation_threads),
    )))

    print("Get Results")
    if parallelize and show_progressbar:
        pool = mp.Pool(n_workers)
        results = list(tqdm.tqdm(pool.imap(__run_sim_helper, values), total=len(values)))
    elif not parallelize and show_progressbar:
        results = [__run_sim_helper(v) for v in values]
    elif parallelize and not show_progressbar:
        pool = mp.Pool(n_workers)
        results = list(pool.imap(__run_sim_helper, values), total=len(values))
    else:
        results = [__run_sim_helper(v) for v in values]

    figsize_x = len(values_potential_strength_cargo_atg11w19) * 4
    figsize_y = 1.2 * (1 + len(values_potential_strength_atg11w19_atg11w19))\
        / (1 + len(values_potential_strength_cargo_atg11w19))\
        * figsize_x
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
    ax.set_xlim([0, len(values_potential_strength_cargo_atg11w19)+1])
    ax.set_ylim([0, len(values_potential_strength_atg11w19_atg11w19)+1])

    print("Creating Plots")
    if show_progressbar:
        iterator_list = tqdm.tqdm(results, total=len(results))
    else:
        iterator_list = results
    for ny, nx, opath, sim_settings in iterator_list:
        # Retrieve information and plot last iteration
        iterations = np.sort(cra.get_all_iterations(opath))
        last_iter = iterations[-1]
        arr_img = cra.save_snapshot(
            opath,
            last_iter,
            transparent_background=True,
            overwrite=True,
            view_angles=(angle, 0, 0),
            scale=2,
        )
        # Plot the box of the result
        if arr_img is not None:
            img = OffsetImage(arr_img, zoom=0.2)
            ab = AnnotationBbox(
                img,
                (nx+1, ny+1),# (pot_ac, pot_aa),
                pad=0,
                box_alignment=(0.5, 0.5),
                annotation_clip=True,
                frameon=False,
            )
            ax.add_artist(ab)

    # Plot table with settings
    variables = [
        "n_cells_atg11w19",
        "n_cells_cargo",

        "cell_radius_cargo",
        "cell_radius_atg11w19",

        "diffusion_cargo",
        "diffusion_atg11w19",

        "temperature_atg11w19",
        "temperature_cargo",

        "potential_strength_cargo_cargo",
        "potential_strength_cargo_atg11w19",
        "potential_strength_atg11w19_atg11w19",

        "interaction_range_cargo_cargo",
        "interaction_range_atg11w19_cargo",
        "interaction_range_atg11w19_atg11w19",
        "relative_neighbour_distance",

        "dt",
        "t_max",
        "save_interval",
        "n_threads",

        "domain_size",
        "domain_cargo_radius_max",
        "domain_atg11w19_radius_min",
        "domain_n_voxels",

        "substitute_date",
        "show_progressbar",
        "random_seed",
    ]
    # Get variables from first element of the results list
    if len(results) > 0:
        _, _, _, sim_settings = results[0]
        def format_entry(x):
            if type(x) is float:
                return "{:5.2e}".format(x)
            else:
                return str(x)
        table = ax.table(
                cellText=[
                    (var_name, format_entry(getattr(sim_settings, var_name)))# "{:5.2e}".format(getattr(sim_settings, var_name)))
                    for var_name in variables
                ],
            loc='right',
            cellLoc='left',
            fontsize=8,
        )
        table.auto_set_column_width(0)
        table.auto_set_column_width(1)

    ax.set_xticks(
        range(1, 1+len(values_potential_strength_cargo_atg11w19)),
        ["{:5.2e}".format(v) for v in values_potential_strength_cargo_atg11w19]
    )
    ax.set_yticks(
        range(1, 1+len(values_potential_strength_atg11w19_atg11w19)),
        ["{:5.2e}".format(v) for v in values_potential_strength_atg11w19_atg11w19]
    )

    ax.set_xlabel("Potential Strength Cargo-Protein")
    ax.set_ylabel("Potential Strength Protein-Protein")
    fig.tight_layout()

    # Create new file every time a new plot is done
    def set_save_name(suffix):
        # Create folder if it does not exist
        path = Path("parameter-space-plt")
        path.mkdir(parents=True, exist_ok=True)
        return path / "snapshot-{}.png".format(suffix)
    if custom_suffix is None:
        i = 0
        save_name = set_save_name("{:06}".format(0))
        while os.path.exists(save_name):
            i += 1
            save_name = set_save_name("{:06}".format(i))
    else:
        save_name = set_save_name(custom_suffix)
    print("Saving under {}".format(save_name))
    fig.savefig(save_name)

def _plotter(angle):
    return plot_with_angle(
        angle,
        show_progressbar=False,
        parallelize=False,
        custom_suffix="angle-{:03.0f}".format(angle)
    )

if __name__ == "__main__":
    plot_with_angle(112, total_threads=12)
    # pool = mp.Pool(20)
    # _ = list(pool.map(_plotter, np.arange(0, 360, 4)))
    # for angle in np.linspace(0, 360, 30):
    #     _plotter(angle)
