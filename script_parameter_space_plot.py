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

if __name__ == "__main__":
    units = cra.MICROMETRE**2 / cra.SECOND**2
    values_potential_strength_cargo_atg11w19 = units * np.array([0.0, 5e-2, 1e-1, 5e-1, 1e1])
    values_potential_strength_atg11w19_atg11w19 = units * np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    def _run_sim(
        n_run: int,
        ny_pot_aa: tuple[int, float],
        nx_pot_ac: tuple[int, float],
        n_threads:int=1
    ) -> tuple[int, int, Path, SimulationSettings]:
        simulation_settings = SimulationSettings()
        simulation_settings.storage_name = OUT_PATH
        simulation_settings.substitute_date = str("{:010}".format(n_run))
        simulation_settings.n_threads = n_threads
        simulation_settings.show_progressbar = False
        simulation_settings.domain_size *= 3
        simulation_settings.n_cells_atg11w19 *= 2

        factor = 2
        simulation_settings.t_max = 40 * cra.MINUTE
        simulation_settings.dt *= 10 / factor

        simulation_settings.potential_strength_cargo_atg11w19 = nx_pot_ac[1]
        simulation_settings.potential_strength_atg11w19_atg11w19 = ny_pot_aa[1]
        simulation_settings.potential_strength_cargo_cargo *= factor
        # simulation_settings.diffusion_atg11w19 = 8e-5 * MICROMETRE**2 / SECOND

        simulation_settings.interaction_range_atg11w19_cargo *= 0.5

        return (ny_pot_aa[0], nx_pot_ac[0], *generate_results(simulation_settings))

    n_threads = 2
    n_cores = mp.cpu_count()
    n_workers = max(1, math.floor(n_cores / n_threads))

    def __run_sim_helper(args: list) -> tuple[int, int, Path, SimulationSettings]:
        return _run_sim(*args)

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
            itertools.repeat(n_threads),
    )))

    pool = mp.Pool(n_workers)
    import tqdm
    print("Get Results")
    results = list(tqdm.tqdm(pool.imap(__run_sim_helper, values), total=len(values)))

    figsize_x = 12
    figsize_y = (1 + len(values_potential_strength_atg11w19_atg11w19))\
        / (1 + len(values_potential_strength_cargo_atg11w19))\
        * figsize_x
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
    ax.set_xlim([0, len(values_potential_strength_cargo_atg11w19)+1])
    ax.set_ylim([0, len(values_potential_strength_atg11w19_atg11w19)+1])

    print("Create Plot")
    for ny, nx, opath, sim_settings in tqdm.tqdm(results, total=len(results)):
        # Retrieve information and plot last iteration
        last_iter = np.sort(cra.get_all_iterations(opath))[-1]
        try:
            arr_img = cra.save_snapshot(
                opath,
                last_iter,
                transparent_background=True,
                overwrite=True
            )
        except:
            print("Failed to plot results from {}".format(opath))

        # Plot the box of the result
        img = OffsetImage(arr_img, zoom=0.35)
        ab = AnnotationBbox(
            img,
            (nx+1, ny+1),# (pot_ac, pot_aa),
            pad=0,
            box_alignment=(0.5, 0.5),
            annotation_clip=True,
            frameon=False,
        )
        ax.add_artist(ab)

    ax.set_xticks(
            range(1, 1+len(values_potential_strength_cargo_atg11w19)),
        values_potential_strength_cargo_atg11w19
    )
    ax.set_yticks(
        range(1, 1+len(values_potential_strength_atg11w19_atg11w19)),
        values_potential_strength_atg11w19_atg11w19
    )

    ax.set_xlabel("Potential Strength Cargo-Protein")
    ax.set_ylabel("Potential Strength Protein-Protein")
    fig.tight_layout()
    fig.savefig("parameter_space_plt.png")
