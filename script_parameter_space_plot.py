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
    values_potential_strength_cargo_atg11w19 = units * np.array([1e-2, 4e-1, 16e0])
    values_potential_strength_atg11w19_atg11w19 = units * np.array([2e-1, 1e0, 5e0])

    def _run_sim(n_run: int, pot_aa: float, pot_ac: float, n_threads:int=1) -> tuple[Path, SimulationSettings]:
        simulation_settings = SimulationSettings()
        simulation_settings.storage_name = OUT_PATH
        simulation_settings.substitute_date = str("{:010}".format(n_run))
        simulation_settings.n_threads = n_threads
        simulation_settings.show_progressbar = False
        simulation_settings.domain_size *= 1.5

        simulation_settings.t_max = 6 * cra.MINUTE
        simulation_settings.dt *= 10

        simulation_settings.potential_strength_cargo_atg11w19 = pot_ac
        simulation_settings.potential_strength_atg11w19_atg11w19 = pot_aa
        # simulation_settings.diffusion_atg11w19 = 8e-5 * MICROMETRE**2 / SECOND

        simulation_settings.interaction_range_atg11w19_cargo *= 0.5

        return generate_results(simulation_settings)

    n_threads = 2
    n_cores = mp.cpu_count()
    n_workers = max(1, math.floor(n_cores / n_threads))

    def __run_sim_helper(args: list):
        return _run_sim(*args)

    n_prev_runs = max([int(str(Path(p)).split("/")[-1]) for p in glob.glob(str(OUT_PATH) + "/*")] + [0]) + 1
    values = list(map(lambda x: (x[0], *x[1], *x[2:]),
        zip(
            itertools.count(n_prev_runs),
            itertools.product(
                values_potential_strength_atg11w19_atg11w19,
                values_potential_strength_cargo_atg11w19
            ),
            itertools.repeat(n_threads),
    )))

    pool = mp.Pool(n_workers)
    results = list(pool.imap(__run_sim_helper, values))
    results = [(opath, cra.get_simulation_settings(opath)) for opath in results]

    fig, ax = plt.subplots(figsize=(16, 12))
    pot_ac_max = np.max([sims[1].potential_strength_cargo_atg11w19 for sims in results])
    pot_ac_min = np.min([sims[1].potential_strength_cargo_atg11w19 for sims in results])
    pot_aa_max = np.max([sims[1].potential_strength_atg11w19_atg11w19 for sims in results])
    pot_aa_min = np.min([sims[1].potential_strength_atg11w19_atg11w19 for sims in results])
    ax.set_xlim([pot_ac_min / 5, 5 * pot_ac_max])
    ax.set_ylim([pot_aa_min / 5, 5 * pot_aa_max])
    ax.set_xscale("log")
    ax.set_yscale("log")

    for opath, settings in results:
        # Retrieve information and plot last iteration
        last_iter = np.sort(cra.get_all_iterations(opath))[-1]
        try:
            arr_img = cra.save_snapshot(opath, last_iter, transparent_background=True, overwrite=True)
        except:
            print("Failed to plot results from {}".format(opath))
            continue

        sim_settings = cra.get_simulation_settings(opath)
        pot_ac = sim_settings.potential_strength_cargo_atg11w19
        pot_aa = sim_settings.potential_strength_atg11w19_atg11w19

        # Plot the box of the result
        img = OffsetImage(arr_img, zoom=0.45)
        ab = AnnotationBbox(
            img,
            (pot_ac, pot_aa),
            pad=0,
            box_alignment=(0.5, 0.5),
            annotation_clip=True,
            frameon=False,
        )
        ax.add_artist(ab)

    ax.set_xlabel("Potential Strength Cargo-Protein")
    ax.set_ylabel("Potential Strength Protein-Protein")
    fig.tight_layout()
    fig.savefig("parameter_space_plt.png")
