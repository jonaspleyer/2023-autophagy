from cr_autophagy_pyo3 import SimulationSettings, run_simulation
import cr_autophagy as cra
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from pathlib import Path
from typing import Optional

def get_previous_simulation_run_opath(simulation_settings: SimulationSettings) -> Optional[Path]:
    for opath in list(glob.glob("out/autophagy/*")):
        opath = Path(opath)
        sim_settings_prev = cra.get_simulation_settings(opath)
        if sim_settings_prev.approx_eq(simulation_settings):
            return opath
    return None

def generate_results(simulation_settings: SimulationSettings):
    # Check if a previous simulation was run with identical settings
    prev_output_path = get_previous_simulation_run_opath(simulation_settings)
    if prev_output_path == None:
        storager = run_simulation(simulation_settings)
        return Path(storager.get_output_path())
    else:
        return prev_output_path

if __name__ == "__main__":
    units = cra.MICROMETRE**2 / cra.SECOND**2
    values_potential_strength_cargo_atg11w19 = units * np.array([2e-1, 4e-1])#, 6e-1, 1e-1]) * units
    values_potential_strength_atg11w19_atg11w19 = units * np.array([2e-1, 6e-1, 1e0])

    def _run_sim(pot_aa, pot_ac, n_threads:int=1):
        simulation_settings = SimulationSettings()
        simulation_settings.n_threads = n_threads
        simulation_settings.n_threads = n_threads
        
        # FIXME this should only be here temporarily
        # simulation_settings.t_max = 0.05 * cra.MINUTE
        simulation_settings.t_max = 4 * cra.MINUTE

        simulation_settings.potential_strength_cargo_atg11w19 = pot_ac
        simulation_settings.potential_strength_atg11w19_atg11w19 = pot_aa

        output_path = generate_results(simulation_settings)
        return (output_path, simulation_settings)

    results = []

    for pot_aa in values_potential_strength_atg11w19_atg11w19:
        for pot_ac in values_potential_strength_cargo_atg11w19:
            out = _run_sim(pot_aa, pot_ac, 5)
            results.append(out)

    fig, ax = plt.subplots(figsize=(8, 6))
    pot_ac_max = np.max(values_potential_strength_cargo_atg11w19)
    pot_aa_max = np.max(values_potential_strength_atg11w19_atg11w19)
    ax.set_xlim([0, 1.2 * pot_ac_max])
    ax.set_ylim([0, 1.2 * pot_aa_max])

    for opath, settings in results:
        # Retrieve information and plot last iteration
        last_iter = np.sort(cra.get_all_iterations(opath))[-1]
        cra.save_snapshot(opath, last_iter, transparent_background=True)
        arr_img = plt.imread("{}/snapshots/snapshot_{:08}.png".format(opath, last_iter))

        sim_settings = cra.get_simulation_settings(opath)
        pot_ac = sim_settings.potential_strength_cargo_atg11w19
        pot_aa = sim_settings.potential_strength_atg11w19_atg11w19

        # Plot the box of the result
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        img = OffsetImage(arr_img, zoom=0.25)
        ab = AnnotationBbox(
            img,
            (pot_ac, pot_aa),
            pad=0,
            box_alignment=(0.5, 0.5),
            annotation_clip=True,
            frameon=False,
        )
        ax.add_artist(ab)
    ax.plot(np.linspace(0, pot_ac_max), np.linspace(0, pot_aa_max))

    ax.set_xlabel("Potential Strength Cargo-Protein")
    ax.set_ylabel("Potential Strength Protein-Protein")
    fig.tight_layout()
    plt.show()
