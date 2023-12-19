from cr_autophagy_pyo3 import SimulationSettings, run_simulation
import cr_autophagy as cra
import numpy as np


if __name__ == "__main__":
    simulation_settings = SimulationSettings()

    ##### set biological simulation settings

    # mass and size
    simulation_settings.cell_radius_r11 = 10.0 # nm
    simulation_settings.cell_radius_cargo = simulation_settings.cell_radius_r11
    simulation_settings.mass_r11 = 4.0 / 3.0 * np.pi * simulation_settings.cell_radius_r11**3
    simulation_settings.mass_cargo = 3 * simulation_settings.mass_r11

    # number of cells
    simulation_settings.n_cells_cargo = 420
    simulation_settings.n_cells_r11 = 55

    # domain size
    simulation_settings.domain_size = 200.0
    simulation_settings.domain_cargo_radius_max = 80.0
    simulation_settings.domain_r11_radius_min = 85.0

    # interaction ranges
    simulation_settings.interaction_range_cargo_cargo = 0.4 * (simulation_settings.cell_radius_cargo + simulation_settings.cell_radius_r11)
    simulation_settings.interaction_range_r11_r11 = 0.4 * (simulation_settings.cell_radius_cargo + simulation_settings.cell_radius_r11)
    simulation_settings.interaction_range_r11_cargo = 0.4 * (simulation_settings.cell_radius_cargo + simulation_settings.cell_radius_r11)
    simulation_settings.interaction_relative_neighbour_distance = 2.0


    print(simulation_settings.cell_radius_r11)

    # HIGH AFFINITY CONFIGURATION
    #simulation_settings.potential_strength_cargo_r11 = 0.01
    #simulation_settings.potential_strength_cargo_r11_avidity = 0.00

    # HIGH AVIDITY CONFIGURATION
    simulation_settings.potential_strength_cargo_cargo = 2.0
    simulation_settings.potential_strength_r11_r11 = 0.6
    simulation_settings.potential_strength_cargo_r11 = 0.1
    simulation_settings.potential_strength_cargo_r11_avidity = 1.0

    simulation_settings.n_times = 60_001
    simulation_settings.dt = 2.5
    simulation_settings.save_interval = 1_000
    simulation_settings.extra_saves = np.arange(29_000, 30_001, 100)

    simulation_settings.n_threads = 1

    simulation_settings.kb_temperature_r11 = 0.02

    from pathlib import Path
    import os
    output_path = Path(run_simulation(simulation_settings))

    print("Saving Snapshots")
    cra.save_all_snapshots(output_path, threads=14)#simulation_settings.n_threads)

    print("Saving cluster analysis plots")
   # cra.save_all_cluster_information_plots(output_path, threads=0,show_bar=True, connection_distance=2.5)

    print("Saving kde plots")
   # cra.save_all_kernel_density(output_path, threads=0, bw_method=0.25)

    # Also create a movie with ffmpeg
    print("Generating Snapshot Movie")
    bashcmd = f"ffmpeg -v quiet -stats -y -r 30 -f image2 -pattern_type glob -i '{output_path}/snapshots/*.png' -c:v h264 -pix_fmt yuv420p -strict -2 {output_path}/snapshot_movie.mp4"
    os.system(bashcmd)