from cr_autophagy_pyo3 import SimulationSettings, run_simulation
import cr_autophagy as cra
import os

if __name__ == "__main__":
    simulation_settings = SimulationSettings()
    simulation_settings.n_threads = 4
    simulation_settings.t_max = 4 * cra.MINUTE
    # simulation_settings.save_interval = 0.01 * cra.MINUTE
    # simulation_settings.dt = 0.00025 * cra.MINUTE

    units = cra.MICROMETRE**2 / cra.SECOND**2
    # simulation_settings.potential_strength_cargo_cargo          = units * 3e0
    simulation_settings.potential_strength_cargo_atg11w19       = units * 4e-1
    # simulation_settings.potential_strength_atg11w19_atg11w19    = units * 5e-1

    # High avidity
    simulation_settings.potential_strength_atg11w19_atg11w19    = units * 2e-1

    # simulation_settings.domain_size = 2500 * cra.NANOMETRE
    # simulation_settings.domain_cargo_radius_max = 450 * cra.NANOMETRE
    # # simulation_settings.domain_atg11w19_radius_min = 550 * cra.NANOMETRE
    # simulation_settings.domain_n_voxels = 5

    # simulation_settings.diffusion_cargo = 5e-5  * cra.MICROMETRE**2 / cra.SECOND
    # simulation_settings.diffusion_atg11w19 = 8e-5 * cra.MICROMETRE**2 / cra.SECOND

    # simulation_settings.interaction_range_atg11w19_cargo *= 2.0

    # simulation_settings.n_cells_cargo = 220
    # simulation_settings.n_cells_atg11w19 = 180

    storager = run_simulation(simulation_settings)

    output_path = cra.get_last_output_path()

    print("Saving Snapshots")
    cra.save_all_snapshots(output_path, threads=14, overwrite=True, transparent_background=True)
    #simulation_settings.n_threads)
    # cra.save_snapshot(output_path, 100, overwrite=True)

    # print("Saving cluster analysis plots")
    # cra.save_all_cluster_information_plots(
    #     output_path,
    #     threads=0,
    #     show_bar=True,
    #     connection_distance=2.5
    # )

    # print("Saving kde plots")
    # cra.save_all_kernel_density(
    #     output_path,
    #     threads=0,
    #     bw_method=0.25
    # )

    # Also create a movie with ffmpeg
    print("Generating Snapshot Movie")
    bashcmd = f"ffmpeg\
        -v quiet\
        -stats\
        -y\
        -r 30\
        -f image2\
        -pattern_type glob\
        -i '{output_path}/snapshots/*.png'\
        -c:v h264\
        -pix_fmt yuv420p\
        -strict -2 {output_path}/snapshot_movie.mp4"
    os.system(bashcmd)

    bashcmd2 = f"firefox {output_path}/snapshot_movie.mp4"
    os.system(bashcmd2)
