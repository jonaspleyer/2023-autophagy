from cr_autophagy_pyo3 import SimulationSettings, run_simulation
import cr_autophagy as cra


if __name__ == "__main__":
    simulation_settings = SimulationSettings()

    # HIGH AFFINITY CONFIGURATION
    #simulation_settings.potential_strength_cargo_r11 = 0.01
    #simulation_settings.potential_strength_cargo_r11_avidity = 0.00

    # HIGH AVIDITY CONFIGURATION
    simulation_settings.potential_strength_r11_r11 = 0.01
    simulation_settings.potential_strength_cargo_r11 = 0.01
    simulation_settings.potential_strength_cargo_r11_avidity = 0.00

    simulation_settings.n_times = 40_001
    simulation_settings.dt = 2.0
    simulation_settings.save_interval = 200

    simulation_settings.n_threads = 4

    simulation_settings.kb_temperature_r11 = 0.0025

    simulation_settings.random_seed = 3

    from pathlib import Path
    import os
    output_path = Path(run_simulation(simulation_settings))

    print("Saving Snapshots")
    cra.save_all_snapshots(output_path, threads=14)#simulation_settings.n_threads)

    print("Saving cluster analysis plots")
    cra.save_all_cluster_information_plots(
        output_path,
        threads=0,
        show_bar=True,
        connection_distance=2.5
    )

    print("Saving kde plots")
    cra.save_all_kernel_density(
        output_path,
        threads=0,
        bw_method=0.25
    )

    # Also create a movie with ffmpeg
    print("Generating Snapshot Movie")
    bashcmd = f"ffmpeg -v quiet -stats -y -r 30 -f image2 -pattern_type glob -i '{output_path}/snapshots/*.png' -c:v h264 -pix_fmt yuv420p -strict -2 {output_path}/snapshot_movie.mp4"
    os.system(bashcmd)
