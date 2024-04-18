from cr_autophagy_pyo3 import SimulationSettings, run_simulation
import cr_autophagy as cra
import json

if __name__ == "__main__":
    simulation_settings = SimulationSettings()
    simulation_settings.n_threads = 2
    
    storager = run_simulation(simulation_settings)

    import os
    from pathlib import Path
    output_path = Path(cra.get_last_output_path())

    print("Saving Snapshots")
    cra.save_all_snapshots(output_path, threads=14)#simulation_settings.n_threads)

    # print("Saving cluster analysis plots")
    # cra.save_all_cluster_information_plots(
    #     output_path,
    #     threads=0,
    #     show_bar=True,
    #     connection_distance=2.5
    # )

    print("Saving kde plots")
    cra.save_all_kernel_density(
        output_path,
        threads=0,
        bw_method=0.25
    )

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
