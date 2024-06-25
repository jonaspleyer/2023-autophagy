import os, json
import pandas as pd
import numpy as np
import multiprocessing as mp

from pathlib import Path
from cr_autophagy_pyo3 import *


def get_last_output_path(name = "out/autophagy") -> Path:
    """
    Obtains the output path of the last simulation that was run (assuming
    dates and times were added to the output folder).
    Optionally, one can specify the name of the output folder.

    Consider the following folder structure::

        out/autophagy
        ├── 2023-12-06-T06-02-30
        ├── 2023-12-11-T22-56-36
        ├── 2023-12-12-T00-34-23
        ├── 2023-12-12-T01-05-41
        ├── 2023-12-12-T01-12-52
        ├── 2023-12-12-T21-49-59
        └── 2023-12-13-T01-51-58

    The function above will then produce the following output::

        >>> get_last_output_path()
        Path("out/autophagy/2023-12-13-T01-51-58")

    """
    return Path(name) / sorted(os.listdir(Path(name)))[-1]


def get_simulation_settings(output_path: Path) -> SimulationSettings | None:
    """
    This function loads the `simulation_settings.json` file corresponding to the
    given ``output_path``. It is a wrapper for the ``json.load`` function.
    (See https://docs.python.org/3/library/json.html)
    """
    sim_settings_path = output_path / "simulation_settings.json"
    try:
        return SimulationSettings.load_from_file(sim_settings_path)
    except:
        return None


def _combine_batches(run_directory):
    # Opens all batches in a given directory and stores
    # them in one unified big list
    combined_batch = []
    for batch_file in os.listdir(run_directory):
        f = open(run_directory / batch_file)
        b = json.load(f)["data"]
        b = [bi["element"][0] for bi in b]
        combined_batch.extend(b)
    return combined_batch


def get_particles_at_iter(output_path: Path, iteration) -> pd.DataFrame:
    """
    Loads particles at a specified iteration.
    """
    dir = Path(output_path) / "cells/json"
    run_directory = None
    for x in os.listdir(dir):
        if int(x) == iteration:
            run_directory = dir / x
            break
    if run_directory is not None:
        df = pd.json_normalize(_combine_batches(run_directory))
        df["identifier"] = df["identifier"].apply(lambda x: tuple(x))
        df["cell.mechanics.pos"] = df["cell.mechanics.pos"].apply(lambda x: np.array(x, dtype=float))
        df["cell.mechanics.random_vector"] = df["cell.mechanics.random_vector"].apply(lambda x: np.array(x))
        return df
    else:
        raise ValueError(f"Could not find iteration {iteration} in saved results")


def get_all_iterations(output_path: Path) -> list:
    """
    Get all iterations that the simulation has produced for a given output path.
    """
    return sorted([int(x) for x in os.listdir(Path(output_path) / "cells/json")])


def __iter_to_cells(iteration_dir):
    iteration, dir = iteration_dir
    return (int(iteration), _combine_batches(dir / iteration))


def get_particles_at_all_iterations(output_path: Path, threads:int=1)->list:
    """
    Get all particles for every possible iteration step of the simulation run.
    This process can be parallelized if desired.
    """
    dir = Path(output_path) / "cells/json/"
    runs = [(x, dir) for x in os.listdir(dir)]
    pool = mp.Pool(threads)
    result = list(pool.map(__iter_to_cells, runs[:10]))
    return result
