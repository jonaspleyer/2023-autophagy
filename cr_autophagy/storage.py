import os, json
import pandas as pd
import numpy as np
import multiprocessing as mp

from pathlib import Path
from cr_autophagy_pyo3 import *
from types import SimpleNamespace


def get_last_output_path(name = "out/autophagy"):
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


def get_simulation_settings(output_path) -> SimpleNamespace:
    """
    This function loads the `simulation_settings.json` file corresponding to the
    given ``output_path``. It is a wrapper for the ``json.load`` function.
    (See https://docs.python.org/3/library/json.html)
    """
    f = open(output_path / "simulation_settings.json")
    return json.load(f, object_hook=lambda d: SimpleNamespace(**d))


def _combine_batches(run_directory):
    # Opens all batches in a given directory and stores
    # them in one unified big list
    combined_batch = []
    for batch_file in os.listdir(run_directory):
        f = open(run_directory / batch_file)
        b = json.load(f)["data"]
        combined_batch.extend(b)
    return combined_batch


def get_particles_at_iter(output_path: Path, iteration):
    """
    Loads particles at a specified iteration.
    """
    dir = Path(output_path) / "cell_storage/json"
    run_directory = None
    for x in os.listdir(dir):
        if int(x) == iteration:
            run_directory = dir / x
            break
    if run_directory != None:
        df = pd.json_normalize(_combine_batches(run_directory))
        df["identifier"] = df["identifier"].apply(lambda x: tuple(x))
        df["element.cell.mechanics.pos"] = df["element.cell.mechanics.pos"].apply(lambda x: np.array(x, dtype=float))
        df["element.cell.mechanics.vel"] = df["element.cell.mechanics.vel"].apply(lambda x: np.array(x, dtype=float))
        df["element.cell.mechanics.random_vector"] = df["element.cell.mechanics.random_vector"].apply(lambda x: np.array(x))
        return df
    else:
        raise ValueError(f"Could not find iteration {iteration} in saved results")


def get_all_iterations(output_path):
    return sorted([int(x) for x in os.listdir(Path(output_path) / "cell_storage/json")])


def __iter_to_cells(iteration_dir):
    iteration, dir = iteration_dir
    return (int(iteration), _combine_batches(dir / iteration))


def get_particles_at_all_iterations(output_path: Path, threads=1):
    dir = Path(output_path) / "cell_storage/json/"
    runs = [(x, dir) for x in os.listdir(dir)]
    pool = mp.Pool(threads)
    result = list(pool.map(__iter_to_cells, runs[:10]))
    return result