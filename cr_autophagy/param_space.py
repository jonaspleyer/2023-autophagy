import multiprocessing as mp
import tqdm

from cr_autophagy_pyo3 import SimulationSettings


def create_settings(**kwargs) -> SimulationSettings:
    return SimulationSettings(
        **kwargs
    )


def sample_space(**kwargs):
    for (key, value) in kwargs:
        print(key, value)
