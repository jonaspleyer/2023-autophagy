import cr_autophagy as cra
import sys
from pathlib import Path
import tqdm
import multiprocessing as mp
import itertools

def _plot_angle_image(opath: Path, iteration: int, angle: int, suffix: str):
    return cra.save_snapshot(
        opath,
        iteration,
        ascending_rotation_angle=1/200,
        view_angles=(angle, 0, 0),
        transparent_background=True,
        subfolder= "angle-snapshots-{}".format(suffix),
        suffix="{:03}".format(angle),
    )

def _plotting_helper(opath_angle):
    opath, angle = opath_angle
    initial = iterations[0]
    final = iterations[-1]
    _plot_angle_image(opath, initial, angle, "initial")
    _plot_angle_image(opath, final, angle, "final")

if __name__ == "__main__":
    opaths = sys.argv[1:]

    for opath in opaths:
        opath = Path(opath)
        iterations = cra.get_all_iterations(opath)
        angles = list(range(0, 364, 1))
        args = [(opath, angle) for angle in angles]
        pool = mp.Pool()
        list(tqdm.tqdm(pool.imap(
            _plotting_helper,
            args
        ), total=len(args)))
        cra.create_movie(
            opath,
            subfolder="angle-snapshots-final",
            name="angle-movie-final",
        )
        cra.create_movie(
            opath,
            subfolder="angle-snapshots-initial",
            name="angle-movie-initial",
        )

