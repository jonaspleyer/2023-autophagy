import cr_autophagy as cra
import sys
from pathlib import Path

if __name__ == "__main__":
    opaths = sys.argv[1:]

    for opath in opaths:
        opath = Path(opath)
        cra.save_all_snapshots(
            opath,
            threads=10,
            ascending_rotation_angle=1/200,
            view_angles=(112, 0, 0),
            overwrite=True,
            transparent_background=True,
        )
        cra.create_movie(opath)

