# python bindings for the autophagy simulation
![Python Installation](https://github.com/jonaspleyer/2023-autophagy/actions/workflows/python_installation.yml/badge.svg)
![Stable](https://github.com/jonaspleyer/2023-autophagy/actions/workflows/test_stable.yml/badge.svg)
## Getting Started
Create a new virtual environment and activate it.
```bash
python -m venv .venv
source .venv/bin/activate
```

You will need to have the rust compiler installed.
After having initialized and activated the virtual environment, we
can install the package.
```shell
pip install .
```

Install the required python dependencies
```bash
pip install -r requirements.txt
```

## Example
For a short example with default settings look at `run_sim.py`

## Documentation
You can find the documentation under [jonaspleyer.github.io/2023-autophagy](https://jonaspleyer.github.io/2023-autophagy/index.html).
