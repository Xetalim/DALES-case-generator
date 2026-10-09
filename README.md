# Dales input case generator
This repository contains the `modular_dales` Python package, which generates complete DALES input cases (except for the DALES binary) using a composable Python API.

Cases are defined in Python using the `dales_simulation` class and configuration modules (grid, atmosphere, surface/LSM, radiation, timing, etc.).
For the documentation, see [this](docs/_build/markdown/index.md)

This tool requires Python >=3.13. Install the published package with:

```shell
python -m pip install dales-generator
```

For development, clone the repository and install it into your active environment:

```shell
poetry install --with test
```

Case scripts can be run from any directory. Bundled job and input templates are read from the installed package, while logs are written under `logs/` in the current working directory.

## Usage
1. Edit `machine_conf.yaml` to point to your DALES source and DALES binary. The `case_conf.BASE_OUTPUT_PATH` entry controls where generated cases are written. Tildes (~) do not work in this path (at least on my device).
2. In Python, load `machine_conf.yaml`, construct a `dales_simulation`, add modules, and run the preprocessing pipeline, for example:

   ```python
   import yaml

   from modular_dales.modular import dales_simulation
   from modular_dales.Configuration.defaultnamelist import DefaultNamelistModule

   with open("machine_conf.yaml") as f:
	   machine_conf = yaml.safe_load(f)

   sim = dales_simulation("basic_case", machine_conf)
   sim += DefaultNamelistModule()
   # ... add grid, atmosphere, surface/LSM, radiation, time modules ...
   sim.sim_preprocessing_pipeline()
   ```

This will create an output directory (based on the case name and machine configuration) containing a DALES namelist, `job.001`, and all required input files.


## Changelog:
 -Allowed RTE-RRTMGP runs to be done. Added machine_conf.yaml to prevent having to edit scripts to customise.
 -Added some SLURB model input files (not yet useful)