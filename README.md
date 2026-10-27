# Dales input case generator
This repository contains the `MoDa` Python package, which generates complete DALES input cases (except for the DALES binary) using a composable Python API.

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

   import MoDa

   with open("machine_conf.yaml") as f:
	   machine_conf = yaml.safe_load(f)

   sim = MoDa.dales_simulation("basic_case", machine_conf)
   sim += MoDa.DefaultNamelistModule()
   # ... add grid, atmosphere, surface/LSM, radiation, time modules ...
   sim.sim_preprocessing_pipeline()
   ```

This will create an output directory (based on the case name and machine configuration) containing a DALES namelist, `job.001`, and all required input files.

### Public API

Start with `import MoDa`. Core building blocks are available directly:
`MoDa.dales_simulation`, `MoDa.GridDales`, `MoDa.AtmosphereModule`,
`MoDa.OpenBoundaryModule`, `MoDa.DefaultNamelistModule`, `MoDa.TimeModule`,
`MoDa.RunModule`, `MoDa.SurfaceModule`, and `MoDa.RadiationModule`.

Specialist modules and helpers live in short domain namespaces:

| Namespace | Examples |
| --- | --- |
| `MoDa.Atmosphere` | `AtmosphericProfile`, `lin`, `LS2DAtmosphereModule` |
| `MoDa.Configuration` | `StatsModule`, `FielddumpModule`, `CheckSimulationModule` |
| `MoDa.Surface` | `ConstantFluxesModule`, `LSMModule`, `SlurbModule` |
| `MoDa.Surface.LSM` | `LandUseModification`, `LandUseModifications`, `FromLCZ`, `LandUseSet`, `UniformSoilMoisture` |
| `MoDa.Geometry` | `AllGeometry`, `RectangleRealGeometry`, `MaskGeometry` |
| `MoDa.LBC` | `NestingTopology`, `Nest_in_Dales`, `Nest_in_KNMI` |
| `MoDa.Radiation` | `RRTMGRadiationModule`, `BackradPressureProfile` |
| `MoDa.Emission` | `EmissionModule`, `EmissionTracer`, `EmissionPointSource` |
| `MoDa.IBM` | `IBMModule`, `FromAHN`, `IBMModification` |
| `MoDa.vars` | `ua`, `thetal`, `qt`, `tke` |

For example, configure surface modules without importing implementation files:

```python
import MoDa
import MoDa.Surface

lsm = MoDa.Surface.LSMModule(land_use_set=MoDa.Surface.LSM.LandUseSet.IFS)
lsm.add(
   MoDa.Surface.LSM.LandUseModification(
      geometry=MoDa.Geometry.AllGeometry(), type="grs"
   ),
   priority=0,
)
slurb = MoDa.Surface.SlurbModule()
```

For longer case scripts, `from MoDa.Surface import LSM` is a convenient shorthand
for helpers such as `LSM.FromLCZ()` and `LSM.LandUseModification(...)`.
`SlurbModule` and `SLURBModule` refer to the same class. Existing explicit
top-level class imports and implementation import paths continue to work;
`MoDa.__all__` advertises the smaller core API and domain namespaces.

Modules are constructed as detached configuration objects, without a `sim=` argument.
Configure them first, then attach and register them with `sim += module`.
The read-only `module.sim` property returns the bound simulation and raises
`RuntimeError` while detached. Configuration methods that do not access the
simulation remain usable before attachment. A module cannot be moved to another
simulation; adding the same instance again does not repeat its attachment hook.


## Changelog:
 -Allowed RTE-RRTMGP runs to be done. Added machine_conf.yaml to prevent having to edit scripts to customise.
 -Added some SLURB model input files (not yet useful)