# Dales input case generator
This repository contains the `moda` Python package, which generates complete DALES input cases (except for the DALES binary) using a composable Python API.

Cases are defined in Python using the `dales_simulation` class and configuration modules (grid, atmosphere, surface/LSM, radiation, timing, etc.).
Start with the [documentation guide](docs/index.rst). Build the HTML site with
`sphinx-build -E -b html docs docs/_build/html`.

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

   import moda

   with open("machine_conf.yaml") as f:
	   machine_conf = yaml.safe_load(f)

   sim = moda.dales_simulation("basic_case", machine_conf)
   sim += moda.DefaultNamelistModule()
   # ... add grid, atmosphere, surface/LSM, radiation, time modules ...
   sim.sim_preprocessing_pipeline()
   ```

This will create an output directory (based on the case name and machine configuration) containing a DALES namelist, `job.001`, and all required input files.

### Public API

Start with `import moda`. Core building blocks are available directly:
`moda.dales_simulation`, `moda.GridDales`, `moda.AtmosphereModule`,
`moda.OpenBoundaryModule`, `moda.DefaultNamelistModule`, `moda.TimeModule`,
`moda.RunModule`, `moda.SurfaceModule`, and `moda.RadiationModule`.

Specialist modules and helpers live in short domain namespaces:

| Namespace | Examples |
| --- | --- |
| `moda.Atmosphere` | `AtmosphericProfile`, `lin`, `LS2DAtmosphereModule` |
| `moda.Configuration` | `StatsModule`, `FielddumpModule`, `CheckSimulationModule` |
| `moda.Surface` | `ConstantFluxesModule`, `LSM` |
| `moda.Surface.LSM` | `LandUseModification`, `LandUseModifications`, `FromLCZ`, `LandUseSet`, `UniformSoilMoisture` |
| `moda.Geometry` | `AllMask`, `RectangleRealMask`, `ExplicitArrayMask` |
| `moda.LBC` | `NestingTopology`, `Nest_in_Dales`, `Nest_in_KNMI` |
| `moda.Radiation` | `RRTMGRadiationModule`, `BackradPressureProfile` |
| `moda.Emission` | `EmissionModule`, `EmissionTracer`, `EmissionPointSource` |
| `moda.IBM` | `IBMModule`, `FromAHN`, `IBMModification` |
| `moda.vars` | `ua`, `thetal`, `qt`, `tke` |

For example, configure surface modules without importing implementation files:

```python
import moda
from moda.Surface import LSM

lsm = LSM.LSMModule(
   land_use_set=LSM.LandUseSet.IFS,
   include_slurb=True,
   deep_soil_temperature=290.0,
   building_indoor_temperature=295.0,
)
lsm.add(
   LSM.LandUseModification(
      geometry=moda.Geometry.AllMask(), type="grs"
   ),
   priority=0,
)
lsm.add(LSM.FromLCZ(), priority=100)
lsm.add(LSM.SlurbModification(varname="h_bld", value=12.0), priority=200)
```

`SlurbModification` and `SlurbModifications` belong in `LSMModule` sources,
using `lsm.add(source, priority=...)`. They are also
importable from `moda.Surface.LSM.lsm_sources`. Higher priorities apply later:
a modification above `FromLCZ` (or `FromNetCDF`) overrides its morphology in
the selected geometry; a lower-priority modification supplies values that the
dataset may overwrite. Equal priorities preserve insertion order.

SLURB is integrated into `LSMModule`; there is no separate `SlurbModule`.
Set `include_slurb=True` to enable SLURB dataset morphology and
`inslurb.<exp_id>.nc` output. The default is `False`, which leaves `NAMSLURB`
untouched and writes no SLURB file. SLURB modification sources and explicit
SLURB namelist settings require `include_slurb=True`.
All former SLURB settings are direct LSM constructor arguments, including
`urban_fraction`, `urban_roughness_length`, building and street-canyon settings,
`shf_external`, and `qsws_external`. When enabled, omitted
`deep_soil_temperature` and `building_indoor_temperature` use 283.15 K and
293.15 K, respectively. Regenerate existing YAML by moving SLURB settings onto
`LSMModule`, enabling `include_slurb`, and keeping modifications in its
prioritized sources.

Top10 overlays replace fractions only where the raster contains a class mapped
by the selected land-use catalogue. Missing, unknown and out-of-extent cells
retain fractions from earlier sources, including outside the Netherlands.
Top10-specific LCZ remapping is restricted to that overlay. The optional
`fill_north_sea` setting still explicitly fills its sea mask when enabled.
SLURB dataset fields are assigned directly with
`slbCreatorClass.apply_field(name, values)`, using `(y, x)` or
`(layer, y, x)` arrays; geometric modifications remain prioritized sources.

IBM uses the same explicit source priorities as LSM:

```python
from moda import IBM

ibm = IBM.IBMModule(thlibm=293.15, thlroof=293.15)
ibm.add(IBM.FromGlobalDEM("terrain.tif"), priority=0)
ibm.add(IBM.IBMPostProcessing(IBM.FromAHN()), priority=100)
ibm.add(IBM.IBMModification(height=2.0, mode="add"), priority=200)
```

Higher priorities apply later; equal priorities preserve insertion order.
Terrain sources overlay positive cells, retaining earlier heights in holes.
Terrain readers return heights relative to their minimum elevation, without
mode subtraction or edge buffering. Wrap a source in
`IBMPostProcessing(source, zeroes_buffer=5, subtract_dem_mode=True)` to apply
the previous filtering behavior. The same wrapper accepts a list of sources:

```python
terrain = IBM.IBMPostProcessing(
   [IBM.FromGlobalDEM("terrain.tif"), IBM.FromAHN()],
   zeroes_buffer=5,
   subtract_dem_mode=True,
)
ibm.add(terrain, priority=100)
```

Wrapped sources apply in insertion order to an isolated height field, which
is normalized, optionally mode-subtracted, and buffered once for the whole
group. Positive results overlay earlier layers; zeros preserve their heights.
Wrappers can be nested and can also contain geometric sources. Move existing
`zeroes_buffer` and `subtract_dem_mode` (or AHN's `subtract_ahn_mode`) options
from terrain sources to this wrapper when updating Python or YAML cases.
`IBMModification` and `IBMModifications` are sources too, supporting `replace`,
`add`, and `subtract`. Pass a list to `add` to give multiple sources one priority.
Use `add(source, priority=...)` instead of the former `from_ahn`,
`from_global_dem`, and `ibm_modifications` constructor fields or module addition
operators. Existing IBM YAML must likewise be regenerated with prioritized sources.

Both IBM and LSM sources implement the structural `Source[Context]` protocol
from `moda.modular.sources`: `apply_source(context) -> None`. Modules validate
and sort entries, then call that method without dispatching on source types.
IBM passes its `IBMCreatorClass`; LSM passes an `LSMSourceContext` containing
the module and writer. Custom source classes can implement the same method;
register serializable dataclasses with `register_module` for YAML round-trips.

Time-dependent forcings are collected automatically by `moda.dales_simulation`.
Each module supplies its own time points; preprocessing interpolates the series
onto their combined axis, including time zero and the end of the run, sets
`ltimedep` and `ntimedep`, and writes `forcings.<exp_id>.nc`. No separate
time-dependent module is needed. The common axis must contain at least three
points. To override it, use
`moda.dales_simulation("case", machine_conf, timesteps=[0, 30, 60])`.
Explicit timesteps are retained when saving and loading simulation YAML.

Modules are constructed as detached configuration objects, without a `sim=` argument.
Configure them first, then attach and register them with `sim += module`.
The read-only `module.sim` property returns the bound simulation and raises
`RuntimeError` while detached. Configuration methods that do not access the
simulation remain usable before attachment. A module cannot be moved to another
simulation; adding the same instance again does not repeat its attachment hook.


## Changelog:
 -Allowed RTE-RRTMGP runs to be done. Added machine_conf.yaml to prevent having to edit scripts to customise.
 -Added some SLURB model input files (not yet useful)