Working with IBM, LSM, and nested domains
=========================================

DALES input is assembled from simulation modules. A module owns a part of the
case configuration; sources provide data or modifications to that module. Add
the grid and the modules to a ``dales_simulation``, then call
``sim_preprocessing_pipeline()`` to prepare the case files. For nested runs,
prepare and run each parent before preparing its child.

Immersed boundary method
------------------------

``IBMModule`` builds the two-dimensional ``bc_height`` field used by the
immersed boundary method. It can combine terrain data with explicit geometric
edits. The module also writes IBM namelist options; ``thlibm`` and ``thlroof``
are required by the current module interface.

For example, load a DEM and add a rectangular obstacle on top of it:

.. code-block:: python

   import moda
   from moda.Geometry import GridDales
   from moda.Geometry.masks import RectangleRealMask
   from moda.IBM import IBMModification, IBMModule, FromGlobalDEM

   grid = GridDales(
       itot=100, jtot=100, kmax=40,
       xsize=1000.0, ysize=1000.0, kmax_soil=4,
       xlat=52.0, xlon=5.0, x0=0.0, y0=0.0, dz0=10.0,
   )
   sim = moda.dales_simulation("urban_case", machine_conf)
   sim.add(moda.DefaultNamelistModule())
   sim.add(grid)

   ibm = IBMModule(thlibm=293.15, thlroof=293.15)
   ibm.add(FromGlobalDEM("terrain.tif"), priority=10)
   ibm.add(
       IBMModification(
           mask=RectangleRealMask(minx=400, maxx=500, miny=300, maxy=450),
           height=18.0,
       ),
       priority=20,
   )
   sim.add(ibm)

``FromGlobalDEM`` reads a raster (for example a GeoTIFF) and reprojects it to
the DALES grid. ``FromAHN`` is available for AHN terrain. Both are terrain
sources: positive heights overlay the current field, while zero or negative
cells leave earlier values unchanged. A terrain source therefore does not
clear terrain in its data gaps.

``IBMModification`` applies a mask and a height operation. Masks include
``AllMask``, real-coordinate and index-coordinate circles and rectangles, and
explicit array masks. Real-coordinate masks use metres; index masks use grid
indices. The rectangle bounds are inclusive. Its ``mode`` can be:

* ``"replace"``: set selected cells to ``height`` (including zero).
* ``"add"``: add ``height`` to selected cells.
* ``"subtract"``: subtract ``height`` and clamp the result at zero.

Add sources with ``IBMModule.add(source, priority=...)``. Sources are applied
from lowest to highest priority, so a higher-priority source runs later. Equal
priorities retain insertion order. This ordering applies to terrain sources
and geometric edits alike. Use ``IBMModifications`` to group edits that should
run in a defined sequence. ``IBMPostProcessing`` runs one or more sources on
an isolated height field, normalizes the result, and can zero a border before
overlaying it on the current terrain.

Land surface model
------------------

``LSMModule`` creates the DALES land-surface input from a selected
``LandUseSet`` and a list of sources. The land-use set is required: it defines
which tiles and class mappings are available, and sources can restrict which
sets they support. ``FromLCZ`` and ``FromNetCDF`` support the IFS and DEPAC
sets; ``FromTop10`` and ``FromBofek`` support all currently defined sets.

An LSM setup might combine LCZ land cover, a Top10NL overlay, and BOFEK soil
classes:

.. code-block:: python

   from moda.Surface.LSM import LSMModule, LandUseSet
   from moda.Surface.LSM.lsm_sources import FromBofek, FromLCZ, FromTop10

   lsm = LSMModule(
       land_use_set=LandUseSet.IFS,
       ps=100000.0,
       z0mav=0.1,
       z0hav=0.01,
       iinterp_t=1,
       iinterp_theta=1,
       dz_soil=[1.89, 0.72, 0.21, 0.07],
       albedoav=0.22,
   )
   lsm.add(FromLCZ(), priority=100)
   lsm.add(
       FromTop10(spatial_data_path="/data/spatial_data"),
       priority=200,
   )
   lsm.add(
       FromBofek(spatial_data_path="/data/spatial_data"),
       priority=300,
   )
   sim.add(lsm)

The LSM module starts with its standard land-use geometry, then applies the
configured sources in ascending priority order. A source can populate
fractions, soil state, temperatures, moisture, or targeted land-use
modifications; later sources can replace the fields they write. For example,
``FromLCZ`` supplies land-cover and soil-class data, while ``FromTop10`` can
overlay land-use fractions and ``FromBofek`` supplies soil classes. A source's
priority is therefore part of the data recipe, not just a sorting preference.

The source API is shared in shape but specific in context: an IBM source
implements ``apply_source(ibm_context)`` and an LSM source implements
``apply_source(lsm_context)``. The owning module creates that context and
applies the sources during preparation. Add sources through the owning
module's ``add`` method so their priorities are validated and serialized with
the case configuration.

``LSMModule`` also accepts temperature and soil-moisture sources such as
``UniformSkinTemperature``, ``UniformSoilTemperature``,
``UniformSoilMoisture``, and their spatially varying counterparts. Use
``LandUseModification`` for a mask-based tile modification. If SLURB is
enabled with ``include_slurb=True``, set both ``deep_soil_temperature`` and
``building_indoor_temperature``; SLURB-specific namelist values are only
valid when that output is enabled.

Nesting DALES inside DALES
--------------------------

Use ``NestingTopology`` to describe the ordered grid hierarchy. Add grids from
outermost to innermost and set ``my_idx`` to the index of the grid used by the
current simulation. The outer simulation uses index zero. A middle-level
simulation uses its index in the full hierarchy, and an innermost simulation
uses the final index in the hierarchy it needs.

The parent topology identifies the next child and adds the cross-sections
needed to provide its boundaries. A child topology identifies its parent;
when ``my_idx`` is greater than zero, the simulation must also contain a
``Nest_in_Dales`` open-boundary module. That module uses the parent run's
outputs to prepare the child boundary and initial fields.

Grid placement and vertical levels must agree across the hierarchy. Child
horizontal boundaries must align with parent open-boundary grid coordinates
and remain inside the parent domain. The child top must align with a parent
vertical interface and lie below the parent top. When the child uses a subset
of its parent's vertical levels, ``child_grid.inherit_vertical_grid(parent_grid)``
copies those levels; the child must have fewer vertical levels than its
parent.

The following is the topology wiring for a parent and one child. Add the usual
time, atmosphere, surface, radiation, and run-configuration modules to each
simulation as required by the case:

.. code-block:: python

   from moda import GridDales, dales_simulation
   from moda.LBC import NestingTopology

   parent_sim = dales_simulation("parent", machine_conf)
   parent_grid = GridDales(**parent_grid_options)
   child_grid = GridDales(**child_grid_options)
   child_grid.inherit_vertical_grid(parent_grid)

   parent_sim.add(parent_grid)
   parent_topology = NestingTopology()
   parent_topology.add(parent_grid)
   parent_topology.add(child_grid)
   parent_topology.my_idx = 0
   parent_sim.add(parent_topology)

Prepare and run the parent first so the child can use its files. The child
simulation adds its own grid and topology, then connects ``Nest_in_Dales`` to
the parent paths:

.. code-block:: python

   from moda.LBC import Nest_in_Dales, NestingTopology

   child_sim = dales_simulation("child", machine_conf)
   child_sim.add(child_grid)

   child_topology = NestingTopology()
   child_topology.add(parent_grid)
   child_topology.add(child_grid)
   child_topology.my_idx = 1
   child_sim.add(child_topology)

   child_sim.add(
       Nest_in_Dales(
           e12=0.01,
           time0="2023-07-01T12:00:00",
           start="2023-07-01T12:00:00",
           end="2023-07-01T13:00:00",
           inpath=child_sim.output_path / "input",
           inpath_coarse=parent_sim.output_path / "input",
           outpath_coarse=parent_sim.output_path / "run_001",
           outpath_coarse_old=parent_sim.output_path / "run_001",
       )
   )

The ``inpath_coarse`` value points to the parent's input directory for its
initial fields. ``outpath_coarse`` supplies parent output during the nested
period, and ``outpath_coarse_old`` supplies the preceding output at the
initial boundary time. Set these paths and the time interval to match the
actual parent runs. For more than two levels, include all ancestor grids in
the topology in outer-to-inner order and set ``my_idx`` for the active grid,
as in the triple-nesting case in ``tests/LBC/test_openbc_triple_nesting.py``.