Forcing time axes
=================

Time-dependent scalar and atmospheric inputs use time in seconds from the
start of the simulation. Providers may define different time points. During
preprocessing, the simulation forms one common sorted axis from the union of
provider times, then adds time zero and the configured runtime. Values are
linearly interpolated onto that axis; outside a provider's range, its endpoint
value is held constant.

When any time-dependent forcing is present, the resulting common axis must
contain at least three points. A one-hour simulation with two supplied times
may therefore need additional samples or an explicit axis through
``dales_simulation(..., timesteps=[...])``. Explicit timesteps replace the
provider-time union, but time zero and runtime are still included.

``TimeDependentScalar`` requires equal, non-empty ``times`` and ``values``
lists with no duplicate times. Atmospheric profile times and surface forcing
times use the same seconds-from-start convention. Calendar start time is set
separately by ``TimeModule``.

The :doc:`time-dependent surface example <../gallery/index>` generates a
forcing file with the tutorial's three-point axis. See the
:doc:`modular API <../api/modular>` for the exported scalar and simulation
driver.