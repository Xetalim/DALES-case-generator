Atmosphere
==========

Public atmospheric profiles, forcing helpers, and simulation modules exported
by ``moda.Atmosphere``. The :doc:`first-case tutorial <../quickstart>` shows
the required ``AtmosphereModule`` setup; the
:doc:`forcing-time concept <../concepts/forcing_time_axis>` explains timed
inputs.

``AtmosphereModule`` writes ``input/init.<exp_id>.nc`` from its configured
initial profiles and may write ``forcings.<exp_id>.nc`` when values vary in
time. Add ``AtmosphericProfile(variable, shape)`` objects with
``AtmosphereModule.add``. ``variable`` is a registered DALES profile variable;
its units follow that variable's definition. ``shape`` is a supported shape
function such as ``lin(surf_val, ddz)``. ``ddz`` uses the variable's units per
metre, and all profile heights are in metres on the grid's vertical levels.

User profiles take precedence over values supplied by other atmospheric
providers; missing variables may be filled from those providers. Nudging
profiles are a separate destination and do not fall back to initial-state
profiles. LS2D and HARMONIE providers require their respective external forcing
data and additional setup; see the exported provider descriptions below.

.. automodule:: moda.Atmosphere
   :members: AtmosphereModule, AtmosphericProfile, InterpolatedProfile, TimedAtmosphereProfile, HarmonieAtmosphereModule, LS2DAtmosphereModule, FromLS2D, exp, expsinw, lin, linmlsurf, SHAPE_FUNCTIONS
   :imported-members: