Time-dependent large-scale forcing
==================================

This example adds time-dependent large-scale pressure gradients and
surface temperature using the time-dependent utilities.

What this example demonstrates
------------------------------

- Using ``TimedependentModule`` to define a timeline for forcing
  updates.
- Adding ``TimedAtmosphereProfile`` entries for variables such as
  ``dpdx``.
- Driving surface temperature with a ``TimeDependentScalar``.

Every variable may use its own time points: all series are linearly
interpolated onto the ``timesteps`` of ``TimedependentModule`` (or, when
those are omitted, onto the union of all provided time points). Values
outside a series' range are held constant.

Core ideas
----------

The forcing part of the configuration looks like

.. code-block:: python

  from modular_dales.modular.time_dependent import TimedependentModule
  from modular_dales.modular.time_dependent_scalars import TimeDependentScalar
  from modular_dales.Atmosphere import (
      AtmosphereModule,
      AtmosphericProfile,
      InterpolatedProfile,
      TimedAtmosphereProfile,
  )

  sim += TimedependentModule(timesteps=[0.0, 30.0, 60.0])

  atmo = AtmosphereModule()
  # background profiles for ua, va, wa, thetal, qt, tke
  ...
  atmo += [
      TimedAtmosphereProfile(
          time=0.0,
          profile=AtmosphericProfile(
              variable=dpdx, shape="lin", params=dict(surf_val=1.0, ddz=1e-3)
          ),
      ),
      TimedAtmosphereProfile(
          time=30.0,
          profile=AtmosphericProfile(
              variable=dpdx, shape="lin", params=dict(surf_val=2.0, ddz=1e-3)
          ),
      ),
      TimedAtmosphereProfile(
          time=60.0,
          profile=AtmosphericProfile(
              variable=dpdx, shape="lin", params=dict(surf_val=3.0, ddz=1e-3)
          ),
      ),
  ]

  sim += ConstantSurfaceTemperatureModule(
      thls=TimeDependentScalar(
          times=[0.0, 30.0, 60.0],
          values=[293.15, 294.15, 295.15],
      ),
      z0mav=0.0001,
      z0hav=0.0001,
      ps=100000,
      albedoav=0.22,
  )

  sim += atmo

When to use this pattern
------------------------

- Idealised cases where the large-scale pressure gradient or surface
  forcing changes abruptly or in steps.
- Creating small, self-contained NetCDF forcing files to check that
  time-dependent inputs are interpreted correctly by DALES.
  TODO