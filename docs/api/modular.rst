Modular simulation API
======================

Simulation drivers and time-dependent scalar helpers exported by
``moda.modular``. The :doc:`forcing-time concept
<../concepts/forcing_time_axis>` describes how provider series are aligned.

``TimeDependentScalar(times, values)`` takes matching, non-empty sequences of
seconds-from-start and scalar values. Duplicate times are rejected. When any
time-dependent forcing is present, all providers are interpolated onto one
sorted axis that includes zero and the run duration; the final axis must have
at least three points. ``dales_simulation(..., timesteps=...)`` can provide an
explicit replacement axis.

The simulation driver writes provider outputs during
``sim_preprocessing_pipeline()``; it does not execute DALES. The
:doc:`tutorial <../quickstart>` shows the generated files.

.. automodule:: moda.modular
   :members: TimeDependentScalar, dales_simulation, namelist, simulation_module
   :imported-members: