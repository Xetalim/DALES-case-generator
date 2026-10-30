Lateral boundary conditions
===========================

Open-boundary configuration, nesting topology, and forecast-download helpers
exported by ``moda.LBC``. Read
:doc:`Prepare nested DALES domains <../how_to/nesting>` before constructing a
parent/child topology; that guide explains parent output requirements and run
order.

``NestingTopology.nestings`` stores ``GridDales`` objects outermost to
innermost, and ``my_idx`` must identify the active grid. A nested child needs
an ``OpenBoundaryModule`` implementation, typically ``Nest_in_Dales`` for
DALES-to-DALES nesting. The parent must be prepared and run first; its initial
and time-dependent output paths are inputs to the child boundary workflow.

.. automodule:: moda.LBC
   :members: KNMIHarmonieForecastDownloadModule, Nest_in_AtmosphereProfiles, Nest_in_Dales, Nest_in_Harmonie, Nest_in_KNMI, NestingIndices, NestingTopology, OpenBoundaryConfig, OpenBoundaryModule
   :imported-members: