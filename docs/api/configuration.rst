Configuration
=============

Run controls, physics settings, and output modules exported by
``moda.Configuration``. See the
:doc:`tutorial <../quickstart>` for a working ``TimeModule`` and
:doc:`radiation how-to <../how_to/radiation>` for the surface/radiation
interaction.

``RunModule.runtime`` is measured in seconds; ``iexpnr``, ``ladaptive``, and
``dtmax`` map to the corresponding ``RUN`` namelist values and have no
constructor default. ``TimeModule`` is exported at the package root and
documented on the :doc:`core API page <core>`.

.. automodule:: moda.Configuration
   :members: BulkMicrophysicsSettingsModule, BulkMicrophysicsStatisticsOutputModule, CapeModule, CheckSimulationModule, ColumnStatisticsOutputModule, CrossSectionOutputModule, EasyOutputModule, FielddumpModule, GeneralPhysicsModule, LSMCrossModule, LateralSpongeModule, NetCDFStatisticsSyncModule, RadfieldModule, RunModule, SamplingModule, SprayingModule, StatsModule, TimestatModule, TracerSettingsModule, VirtualMeasurementOutputModule
   :imported-members:
   :exclude-members: check_interval, check_tendencies, stop_on_invalid