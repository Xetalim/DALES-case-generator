Immersed boundary method
========================

Terrain sources, geometric edits, and the IBM simulation module exported by
``moda.IBM``. The :doc:`IBM how-to <../how_to/ibm>` demonstrates a complete
source setup; source composition is explained in
:doc:`Sources and priorities <../concepts/sources_and_priorities>`.

``IBMModule`` requires ``thlibm`` and ``thlroof`` (potential temperatures in
K). ``apply_ibm`` defaults to ``True``. When ``lwallheat=True``, ``thlwall``
must also be set. IBM fixes ``ibas_prf=2`` and ``iadv_mom=2``. Preprocessing
applies sources in ascending priority and writes
``input/ibm.inp_<exp_id>.nc`` with a two-dimensional ``bc_height`` in metres.
Terrain sources preserve existing heights where their input is non-positive;
geometric modifications can explicitly replace cells with zero.

.. automodule:: moda.IBM
   :members: FromAHN, FromGlobalDEM, IBMModification, IBMModifications, IBMModule, IBMPostProcessing, IBMSourceSpec
   :imported-members: