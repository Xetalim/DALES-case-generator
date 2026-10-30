Sources and priorities
======================

IBM and LSM modules accept source objects that implement
``apply_source(context)``. The owning module constructs the appropriate
context and applies its sources when preparing its generated inputs. Sources
are data operations, not simulation modules: attach the owner module to the
simulation, and add each source to that owner.

The priority is an integer ordering key. Lower priorities run first; higher
priorities run later and can replace values written earlier. Sources with the
same priority preserve insertion order. Use distinct priorities when the
result depends on overlay order, and keep each stage explicit in the case
configuration.

The details of overlay semantics differ by source family. IBM terrain readers
overlay positive heights and leave previous values in data gaps, while
geometric modifications can explicitly replace cells with zero. LSM sources
write selected fields; an overlay may leave fields from earlier sources
untouched. Consult the task guides for
:doc:`IBM <../how_to/ibm>` and :doc:`LSM <../how_to/lsm>` before composing
sources from different datasets.

The structural source contract and prioritized-source APIs are listed under
:doc:`IBM <../api/ibm>` and :doc:`LSM sources <../api/lsm_sources>`.