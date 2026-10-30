API Reference
=============

The reference documents public objects re-exported by each package's
``__init__.py``. It intentionally omits implementation modules and private
helpers. ``Surface.LSM`` and its source namespace have their own pages because
they are explicitly exported subpackages.

For workflow instructions, start with the
:doc:`tutorial <../quickstart>`, then use the task guides for
:doc:`LSM <../how_to/lsm>`, :doc:`radiation <../how_to/radiation>`,
:doc:`IBM <../how_to/ibm>`, or :doc:`nesting <../how_to/nesting>`. Reference
pages focus on public constructor contracts and generated output.

.. toctree::
   :maxdepth: 2
   :caption: Core

   core
   modular

.. toctree::
   :maxdepth: 2
   :caption: Domains

   atmosphere
   configuration
   emission
   geometry
   ibm
   lbc
   radiation
   surface
   io_helpers