Build your first case
=====================

This tutorial builds a small valid case, runs preprocessing, and inspects the
generated inputs. Preprocessing generates DALES files; it does not compile or
run the DALES model.

Install Python 3.13 or newer, then install the package:

.. code-block:: console

   python -m pip install dales-generator

To run the example from a source checkout, install that checkout with
``python -m pip install -e .`` and run the command below from the repository
root.

Machine configuration
---------------------

The simulation needs a machine-configuration mapping. The example creates one
with a local output directory and one core, so input generation is portable and
does not require DALES. The executable path is a placeholder until you are
ready to run the generated job. A YAML configuration for a real run typically
has this shape:

.. code-block:: yaml

   job_conf:
     dales_exec: /absolute/path/to/dales
     debugger: lldb
     numcores: 1
   case_conf:
     BASE_OUTPUT_PATH: ./generated_cases

``BASE_OUTPUT_PATH`` is where generated case directories are created.
``job_conf.numcores`` is used to select a valid MPI decomposition;
``job_conf.dales_exec`` is used by ``job.001``. Keep these paths specific to
the machine where you generate or run the case. Modules that read external
datasets may require additional paths.

Required modules
----------------

Before preprocessing, a simulation must contain all four of these:

* ``DefaultNamelistModule`` supplies required default run settings.
* ``GridDales`` defines the horizontal and vertical domain.
* ``AtmosphereModule`` provides the initial atmospheric profiles.
* A ``SurfaceModule`` subclass supplies surface boundary conditions; this
  tutorial uses ``ConstantSurfaceTemperatureModule``.

The tutorial also adds ``TimeModule`` to set the start date and run duration.
Radiation, LSM, IBM, emissions, and open boundaries are optional and depend on
the experiment.

Build and preprocess
--------------------

The complete script, including all imports, is
`examples/first_case.py <examples/first_case.py>`_. It uses a 16 by 16
horizontal grid, 20 vertical levels, constant atmospheric profiles, a
constant-temperature surface, and a 60-second run configuration. It requires
no local raster, forecast, credentials, or DALES installation to generate the
input files.

.. literalinclude:: examples/first_case.py
   :language: python

Run it from the repository root:

.. code-block:: console

   python docs/examples/first_case.py --output-root generated_cases

The script calls ``sim.sim_preprocessing_pipeline()`` and creates
``generated_cases/first_case``.

Inspect the generated case
--------------------------

The generated files are:

.. code-block:: text

   first_case/
   |-- job.001
   |-- app.log
   |-- input/
   |   |-- docs_namoptions.001.md
   |   |-- init.001.nc
   |   |-- namoptions.001
   |   `-- simulation_config.001.yaml
   `-- profiles/
       |-- profile_qt.png
       |-- profile_thetal.png
       |-- profile_tke.png
       |-- profile_ua.png
       |-- profile_va.png
       `-- profile_wa.png

Open ``input/namoptions.001`` to inspect the DALES namelist and
``input/docs_namoptions.001.md`` to see which modules supplied its values.
``input/init.001.nc`` contains the initial atmospheric profiles;
``input/simulation_config.001.yaml`` is the serialized case; the PNG files
plot the initial profiles. Review ``job.001`` for the executable and machine
settings before submitting it.

Run DALES
---------

To run the model, replace the example's ``/path/to/dales`` with your DALES
executable, ensure the executable and MPI runtime are available, then execute
``job.001`` from the generated case directory. You can generate and inspect
inputs without DALES.

Next, use the :doc:`How-To Guides <how_to/index>` to add data sources or
radiation, or read :doc:`Concepts <concepts/index>` to understand module
ownership, preprocessing, source priorities, and forcing time axes.