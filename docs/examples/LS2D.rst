Simple LSM configuration
========================

This example shows how to couple a DALES simulation to the LS2D module by
Bart van Stratum [LS2D_ref]_.
Keep in mind that getting the ERA5 profiles can take at least 10 minutes, during which time the script
will be running without any output to the terminal.

Core setup
----------

The Python structure is

.. code-block:: python


    from modular_dales import dales_simulation
    from modular_dales.modular.time_dependent import TimedependentModule
    from modular_dales.Atmosphere import AtmosphereModule, LS2DAtmosphereModule, FromLS2D
    from modular_dales.Configuration.defaultnamelist import DefaultNamelistModule
    from modular_dales.Configuration.output_modules import CheckSimulationModule, EasyOutputModule
    from modular_dales.Configuration.run_and_time import TimeModule
    from modular_dales.Geometry.GridDales import GridDales
    from modular_dales.Radiation.radiation import RadiationModule
    from modular_dales.Surface.surface import ConstantSurfaceTemperatureModule
    from modular_dales.Surface.LSM import (
        LSMModule,
        UniformSkinTemperature,
        UniformSoilTemperature,
        UniformSoilMoisture,
    )

   sim = dales_simulation("ls2d_atmosphere", machine_conf("ls2d_atmosphere"))

    sim += DefaultNamelistModule()
    domain_info = GridDales(
        itot=16,
        jtot=16,
        kmax=176,
        xsize=160.0,
        ysize=160.0,
        kmax_soil=4,
        xlat=51.971,
        xlon=4.927,
        x0=0.0,
        y0=0.0,
        alpha=1.009,
        dz0=20.0,
    )
    sim += domain_info

    # Without explicit timesteps, the LS2D time points (and any other
    # provided series) are combined into one common time axis.
    sim += TimedependentModule()
    # Configure LS2D-driven atmosphere; central_lat / central_lon and
    # case_name will be taken from GridDales and dales_simulation.
    atmo_ls2d = LS2DAtmosphereModule(
        era5_path=sim.machine_conf.get("ls2d_conf", {}).get("era5_path", "/Users/andrevanginkel/Documents/20_Code/28_dales_input/28.01_Dales_LSM_generator/jupyter_tests/era5_data"),
        start_date=datetime(2016, 8, 15, hour=6),
        end_date=datetime(2016, 8, 16, hour=6),
        write_log=False,
        data_source=sim.machine_conf.get("ls2d_conf", {}).get("data_source", "CDS"),
        n_av=0,
        method="2nd",
    )
    sim += atmo_ls2d

    # Base AtmosphereModule; LS2DAtmosphereModule will inject LS2D-based
    # initial profiles here so that ``init.<exp_id>.nc`` is written via
    # the standard AtmosphereModule machinery.
    atmo = AtmosphereModule()
    sim += atmo

    # Enable radiation. LS2D provides a background radiation file.
    sim += RadiationModule(iradiation=4)

    # Add an LSM module that explicitly opts into LS2D-driven soil and
    # roughness using FromLS2D. We provide simple uniform initial
    # profiles that will be overridden by LS2D when available.
    lsm = LSMModule(
        ps=100000.0,
        z0mav=0.0001,
        z0hav=0.0001,
        albedoav=0.22,
        iinterp_t=1,
        iinterp_theta=1,
        dz_soil=[0.07, 0.21, 0.72, 2.17],
    )
    lsm += FromLS2D()
    lsm += UniformSkinTemperature(skin_temperature=285.0)
    lsm += UniformSoilTemperature(soil_temperature=[283.0, 283.0, 283.0, 283.0])
    lsm += UniformSoilMoisture(soil_moisture=[0.3, 0.3, 0.3, 0.3])
    sim += lsm

    sim += TimeModule(xtime=6, xday=15, xyear=2016, runtime=24 * 3600.0)

    sim += EasyOutputModule(output_interval=60)

    sim += CheckSimulationModule(check_interval=360,check_tendencies=False, stop_on_invalid=False)

    # Run with microphysics turned on (two-moment scheme)
    if sim.nml.get("nammicrophysics") is None:
        sim.nml["nammicrophysics"] = {}
    sim.nml["nammicrophysics"]["imicro"] = 2

    # Run the full preprocessing pipeline: configure, check, prepare,
    # and write all module files, including LS2D outputs.
    sim.sim_preprocessing_pipeline()

.. [LS2D_ref] van Stratum, B. J. H., van Heerwaarden, C. C., & Vilà-Guerau de Arellano, J. (2023). The benefits and challenges of downscaling a global reanalysis with doubly-periodic large-eddy simulations. Journal of Advances in Modeling Earth Systems, 15, e2023MS003750. https://doi.org/10.1029/2023MS003750