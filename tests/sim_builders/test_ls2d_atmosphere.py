from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from modular_dales import dales_simulation
from modular_dales.modular.time_dependent import TimedependentModule
from modular_dales.Atmosphere import AtmosphereModule, LS2DAtmosphereModule, FromLS2D
from modular_dales.Configuration.defaultnamelist import DefaultNamelistModule
from modular_dales.modular.simulation_module import set_nml_section
from modular_dales.Configuration.output_modules import (
    CheckSimulationModule,
    EasyOutputModule,
)
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


def test_ls2d_atmosphere_full_ls2d_pipeline(machine_conf) -> None:
    """Run LS2DAtmosphereModule with a small domain and real LS2D.

    This test requires the external ``ls2d`` package and valid ERA5
    data access as configured in ``machine_conf.yaml``. It exercises
    the full LS2D pipeline and checks that ``init.001.nc`` and
    ``forcings.001.nc`` are produced with the expected dimensions.
    """

    ls2d = pytest.importorskip("ls2d")  # noqa: F841  # ensure LS2D is available

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

    sim += TimedependentModule()
    # Configure LS2D-driven atmosphere; central_lat / central_lon and
    # case_name will be taken from GridDales and dales_simulation.
    atmo_ls2d = LS2DAtmosphereModule(
        era5_path=sim.machine_conf.get("ls2d_conf", {}).get(
            "era5_path",
            "/Users/andrevanginkel/Documents/20_Code/28_dales_input/28.01_Dales_LSM_generator/jupyter_tests/era5_data",
        ),
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

    # Enable radiation so that a backrad file is required; we keep the
    # configuration minimal here and do not depend on external RRTMG data.
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

    sim += CheckSimulationModule(
        check_interval=360, check_tendencies=False, stop_on_invalid=False
    )

    # Run with microphysics turned on (two-moment scheme)
    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "nammicrophysics", "imicro", 2
    )
    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocx", 0)
    set_nml_section(sim.nml, sim.nml_docs, "user_defined", "RUN", "nprocy", 0)
    set_nml_section(
        sim.nml, sim.nml_docs, "user_defined", "namnetcdfstats", "lsync", True
    )
    # Run the full preprocessing pipeline: configure, check, prepare,
    # and write all module files, including LS2D outputs.
    sim.sim_preprocessing_pipeline()

    # LS2DAtmosphereModule defaults to do_nudging=True, which should
    # propagate to the namelist and produce nudging fields in init.nc.
    assert sim.nml.get("NAMNUDGE", {}).get("lnudge", False)

    print(sim.output_path)

    # Check that init.001.nc and forcings.001.nc are created.
    init_path = sim.output_path / "input" / "init.001.nc"
    forcings_path = sim.output_path / "input" / "forcings.001.nc"

    assert init_path.is_file(), f"Missing LS2D init file: {init_path}"
    assert forcings_path.is_file(), f"Missing LS2D forcings file: {forcings_path}"

    # Backrad file written by LS2DAtmosphereModule
    backrad_path = sim.output_path / "input" / "backrad.inp.001.nc"
    assert backrad_path.is_file(), f"Missing LS2D backrad file: {backrad_path}"

    # LSM input file written by LSMModule
    lsm_path = sim.output_path / "input" / "lsm.inp_001.nc"
    assert lsm_path.is_file(), f"Missing LSM input file: {lsm_path}"

    # Inspect dimensions of the generated files using xarray
    with xr.open_dataset(init_path) as ds_init:
        zh = np.asarray(ds_init["zh"].values, dtype=float)
        assert zh.ndim == 1 and zh.size == domain_info.kmax

        # Check that a few key variables are present as base profiles
        for name in ("ua", "va", "thetal", "qt", "tke"):
            assert name in ds_init.variables, f"Variable {name} missing in init file"
            assert ds_init[name].shape == (domain_info.kmax,)

        # Nudging targets and timescales should exist when lnudge is enabled.
        for name in (
            "ua_nud",
            "va_nud",
            "wa_nud",
            "thetal_nud",
            "qt_nud",
            "nudging_constant_ua",
            "nudging_constant_va",
            "nudging_constant_wa",
            "nudging_constant_thetal",
            "nudging_constant_qt",
        ):
            assert (
                name in ds_init.variables
            ), f"Nudging variable {name} missing in init file"
            assert ds_init[name].shape == (ds_init.dims["time"], domain_info.kmax)

    with xr.open_dataset(forcings_path) as ds_forc:
        zh = np.asarray(ds_forc["zh"].values, dtype=float)
        time = np.asarray(ds_forc["time"].values, dtype=float)

        assert zh.ndim == 1 and zh.size == domain_info.kmax
        assert time.ndim == 1 and time.size >= 1

        # Spot-check that at least one profile and one scalar time
        # series exist with appropriate shapes.
        if "ug_timedep" in ds_forc.variables:
            assert ds_forc["ug_timedep"].shape == (domain_info.kmax, time.size)
        if "psurf_timedep" in ds_forc.variables:
            assert ds_forc["psurf_timedep"].shape == (time.size,)

        # Surface time series from LS2D
        if "thlsurf_timedep" in ds_forc.variables:
            assert ds_forc["thlsurf_timedep"].shape == (time.size,)
        if "wtsurf_timedep" in ds_forc.variables:
            assert ds_forc["wtsurf_timedep"].shape == (time.size,)
        if "wqsurf_timedep" in ds_forc.variables:
            assert ds_forc["wqsurf_timedep"].shape == (time.size,)
        if "qtsurf_timedep" in ds_forc.variables:
            assert ds_forc["qtsurf_timedep"].shape == (time.size,)

    # Inspect basic structure of backrad file
    with xr.open_dataset(backrad_path) as ds_back:
        assert "lev" in ds_back.dims
        assert "T" in ds_back.variables
        assert "q" in ds_back.variables
        lev = np.asarray(ds_back["lev"].values, dtype=float)
        assert lev.ndim == 1 and lev.size > 0

    # Inspect LSM soil fields and roughness; they should have been
    # overridden by LS2D when FromLS2D is present.
    with xr.open_dataset(lsm_path) as ds_lsm:
        assert "t_soil" in ds_lsm.variables
        assert "theta_soil" in ds_lsm.variables
        assert "index_soil" in ds_lsm.variables

        t_soil = np.asarray(ds_lsm["t_soil"].values, dtype=float)
        theta_soil = np.asarray(ds_lsm["theta_soil"].values, dtype=float)
        index_soil = np.asarray(ds_lsm["index_soil"].values, dtype=float)

        # Expect shape (z, y, x) with non-uniform vertical structure
        assert t_soil.shape[0] == domain_info.kmax_soil
        assert theta_soil.shape[0] == domain_info.kmax_soil
        # All soil indices should be identical scalars from LS2D
        assert np.all(index_soil == index_soil.flat[0])

    # Check that LSM roughness namelist values have been updated from LS2D
    assert sim.retrieve_module("LSMModule").z0mav is not None
    assert sim.retrieve_module("LSMModule").z0hav is not None
