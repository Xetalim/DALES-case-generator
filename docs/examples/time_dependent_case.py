"""Generate a case with a time-dependent surface temperature."""

from __future__ import annotations

import argparse
from pathlib import Path

from moda import (
    DefaultNamelistModule,
    GridDales,
    TimeDependentScalar,
    TimeModule,
    dales_simulation,
    vars,
)
from moda.Atmosphere import AtmosphereModule, AtmosphericProfile, lin
from moda.Surface import ConstantSurfaceTemperatureModule


def build_time_dependent_case(output_root: Path) -> dales_simulation:
    """Build and preprocess a case with a three-point surface-temperature series."""
    machine_conf = {
        "job_conf": {
            "numcores": 1,
            "dales_exec": "/path/to/dales",
            "debugger": "lldb",
        },
        "case_conf": {"BASE_OUTPUT_PATH": str(output_root.resolve())},
    }
    sim = dales_simulation("time_dependent_case", machine_conf)
    sim.add(DefaultNamelistModule())
    sim.add(
        GridDales(
            itot=16,
            jtot=16,
            kmax=20,
            xsize=160.0,
            ysize=160.0,
            kmax_soil=4,
            xlat=52.25,
            xlon=5.45,
            x0=0.0,
            y0=0.0,
            alpha=1.0,
            dz0=5.0,
        )
    )

    atmosphere = AtmosphereModule()
    atmosphere.add(
        AtmosphericProfile(variable=vars.ua, shape=lin(surf_val=3.0, ddz=1e-3))
    )
    atmosphere.add(
        AtmosphericProfile(variable=vars.va, shape=lin(surf_val=0.0, ddz=0.0))
    )
    atmosphere.add(
        AtmosphericProfile(variable=vars.thetal, shape=lin(surf_val=293.15, ddz=1e-2))
    )
    atmosphere.add(
        AtmosphericProfile(variable=vars.qt, shape=lin(surf_val=0.0, ddz=0.0))
    )
    atmosphere.add(
        AtmosphericProfile(variable=vars.wa, shape=lin(surf_val=0.0, ddz=0.0))
    )
    sim.add(atmosphere)

    sim.add(
        ConstantSurfaceTemperatureModule(
            thls=TimeDependentScalar(
                times=[0.0, 30.0, 60.0],
                values=[293.15, 294.15, 295.15],
            ),
            z0mav=0.0001,
            z0hav=0.0001,
            ps=100000.0,
            albedoav=0.22,
        )
    )
    sim.add(TimeModule(xtime=0.0, xday=1, xyear=2025, runtime=60.0))
    sim.sim_preprocessing_pipeline()
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("generated_cases"),
        help="Directory in which the time_dependent_case directory will be created.",
    )
    args = parser.parse_args()
    sim = build_time_dependent_case(args.output_root)
    print(f"Generated case: {sim.output_path}")
    print("Set job_conf.dales_exec to your DALES executable before running job.001.")


if __name__ == "__main__":
    main()
