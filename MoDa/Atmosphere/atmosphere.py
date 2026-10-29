import logging
from dataclasses import dataclass, field, replace

import numpy as np
import xarray as xr

from moda.Atmosphere.Profile import (
    AtmosphericProfile,
    InterpolatedProfile,
    TimedAtmosphereProfile,
    profiles_to_forcings,
)
from moda.IO_helpers import AtmosphereProfileWriter
from moda.modular.dales_simulation import dales_simulation
from moda.modular.forcing import (
    ForcingSet,
    collect_forcings,
    to_dataset,
    var_attrs,
)
from moda.modular.simulation_module import simulation_module
from moda.modular.time_dependent_scalars import (
    TIME_DIM,
    Z_DIM,
    resample_time,
    resolve_time_axis,
)
from moda.module_registry import register_module
from moda.vars import (
    ProfileVariable,
    get_var_by_name,
)

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


@register_module
@dataclass
class AtmosphereModule(simulation_module):
    """Main atmosphere module to add to the dales_simulation.
    Add physical profiles with an explicit destination:
    ```python
    atmo = AtmosphereModule()
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=3, ddz=0)))
    atmo.add(AtmosphericProfile(variable=ua, shape=lin(surf_val=5, ddz=0)),
             nudging=True)
    ```

    User-configured profiles take precedence over profiles provided by other
    modules (e.g. :class:`LS2DAtmosphereModule`); missing variables are filled
    from those providers. Writes ``init.<exp_id>.nc``, including nudging
    targets interpolated to a common time axis.

    Args:
        sim: Parent simulation instance
        profiles: Regular profiles at specific simulation times.
        nudging_profiles: Physical profiles used as nudging targets, independently
            of the initial state. No state-to-nudging fallback is applied.
    """

    profiles: list[TimedAtmosphereProfile] = field(default_factory=list)
    nudging_profiles: list[TimedAtmosphereProfile] = field(default_factory=list)
    profile_writer: AtmosphereProfileWriter = field(
        default_factory=AtmosphereProfileWriter,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )

    forcings: ForcingSet = field(
        default_factory=ForcingSet,
        init=False,
        repr=False,
        metadata={"serialize": False},
    )
    """Merged forcings of this module and all other providers (set in prepare_calculation)."""

    module_name: str = field(
        default="AtmosphereModule", init=False, metadata={"serialize": False}
    )

    def _initialize_from_sim(self, sim: "dales_simulation") -> None:
        sim.has_atmosphere_module = True

    def add(
        self,
        other: AtmosphericProfile
        | InterpolatedProfile
        | TimedAtmosphereProfile
        | list[AtmosphericProfile | InterpolatedProfile | TimedAtmosphereProfile],
        nudging: bool = False,
    ) -> None:
        """Add profiles, optionally to nudging instead of the regular destination."""
        destination = self.nudging_profiles if nudging else self.profiles
        if isinstance(other, (AtmosphericProfile, InterpolatedProfile)):
            destination.extend(other.timed_forcing_series())
        elif isinstance(other, TimedAtmosphereProfile):
            destination.append(other)
        elif isinstance(other, (list)):
            for item in other:
                if isinstance(
                    item,
                    (AtmosphericProfile, InterpolatedProfile, TimedAtmosphereProfile),
                ):
                    self.add(item, nudging=nudging)
                else:
                    raise TypeError(
                        f"Cannot add object of type {type(item)} to AtmosphereModule"
                    )
        else:
            raise TypeError(
                f"Cannot add object of type {type(other)} to AtmosphereModule"
            )

    def provide_forcings(self) -> ForcingSet:
        """Evaluate the user-configured profiles on the grid."""

        nudging_profiles = []
        for entry in self.nudging_profiles:
            entries = (
                [entry]
                if isinstance(entry, TimedAtmosphereProfile)
                else entry.timed_forcing_series()
            )
            for timed in entries:
                variable = timed.profile.variable
                if not isinstance(variable, ProfileVariable):
                    raise TypeError("Nudging requires an atmospheric state profile")
                nudging_profiles.append(
                    replace(
                        timed,
                        profile=replace(timed.profile, variable=variable.nudger),
                    )
                )
        return profiles_to_forcings(
            np.asarray(self.grid.zt, dtype=float), self.profiles + nudging_profiles
        )

    def prepare_calculation(self) -> None:
        """Merge own profiles with those of other providers (e.g. LS2D)."""

        if any(module is self for module in self.sim.modules):
            self.forcings = self.sim.forcings()
        else:
            # An AtmosphereModule outside the simulation (e.g. an open-boundary
            # source) combines its own profiles with the other providers.
            self.forcings = collect_forcings(
                [m for m in self.sim.modules if not isinstance(m, AtmosphereModule)]
                + [self]
            )
        logger.info("AtmosphereModule: Profiles prepared")

    def initial_profiles(self) -> xr.Dataset:
        """Initial profiles written to ``init.<exp_id>.nc`` (on ``zt``)."""
        vars_by_name = get_var_by_name()
        z = np.asarray(self.grid.zt, dtype=float)
        return to_dataset(
            {
                name: values
                for name, values in self.forcings.initial.items()
                if vars_by_name[name].is_initial_profile
            }
        ).assign_coords({Z_DIM: z})

    def init_dataset(self) -> xr.Dataset:
        """Contents of ``init.<exp_id>.nc``.

        Initial profiles plus the ``init_time_height`` variables (nudging
        targets and timescales) interpolated onto one common time axis.
        """
        vars_by_name = get_var_by_name()
        time_height = {
            name: series
            for name, series in self.forcings.series.items()
            if vars_by_name[name].init_time_height
        }
        for name, values in self.forcings.initial.items():
            if vars_by_name[name].init_time_height and name not in time_height:
                time_height[name] = values.expand_dims({TIME_DIM: [0.0]})

        dataset = self.initial_profiles()
        if time_height:
            times = resolve_time_axis(time_height.values())
            dataset = dataset.merge(
                to_dataset(
                    {
                        vars_by_name[name].init_name: resample_time(
                            series, times
                        ).assign_attrs(var_attrs(vars_by_name[name]))
                        for name, series in time_height.items()
                    }
                ),
                join="exact",
            )
        return dataset

    def write_files(self) -> None:
        """Write ``init.<exp_id>.nc`` (incl. nudging) and generate plots."""

        output_input_path = self.output_path / "input"
        self.profile_writer.write_init(
            self.init_dataset(), output_input_path, self.exp_id
        )
        self.profile_writer.plot_profiles(
            self.initial_profiles(), output_input_path, self.exp_id
        )
