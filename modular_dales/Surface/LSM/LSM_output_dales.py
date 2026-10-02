# import matplotlib.pyplot as plt
import logging
from typing import Literal

import netCDF4 as nc4
import numpy as np

from modular_dales.Geometry.geometry_modification import AllGeometry, ModifierClass
from modular_dales.Geometry.GridDales import GridDales
from modular_dales.logging_wrapper import logwrap
from modular_dales.Surface.LSM.LCZ import get_from_LCZ

# Custom Python scripts/tools/...
from modular_dales.Surface.LSM.SLuRB.slurb import (
    SLURBModification,
    SLURBVariableModification,
    slbCreatorClass,
)
from modular_dales.Surface.LSM.translation_tables.vegetation_properties import (
    ifs_vegetation,
)

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


class LSM_output_dales:
    """
    Data structure for the required input for the new LSM
    """

    @logwrap
    def __init__(
        self,
        grid: GridDales,
        lu_types,
        soil_levels=4,
        debug=False,
    ):

        # land use parameters
        self.parnames = [
            "cover",
            "c_veg",
            "z0m",
            "z0h",
            "lai",
            "ar",
            "br",
            "lambda_s",
            "lambda_us",
            "rs_min",
            "gD",
            "tskin",
            "lutype",
        ]
        dtype_float = np.float64
        dtype_int = np.int32
        dtype_str = object
        self.nlu = len(lu_types)
        self.lu_types = None
        self.soil_levels = soil_levels
        self.grid: GridDales = grid

        self.value_dic = {
            "t_soil": np.zeros((soil_levels, grid.jtot, grid.itot), dtype=dtype_float),
            "index_soil": np.zeros(
                (soil_levels, grid.jtot, grid.itot), dtype=dtype_int
            ),
            "theta_soil": np.zeros(
                (soil_levels, grid.jtot, grid.itot), dtype=dtype_float
            ),
        }

        # Soil temperature, moisture content, and index in van Genuchten lookup table.
        fields = ["index_soil", "t_soil", "theta_soil"]

        # LU types
        self.luname = np.empty(len(lu_types), dtype=dtype_str)
        self.lushort = np.empty(len(lu_types), dtype=dtype_str)
        self.lveg = np.empty(len(lu_types), dtype=dtype_str)
        self.laqu = np.empty(len(lu_types), dtype=dtype_str)
        self.ilu = np.zeros(len(lu_types), dtype=dtype_int)

        # Sub-grid fraction of LU type (-)
        zeros = np.zeros((grid.jtot, grid.itot), dtype=dtype_float)
        for lu in lu_types:
            for parname in self.parnames:
                varname = "_".join([parname, lu])
                self.value_dic[varname] = zeros.copy()
                fields.append(varname)
        # total land use cover
        self.c_tot = np.zeros((grid.jtot, grid.itot), dtype=dtype_float)
        fields.append("cover_tot")
        self.c_veg_tot = np.zeros((grid.jtot, grid.itot), dtype=dtype_float)
        fields.append("c_veg_tot")

        # List of fields which are written to the binary input files for DALES
        self.fields = sorted(fields)

        # Bonus, for offline LSM (not written to DALES input)
        for lu in lu_types:
            ## LU type (-)
            varname = "type_" + lu
            self.value_dic[varname] = zeros.copy()

        if debug:
            # Init all values at NaN
            for field in self.fields:
                data = self.value_dic[field]
                if data.dtype == dtype_int:
                    continue
                data[:] = np.nan

        self.setup_lu_types(lu_types)

    def from_lcz(
        self,
        urban_natural_lcz_to_10: bool = False,
        urban_natural_lcz_to_natural_lsm: bool = False,
    ):
        if self.grid.crs is None:
            raise ValueError(
                "Need a valid projection to get LSM data from a real world map!"
            )
        print(self.grid.crs)
        print("Hello world")

        # cog = get_from_LCZ.get_cog(self.grid)

        LCZ_ds = get_from_LCZ.do_everything(
            self.grid,
            urban_natural_lcz_to_10=urban_natural_lcz_to_10,
            urban_natural_lcz_to_natural_lsm=urban_natural_lcz_to_natural_lsm,
        )
        self.LCZ_ds = LCZ_ds
        self.value_dic["index_soil"][:, :] = LCZ_ds["index_soil"][:, :]

        for lu_type, lu_dic in self.lu_types.items():
            ifs_id = lu_dic["ifs_id"]
            self.lu_types[lu_type]["lu_frac"][LCZ_ds["ifs_land_cover"] == ifs_id] = 1

        self.init_lutypes_ifs()
        # any gridpoint with total_cover <1 is set to bare soil to make sure the total cover is 1
        self.recalculate_remaining_cover()

    def apply_from_netcdf(self, ds, slb_generator):
        """Fill lu_types fracs + SLuRB morphology from a ``load_from_netcdf`` dataset.

        Parameters
        ----------
        ds:
            xarray Dataset returned by
            ``modular_dales.Surface.LSM.LCZ.from_netcdf.load_from_netcdf``.
        slb_generator:
            ``slbCreatorClass`` instance (may be *None* when SLURBModule is absent).
        """

        self.value_dic["index_soil"][:, :] = ds["index_soil"].values

        frac_water = np.asarray(ds["frac_water"].values, dtype=float)
        frac_sea = np.asarray(ds["frac_sea"].values, dtype=float)
        frac_town = np.asarray(ds["frac_town"].values, dtype=float)
        frac_nature = np.asarray(ds["frac_nature"].values, dtype=float)
        ifs_cover = np.asarray(ds["ifs_land_cover"].values, dtype=float)

        # Clip to [0, 1] to guard against reprojection artefacts
        frac_water = np.clip(frac_water, 0.0, 1.0)
        frac_sea = np.clip(frac_sea, 0.0, 1.0)
        frac_town = np.clip(frac_town, 0.0, 1.0)
        frac_nature = np.clip(frac_nature, 0.0, 1.0)

        # Combined ocean / water fraction → lu types with ifs_id == 22 (laqu=True)
        frac_water_total = np.clip(frac_water + frac_sea, 0.0, 1.0)
        water_lu_keys = [
            lu_key
            for lu_key, lu_dic in self.lu_types.items()
            if lu_dic.get("ifs_id") == 22
        ]
        # Distribute equally among all water lu_types (usually just 'wat')
        n_water = max(len(water_lu_keys), 1)
        for lu_key in water_lu_keys:
            self.lu_types[lu_key]["lu_frac"][:, :] = frac_water_total / n_water

        # Urban / SLuRB fraction → lu types with ifs_id == 20
        slb_lu_keys = [
            lu_key
            for lu_key, lu_dic in self.lu_types.items()
            if lu_dic.get("ifs_id") == 20
        ]
        n_slb = max(len(slb_lu_keys), 1)
        for lu_key in slb_lu_keys:
            self.lu_types[lu_key]["lu_frac"][:, :] = frac_town / n_slb

        # Nature fraction → distributed proportionally by dominant ESA IFS type
        # For each grid cell, frac_nature is split among non-water/non-urban IFS types
        # according to the fractional pixel count from the ESA raster.
        nature_lu_keys = [
            lu_key
            for lu_key, lu_dic in self.lu_types.items()
            if lu_dic.get("ifs_id") not in (20, 22) and lu_dic.get("ifs_id") is not None
        ]
        if nature_lu_keys:
            # Build a count array for each nature IFS id across the grid.
            # The ESA raster has one IFS code per pixel; we use it as a soft
            # distribution weight (each pixel = 1 unit of area).

            jtot, itot = frac_nature.shape
            weights = {
                lu_key: np.zeros((jtot, itot), dtype=float) for lu_key in nature_lu_keys
            }

            for lu_key in nature_lu_keys:
                ifs_id = self.lu_types[lu_key]["ifs_id"]
                weights[lu_key] = (ifs_cover == ifs_id).astype(float)

            total_weight = sum(weights[k] for k in nature_lu_keys)
            # Where total weight is zero (e.g. ESA has no matching class) give equal share
            zero_mask = total_weight == 0
            n_nat = len(nature_lu_keys)
            for lu_key in nature_lu_keys:
                w = weights[lu_key].copy()
                w[zero_mask] = 1.0 / n_nat
                total_w = total_weight.copy()
                total_w[zero_mask] = 1.0
                self.lu_types[lu_key]["lu_frac"][:, :] = frac_nature * (w / total_w)

        self.init_lutypes_ifs()
        self.recalculate_remaining_cover()

        # SLuRB morphological fields
        if slb_generator is not None:
            slurb_field_map = {
                "hw_can": "hw_can",
                "f_bld": "f_bld",
                "h_bld": "h_bld",
                "z0_urb": "z0_urb",
            }
            for nc_field, slurb_field in slurb_field_map.items():
                if nc_field not in ds:
                    continue
                modification = {
                    "geometry": "all",
                    "params": None,
                    "vars": [{"varname": slurb_field, "value": 0, "dtype": "real"}],
                }
                slb_generator.parse_yaml_name(modification)
                getattr(slb_generator, slurb_field)[:, :] = ds[nc_field].values

    def apply_slurb_parameters_lcz(self, slb_generator: slbCreatorClass):
        LCZ_ds = self.LCZ_ds
        # slb_modifications:
        #   - geometry: 'all'
        #     params:
        #     vars:
        #       - varname: "albedo_win"
        #         value: 1
        #         dtype: real
        for LCZ_field, slurb_field in get_from_LCZ.LCZ_field_to_slurb.items():
            modification = SLURBModification(
                geometry=AllGeometry(),
                vars=[
                    SLURBVariableModification(
                        varname=slurb_field,
                        value=0,
                        dtype="real",
                    )
                ],
            )

            slb_generator.apply_modification(modification)
            values = LCZ_ds[LCZ_field].values
            if slurb_field == "f_bld":
                urban_cover_key = next(
                    (
                        f"cover_{lu_key}"
                        for lu_key, lu_dic in self.lu_types.items()
                        if lu_dic.get("ifs_id") == 20
                    ),
                    None,
                )
                if urban_cover_key is None:
                    logger.warning(
                        "Could not find urban land-use type (IFS id 20) to scale f_bld; using unscaled values"
                    )
                else:
                    urban_cover = np.clip(self.value_dic[urban_cover_key], 0.001, 1.0)
                    values = (
                        np.clip(np.nan_to_num(values, nan=0.01), 0.001, 1.0)
                        * urban_cover
                    )

            getattr(slb_generator, slurb_field)[:, :] = values

    def set_uniform_soil_temperature(self, temperature):
        for i in range(self.soil_levels):
            self.value_dic["t_soil"][i, :, :] = temperature[i]

    def set_uniform_soil_moisture(self, moisture):
        for i in range(self.soil_levels):
            self.value_dic["theta_soil"][i, :, :] = moisture[i]

    def standard_fill_geometry_modification(self):
        self.value_dic["index_soil"][:, :] = 2
        # Python -> Fortran indexing
        self.value_dic["index_soil"] += 1

        # represents average over index_soil=2
        self.set_uniform_soil_moisture(
            [
                0.36867549,
                0.25300502,
                0.14997292,
                0.16459982,
            ]
        )
        self.set_uniform_soil_temperature(
            [
                283.26038083,
                286.79894009,
                290.88998902,
                288.09079126,
            ]
        )

        # inits the different land use parameters
        self.init_lutypes_ifs()
        # we need to set a skin temperature everywhere
        self.set_skin_temperature(273.15, lu_type="all")
        # any gridpoint with total_cover <1 is set to bare soil to make sure the total cover is 1
        self.recalculate_remaining_cover()

    def setup_lu_types(self, lu_types):
        """
        Set up arrays for luname, lushort, lveg, laqu, ilu. Doesn't set any values yet.

        """

        luname = [value["lu_long"] for key, value in lu_types.items()]
        lushort = [value["lu_short"] for key, value in lu_types.items()]
        lveg = [value["lveg"] for key, value in lu_types.items()]
        laqu = [value["laqu"] for key, value in lu_types.items()]
        ilu = np.arange(len(lu_types)) + 1

        self.luname = luname
        self.lushort = lushort
        self.lveg = lveg
        self.laqu = laqu
        self.ilu = ilu

        # set LU cover for each grid cell
        for lu in lu_types:
            lu_types[lu]["lu_domid"] = (
                np.ones((self.grid.jtot, self.grid.itot)) * lu_types[lu]["ifs_id"]
            )
            lu_types[lu]["lu_frac"] = np.ones((self.grid.jtot, self.grid.itot)) * 0
            self.value_dic["c_" + lu] = lu_types[lu]["lu_frac"]
        self.lu_types = lu_types

    def trim_landuse(self):
        """
        Remove unused land use types from the LSM_input_dales object.
        """
        used_lu_types = []
        unused_lu_types = []
        for lu in self.lu_types:
            cover = self.value_dic["cover_" + lu]
            if np.sum(cover) > 0:
                used_lu_types.append(lu)
            else:
                unused_lu_types.append(lu)
        if len(used_lu_types) < len(self.lu_types):
            logger.info(
                "Trimming unused land use types: %s",
                set(self.lu_types.keys()) - set(used_lu_types),
            )

        # update lu_types
        self.lu_types = {lu: self.lu_types[lu] for lu in used_lu_types}
        self.nlu = len(self.lu_types)

        luname = [value["lu_long"] for key, value in self.lu_types.items()]
        lushort = [value["lu_short"] for key, value in self.lu_types.items()]
        lveg = [value["lveg"] for key, value in self.lu_types.items()]
        laqu = [value["laqu"] for key, value in self.lu_types.items()]
        ilu = np.arange(len(self.lu_types)) + 1

        self.luname = luname
        self.lushort = lushort
        self.lveg = lveg
        self.laqu = laqu
        self.ilu = ilu
        # update luname, lushort, lveg, laqu, ilu

        # remove unused land use parameters from value_dic and fields
        for lu in unused_lu_types:
            for parname in self.parnames:
                varname = "_".join([parname, lu])
                del self.value_dic[varname]
                self.fields.remove(varname)

    @logwrap
    def save_netcdf(
        self,
        output_path,
        exp_id,
    ):
        """
        Save to NetCDF for visualisation et al.
        """
        nc = nc4.Dataset(f"{output_path}/lsm.inp_{exp_id:03d}.nc", "w")

        nc.createDimension("x", self.grid.itot)
        nc.createDimension("y", self.grid.jtot)
        nc.createDimension("z", self.soil_levels)
        nc.createDimension("nlu", self.nlu)
        nc.createDimension("str1", 1)
        nc.createDimension("str3", 3)
        nc.createDimension("str32", 32)

        var_x = nc.createVariable("x", float, "x")
        var_y = nc.createVariable("y", float, "y")

        var_x[:] = self.grid.xt[:]
        var_y[:] = self.grid.yt[:]

        for field in self.fields:
            data = self.value_dic[field]
            dims = ["y", "x"] if data.ndim == 2 else ["z", "y", "x"]
            var = nc.createVariable(field, float, dims)
            var[:] = data[:]
        self.grid.set_cf_grid_mapping(nc, "crs", self.fields)

        def _to_char_matrix(values, width):
            # Use null-padding to avoid trailing-space artifacts when consumers decode
            # fixed-width strings (e.g., "sg\x00" instead of "sg ").
            matrix = np.zeros((self.nlu, width), dtype="S1")
            for idx, value in enumerate(values):
                text = str(value).strip()[:width]
                encoded = text.encode("ascii", errors="replace")
                matrix[idx, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
            return matrix

        var_lun = nc.createVariable(
            "luname", datatype="S1", dimensions=("nlu", "str32")
        )
        var_lun[:, :] = _to_char_matrix(self.luname, 32)

        var_lus = nc.createVariable(
            "lushort", datatype="S1", dimensions=("nlu", "str3")
        )
        var_lus[:, :] = _to_char_matrix(self.lushort, 3)

        var_lveg = nc.createVariable("lveg", datatype="S1", dimensions=("nlu", "str1"))
        var_lveg[:, :] = _to_char_matrix(self.lveg, 1)

        var_laqu = nc.createVariable("laqu", datatype="S1", dimensions=("nlu", "str1"))
        var_laqu[:, :] = _to_char_matrix(self.laqu, 1)

        ilu = np.array(self.ilu, dtype=int)
        var_ilu = nc.createVariable("ilu", int, ("nlu",))
        var_ilu[:] = ilu

        nc.close()

        if getattr(self, "LCZ_ds", None) is not None:
            lcz_out_path = f"{output_path}/lcz_ds_{exp_id:03d}.nc"
            self.grid.set_cf_grid_mapping(
                self.LCZ_ds, "crs", list(self.LCZ_ds.data_vars.keys())
            )
            self.LCZ_ds.to_netcdf(lcz_out_path)
            logger.info(f"Saved LCZ dataset to {lcz_out_path}")


    @logwrap
    def set_skin_temperature(
        self, temperature, lu_type: Literal["all"] | str = "all"
    ):
        shape = (self.grid.jtot, self.grid.itot)
        temp_arr = np.full(shape, temperature)
        if lu_type == "all":
            for lu in self.lu_types.keys():
                self.value_dic[f"tskin_{lu}"] = temp_arr
        else:
            self.value_dic[f"tskin_{lu_type}"] = temp_arr

    @logwrap
    def get_laqu_lu_names(self):
        return [
            lu_name
            for lu_name, lu_cfg in self.lu_types.items()
            if bool(lu_cfg.get("laqu", False))
        ]

    @logwrap
    def set_skin_temperature_laqu(self, temperature):
        shape = (self.grid.jtot, self.grid.itot)
        temp_arr = np.full(shape, temperature)
        for lu_name in self.get_laqu_lu_names():
            self.value_dic[f"tskin_{lu_name}"] = temp_arr

    @logwrap
    def set_skin_temperature_array(
        self, temperature_array, lu_type: Literal["all"] | str = "all"
    ):
        shape = (self.grid.jtot, self.grid.itot)
        if temperature_array.shape != shape:
            raise ValueError(
                f"Incompatible shape of temperature array {temperature_array.shape} and grid shape {shape}"
            )
        if lu_type == "all":
            for lu in self.lu_types.keys():
                self.value_dic[f"tskin_{lu}"][:] = temperature_array
        else:
            self.value_dic[f"tskin_{lu_type}"] = temperature_array

    @logwrap
    def set_soil_temperature_array(self, soil_temperature_array):
        if soil_temperature_array.shape != self.value_dic["t_soil"].shape:
            raise ValueError(
                f"Incompatible shape of soil temperature array {soil_temperature_array.shape} and grid shape {self.value_dic['t_soil'].shape}"
            )
        self.value_dic["t_soil"][:, :, :] = soil_temperature_array[:, :, :]

    @logwrap
    def set_soil_moisture_array(self, soil_moisture_array):
        if soil_moisture_array.shape != self.value_dic["theta_soil"].shape:
            raise ValueError(
                f"Incompatible shape of soil moisture array {soil_moisture_array.shape} and grid shape {self.value_dic['theta_soil'].shape}"
            )
        self.value_dic["theta_soil"][:, :, :] = soil_moisture_array[:, :, :]

    @logwrap
    def init_lutypes_ifs(self):
        """
        Given some pre-existing land covers, gets properties from the IFS lookup table... TODO UNDERSTAND


        """
        shape = (self.grid.jtot, self.grid.itot)
        for lu in self.lu_types.keys():
            for parname in self.parnames:
                if parname == "cover" or parname == "c_veg":
                    parfield = self.lu_types[lu]["lu_frac"].copy()
                else:
                    parfield = np.full(shape, 0.0)

                ifs_vegetation_index = self.lu_types[lu]["ifs_id"]

                if parname == "lutype":
                    # LG: Only apply mask to cover and c_veg (DALES crashes when zeros or nans are in the array)
                    parfield[:] = ifs_vegetation_index
                elif parname in ["cover", "c_veg", "tskin"]:
                    # don't get any values from the IFS table for these values
                    pass
                else:
                    parname_translation_dic = {"ar": "a_r", "br": "b_r"}
                    if parname in parname_translation_dic:
                        parname_ifs = parname_translation_dic[parname]
                    else:
                        parname_ifs = parname
                    # parfield[mask] = getattr(ifs_vegetation, parname_ifs) [iv]
                    parfield[:] = getattr(ifs_vegetation, parname_ifs)[
                        ifs_vegetation_index
                    ]
                    # LG: Only apply mask to cover and c_veg
                self.value_dic[f"{parname}_{lu}"] = parfield

        totcover = self.calc_totcover("cover")
        self.value_dic["cover_tot"] = totcover

        totcveg = self.calc_totcover("c_veg")
        self.value_dic["c_veg_tot"] = totcveg

    @logwrap
    def recalculate_remaining_cover(self):
        # TODO: more consistent way to check for LU type with bare soil
        try:
            bs_name = [
                k
                for k in self.lu_types.keys()
                if (
                    "barren" in self.lu_types[k]["lu_long"].lower()
                    or "bare soil" in self.lu_types[k]["lu_long"].lower()
                )
            ][0]
        except IndexError:
            bs_name = [
                k for k in self.lu_types.keys() if self.lu_types[k]["bare_soil"]
            ][0]
        totcover = self.calc_totcover("cover")
        self.value_dic["cover_tot"] = totcover

        totcveg = self.calc_totcover("c_veg")
        self.value_dic["c_veg_tot"] = totcveg

        cover = self.value_dic["cover_tot"]
        cover_bs0 = self.value_dic[f"cover_{bs_name}"]
        cover_bs = np.round(1 - cover + cover_bs0, 6)

        self.value_dic["cover_" + bs_name] = cover_bs

        # recalculate
        totcover = self.calc_totcover("cover")
        self.value_dic["cover_tot"] = totcover

        totcveg = self.calc_totcover("c_veg")
        self.value_dic["c_veg_tot"] = totcveg

    def calc_totcover(self, ctype):
        """
        Calculate sum over cover of individual LU types to check if it sums up to 1

        Parameters
        ----------
        lsm_input : LSM_input_DALES
            Class containing Dales input parameters for all LU types.
        lu_types : dict
            LU type properties.
        ctype : str
            LU cover type to be summed.

        Returns
        -------
        totcover : np.array
            Total LU cover.

        """
        covers = [ctype + "_" + s for s in self.lu_types.keys()]
        totcover = np.zeros([self.grid.jtot, self.grid.itot])
        for c in covers:
            totcover += self.value_dic[c]

        return totcover


class LsmModifier(ModifierClass):
    # class to edit land use types (lu_types) for the DALES LSM input. Set type using set_type and
    # give a mask with any of the geometry primitives.
    def __init__(self, lsm_input: LSM_output_dales, grid: GridDales):
        self.lsm_input = lsm_input
        super().__init__(grid)

    def set_type(self, mask, lu_type, frac=1, mode="replace"):
        frac = float(frac)
        if frac < 0 or frac > 1:
            raise ValueError(f"Land-use fraction must be in [0, 1], got {frac}")

        if lu_type not in self.lsm_input.lu_types.keys():
            raise KeyError(
                f"Incorrect lu_type given {lu_type}, {self.lsm_input.lu_types.keys()}"
            )

        mode_norm = str(mode).strip().lower()
        lu_frac = self.lsm_input.lu_types[lu_type]["lu_frac"]

        if mode_norm == "replace":
            lu_frac[mask] = frac
            if frac == 1:
                for other_lu_type in self.lsm_input.lu_types.keys():
                    if lu_type != other_lu_type:
                        self.lsm_input.lu_types[other_lu_type]["lu_frac"][mask] = 0
            return

        if mode_norm == "add":
            target = lu_frac[mask] + frac
            if np.any(target > 1.0):
                raise ValueError(
                    "Add mode would set land-use fraction above 1.0 for type "
                    f"'{lu_type}'"
                )
            lu_frac[mask] = target
            return

        raise ValueError(
            f"Unsupported land-use modification mode '{mode}'. "
            "Expected one of: replace, add"
        )

    def do_modification(self, geometry, modification):
        frac = modification.frac if modification.frac is not None else 1.0
        mode = getattr(modification, "mode", "replace")
        self.set_type(geometry, modification.type, frac=frac, mode=mode)
