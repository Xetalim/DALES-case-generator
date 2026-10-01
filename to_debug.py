import xarray as xr
import time
import cartopy

print("imported cartopy")
time.sleep(1)
import earthkit.data as ekd
import pathlib

path = "/Users/andrevanginkel/Documents/40_Input_and_Runs/42_Dales_Cases/42.01_generated_cases/knmi_download_cache/HARM43_V1_P3_2026032618"
import glob

# time.sleep(2)
ds = ekd.from_source("file", f"{path}/*")
# ds = xr.open_mfdataset(f"{path}/*",engine="earthkit",drop_dims="step")
fl = ds.to_fieldlist()
new_fields = []

assignments = {}
used_names = set()

for field in fl:
    parameter = field.parameter.variable()
    level_type = field.vertical.level_type()

    tri = field.get("metadata.timeRangeIndicator")
    if tri is None:
        tri = 0

    key = (parameter, level_type, tri)

    if key in assignments:
        new_parameter = assignments[key]
    else:
        # First occurrence of this parameter/level/TRI combination
        if parameter not in used_names:
            new_parameter = parameter
        else:
            n = 2
            new_parameter = f"{parameter}_{n}_{tri}"

            while new_parameter in used_names:
                n += 1
                new_parameter = f"{parameter}_{n}_{tri}"

        assignments[key] = new_parameter
        used_names.add(new_parameter)

    if new_parameter == parameter:
        new_fields.append(field)
    else:
        new_fields.append(field.set({"parameter.variable": new_parameter}))

new_fl = ekd.FieldList.from_fields(new_fields)
# List available fields
# ds.to_fieldlist().ls()
dataset = new_fl.to_xarray(
    # drop_dims=["step"],
    allow_holes=True,
    # level_dim_mode="level_per_type",
    # time_dims=["valid_time"],
    # remapping={"parameter.variable": "{parameter.variable}_{vertical.level_type}"},
)
