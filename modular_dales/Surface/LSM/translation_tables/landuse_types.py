"""Named land-use catalogues with immutable metadata definitions."""

from dataclasses import replace
from enum import StrEnum

from ..lsm_data import LandUseDefinition, LandUseName

__all__ = [
    "LandUseSet",
    "get_land_use_definitions",
    "lu_types_basic",
    "lu_types_build",
    "lu_types_crop",
    "lu_types_depac",
    "lu_types_ifs",
]


class LandUseSet(StrEnum):
    """Available land-use classification schemes."""

    BASIC = "basic"
    BUILD = "build"
    CROP = "crop"
    DEPAC = "depac"
    IFS = "ifs"


lu_types_basic: dict[LandUseName, LandUseDefinition] = {
    "lv": LandUseDefinition(
        lu_long="low vegetation",
        lu_short="lv",
        lveg=True,
        lu_ids=(9, 10, 11, 13, 26),
    ),
    "hv": LandUseDefinition(
        lu_long="high vegetation",
        lu_short="hv",
        lveg=True,
        lu_ids=(1, 2, 3, 4, 5, 6, 7, 8, 27),
    ),
    "bs": LandUseDefinition(
        lu_long="bare soil",
        lu_short="bs",
        lu_ids=(12, 28),
        bare_soil=True,
    ),
    "aq": LandUseDefinition(
        lu_long="water surface",
        lu_short="aq",
        laqu=True,
        lu_ids=(14, 15, 16, 17, 18),
    ),
    "ap": LandUseDefinition(
        lu_long="asphalt",
        lu_short="ap",
        lu_ids=(20, 22, 23, 24, 25, 29, 30),
    ),
}

lu_types_build: dict[LandUseName, LandUseDefinition] = {
    **lu_types_basic,
    "ap": replace(lu_types_basic["ap"], lu_ids=(20, 22, 23, 24, 25)),
    "bu": LandUseDefinition(
        lu_long="buildings",
        lu_short="bu",
        lu_ids=(29, 30),
    ),
}

lu_types_crop: dict[LandUseName, LandUseDefinition] = {
    **lu_types_build,
    "ba": LandUseDefinition(
        lu_long="barley",
        lu_short="ba",
        lveg=True,
        lu_ids=(1101,),
    ),
    "fl": LandUseDefinition(
        lu_long="flower",
        lu_short="fl",
        lveg=True,
        lu_ids=(1102,),
    ),
    "fo": LandUseDefinition(
        lu_long="fodder",
        lu_short="fo",
        lveg=True,
        lu_ids=(1103,),
    ),
    "ma": LandUseDefinition(
        lu_long="maize",
        lu_short="ma",
        lveg=True,
        lu_ids=(1104,),
    ),
    "oa": LandUseDefinition(
        lu_long="oat",
        lu_short="oa",
        lveg=True,
        lu_ids=(1105,),
    ),
    "oc": LandUseDefinition(
        lu_long="other cereal",
        lu_short="oc",
        lveg=True,
        lu_ids=(1106,),
    ),
    "po": LandUseDefinition(
        lu_long="potato",
        lu_short="po",
        lveg=True,
        lu_ids=(1107,),
    ),
    "ra": LandUseDefinition(
        lu_long="rapeseed",
        lu_short="ra",
        lveg=True,
        lu_ids=(1108,),
    ),
    "ry": LandUseDefinition(
        lu_long="rye",
        lu_short="ry",
        lveg=True,
        lu_ids=(1109,),
    ),
    "su": LandUseDefinition(
        lu_long="sugar beet",
        lu_short="su",
        lveg=True,
        lu_ids=(1110,),
    ),
    "wh": LandUseDefinition(
        lu_long="wheat",
        lu_short="wh",
        lveg=True,
        lu_ids=(1111,),
    ),
}

lu_types_depac: dict[LandUseName, LandUseDefinition] = {
    "ara": LandUseDefinition(
        lu_long="Arable land",
        lu_short="ara",
        ifs_id=9,
        lveg=True,
        lu_ids=(11,),
    ),
    "crp": LandUseDefinition(
        lu_long="Permanent crops",
        lu_short="crp",
        ifs_id=16,
        lveg=True,
        lu_ids=(6, 7, 8),
    ),
    "fce": LandUseDefinition(
        lu_long="Coniferous evergreen forest",
        lu_short="fce",
        ifs_id=17,
        lveg=True,
        lu_ids=(2, 3),
    ),
    "fbd": LandUseDefinition(
        lu_long="Broadleaf deciduous forest",
        lu_short="fbd",
        ifs_id=17,
        lveg=True,
        lu_ids=(1, 4, 5, 27),
    ),
    "aqu": LandUseDefinition(
        lu_long="Aquatic",
        lu_short="aqu",
        ifs_id=22,
        laqu=True,
        lu_ids=(14, 15, 16, 17, 18, 19),
    ),
    "brn": LandUseDefinition(
        lu_long="Barren land",
        lu_short="brn",
        ifs_id=10,
        lu_ids=(12, 21, 28),
        bare_soil=True,
    ),
    "sem": LandUseDefinition(
        lu_long="Semi-natural vegetation",
        lu_short="sem",
        ifs_id=10,
        lveg=True,
        lu_ids=(9, 13, 26),
    ),
    "grs": LandUseDefinition(
        lu_long="Grassland",
        lu_short="grs",
        ifs_id=1,
        lveg=True,
        lu_ids=(10,),
    ),
    "urb": LandUseDefinition(
        lu_long="Urban",
        lu_short="urb",
        ifs_id=20,
        lu_ids=(20, 22, 23, 24, 25, 29, 30),
        lslurb=True,
    ),
}

lu_types_ifs: dict[LandUseName, LandUseDefinition] = {
    "cmf": LandUseDefinition(
        lu_long="crops_mixed_farming",
        lu_short="cmf",
        ifs_id=0,
        lveg=True,
    ),
    "sg": LandUseDefinition(
        lu_long="short_grass",
        lu_short="sg",
        ifs_id=1,
        lveg=True,
        bare_soil=True,
    ),
    "en": LandUseDefinition(
        lu_long="evergreen_needleleaf",
        lu_short="en",
        ifs_id=2,
        lveg=True,
    ),
    "dn": LandUseDefinition(
        lu_long="deciduous_needleleaf",
        lu_short="dn",
        ifs_id=3,
        lveg=True,
    ),
    "db": LandUseDefinition(
        lu_long="deciduous_broadleaf",
        lu_short="db",
        ifs_id=4,
        lveg=True,
    ),
    "eb": LandUseDefinition(
        lu_long="evergreen_broadleaf",
        lu_short="eb",
        ifs_id=5,
        lveg=True,
    ),
    "tg": LandUseDefinition(
        lu_long="tall_grass",
        lu_short="tg",
        ifs_id=6,
        lveg=True,
    ),
    "de": LandUseDefinition(lu_long="desert", lu_short="de", ifs_id=7),
    "tu": LandUseDefinition(lu_long="tundra", lu_short="tu", ifs_id=8),
    "ic": LandUseDefinition(
        lu_long="irrigated_crops",
        lu_short="ic",
        ifs_id=9,
        lveg=True,
    ),
    "sd": LandUseDefinition(lu_long="semidesert", lu_short="sd", ifs_id=10),
    "bm": LandUseDefinition(
        lu_long="bogs_marshes",
        lu_short="bm",
        ifs_id=12,
        laqu=True,
    ),
    "es": LandUseDefinition(
        lu_long="evergreen_shrubs",
        lu_short="es",
        ifs_id=15,
        lveg=True,
    ),
    "ds": LandUseDefinition(
        lu_long="deciduous_shrubs",
        lu_short="ds",
        ifs_id=16,
        lveg=True,
    ),
    "mf": LandUseDefinition(
        lu_long="mixed_forest_wood",
        lu_short="mf",
        ifs_id=17,
        lveg=True,
    ),
    "if": LandUseDefinition(
        lu_long="interrupted_forest",
        lu_short="if",
        ifs_id=18,
        lveg=True,
    ),
    "urb": LandUseDefinition(lu_long="urban", lu_short="urb", ifs_id=20, lslurb=True),
    "rd": LandUseDefinition(lu_long="road", lu_short="rd", ifs_id=21),
    "wat": LandUseDefinition(
        lu_long="water",
        lu_short="wat",
        ifs_id=22,
        laqu=True,
    ),
}

_LAND_USE_SETS: dict[LandUseSet, dict[LandUseName, LandUseDefinition]] = {
    LandUseSet.BASIC: lu_types_basic,
    LandUseSet.BUILD: lu_types_build,
    LandUseSet.CROP: lu_types_crop,
    LandUseSet.DEPAC: lu_types_depac,
    LandUseSet.IFS: lu_types_ifs,
}


def get_land_use_definitions(
    choice: LandUseSet | str,
) -> dict[LandUseName, LandUseDefinition]:
    """Return a fresh catalogue dictionary containing shared frozen definitions.

    Basic, build and crop definitions have no IFS ids. Callers needing IFS
    properties must supply a mapping rather than use source ids as IFS ids.
    """
    try:
        selected = LandUseSet(choice)
    except ValueError:
        choices = ", ".join(member.value for member in LandUseSet)
        raise ValueError(
            f"Unknown land-use set {choice!r}; choose one of: {choices}"
        ) from None
    return _LAND_USE_SETS[selected].copy()
