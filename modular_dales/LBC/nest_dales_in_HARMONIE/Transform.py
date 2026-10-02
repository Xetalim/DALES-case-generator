import logging

from pyproj import Transformer

logger = logging.getLogger(__name__)
logger.debug("Entered module: %s", __name__)


class Transform:
    def __init__(self, parameters):
        self.parameters = parameters
        self.crs_latlon = "epsg:4326"
        # Construct transformation objects
        self.latlon_to_xy_transform = Transformer.from_crs(
            self.crs_latlon, self.parameters["proj4"]
        )
        self.xy_to_latlon_transform = Transformer.from_crs(
            self.parameters["proj4"], self.crs_latlon
        )

    def latlon_to_xy(self, lat, lon):
        return self.latlon_to_xy_transform.transform(lat, lon)

    def xy_to_latlon(self, x, y):
        return self.xy_to_latlon_transform.transform(x, y)
