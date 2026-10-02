# Source - https://stackoverflow.com/a
# Posted by HanSooloo, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-16, License - CC BY-SA 4.0
import logging
import logging.config
import pathlib

import yaml

from modular_dales.helpers import package_resource_path


# At the beginning of every .py file in the project
def logwrap(fn):
    return fn
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(fn.__module__)
        module = None
        for arg in args:
            if hasattr(arg, "module_name"):
                module = arg.module_name
                break
        else:
            for arg in kwargs.values():
                if hasattr(arg, "module_name"):
                    module = arg.module_name
        if module:
            logger.debug("Entering %s (module: %s)", fn.__name__, module)
        else:
            logger.debug("Entering %s", fn.__name__)
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            logger.error("Function %s failed with exception %s: ", fn.__name__, e.args)
            raise e
        if module:
            logger.debug("Exiting %s (module: %s)", fn.__name__, module)
        else:
            logger.debug("Exiting %s", fn.__name__)
        # Return the return value
        return out

    return wrapper


def setup_logging(config_path="logging.yaml"):
    path = pathlib.Path(config_path)
    if path.exists():
        _configure_logging(path)
        return

    if path.name == "logging.yaml":
        with package_resource_path("logging.yaml") as default_path:
            _configure_logging(default_path)
        return

    logging.basicConfig()


def _configure_logging(config_path):
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    for handler in config.get("handlers", {}).values():
        filename = handler.get("filename")
        if filename:
            pathlib.Path(filename).expanduser().parent.mkdir(
                parents=True, exist_ok=True
            )

    logging.config.dictConfig(config)
