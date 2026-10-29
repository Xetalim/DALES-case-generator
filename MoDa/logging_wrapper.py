# Source - https://stackoverflow.com/a
# Posted by HanSooloo, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-16, License - CC BY-SA 4.0
import logging
import logging.config
import pathlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import ParamSpec, TypeVar

import yaml

from moda.helpers import package_resource_path

P = ParamSpec("P")
R = TypeVar("R")


# At the beginning of every .py file in the project
def logwrap[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
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
            raise
        if module:
            logger.debug("Exiting %s (module: %s)", fn.__name__, module)
        else:
            logger.debug("Exiting %s", fn.__name__)
        # Return the return value
        return out

    return wrapper


def setup_logging(config_path: str = "logging.yaml") -> None:
    path = pathlib.Path(config_path)
    if path.exists():
        _configure_logging(path)
        return

    if path.name == "logging.yaml":
        with package_resource_path("logging.yaml") as default_path:
            _configure_logging(default_path)
        return

    logging.basicConfig()


def _configure_logging(config_path: pathlib.Path) -> None:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    for handler in config.get("handlers", {}).values():
        filename = handler.get("filename")
        if filename:
            pathlib.Path(filename).expanduser().parent.mkdir(
                parents=True, exist_ok=True
            )

    logging.config.dictConfig(config)


@contextmanager
def case_logging(case_path: pathlib.Path) -> Iterator[None]:
    root_logger = logging.getLogger()
    handler = logging.FileHandler(case_path / "app.log", mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    formatter = next(
        (
            existing.formatter
            for existing in root_logger.handlers
            if isinstance(existing, logging.FileHandler)
            and existing.formatter is not None
        ),
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        handler.close()
