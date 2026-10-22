MODULE_REGISTRY: dict[str, type] = {}
SINGLETON_REGISTRY: dict[str, type] = {}  # Keep track of singleton module names
SPECIAL_SERIALIZING_REGISTRY: dict[
    str, type
] = {}  # For modules that require special handling during serialization


def register_module[T: type](cls: T) -> T:
    """Class decorator to register simulation modules."""
    MODULE_REGISTRY[cls.__name__] = cls
    return cls


def register_singleton[T: type](cls: T) -> T:
    """Class decorator to register a singleton simulation module."""
    SINGLETON_REGISTRY[cls.__name__] = cls
    return cls


def register_special_serializing[T: type](cls: T) -> T:
    """Class decorator to register a module that requires special handling during serialization."""
    SPECIAL_SERIALIZING_REGISTRY[cls.__name__] = cls
    return cls
