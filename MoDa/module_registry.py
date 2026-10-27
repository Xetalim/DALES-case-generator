module_registry: dict[str, type] = {}


def register_module[T: type](cls: T) -> T:
    """Register a type for simulation loading and typed serialization."""
    module_registry[cls.__name__] = cls
    return cls
