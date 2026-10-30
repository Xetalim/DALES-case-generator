"""Common application contract for composable simulation sources."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Source[Context](Protocol):
    """Apply a source to the context supplied by its owning module."""

    def apply_source(self, context: Context) -> None: ...
