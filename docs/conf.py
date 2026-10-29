"""
Sphinx configuration file for the DALES Case Generator project documentation.

This module configures Sphinx for generating documentation from the codebase.
It sets up extensions for automatic documentation generation, API documentation,
and cross-referencing with external documentation sources.
"""

import os
import sys
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime

# Ensure the project root is on sys.path so `moda` can be imported
sys.path.insert(0, os.path.abspath(".."))

from moda.Core.namelist import get_namelist_spec
from moda.modular.simulation_module import simulation_module

project = "DALES Case Generator"  # pylint: disable=C0103
author = "Project contributors"  # pylint: disable=C0103
current_year = datetime.now(UTC).year
copyright = f"{current_year}, {author}"  # pylint: disable=W0622,C0103

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_markdown_builder",
]

autosummary_generate = True  # pylint: disable=C0103

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True  # pylint: disable=C0103
napoleon_numpy_docstring = True  # pylint: disable=C0103

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"  # pylint: disable=C0103

html_static_path = ["_static"]


def _append_namelist_field_docs(what, obj, lines):
    """Append a "Namelist parameters" section for simulation modules.

    Any dataclass field on a subclass of ``simulation_module`` that carries
    a typed ``nml_field`` binding will be listed, so the rendered
    HTML explicitly shows which namelist sections and keys a module controls.
    """

    if what != "class":
        return

    # Only consider subclasses of the core simulation_module base class
    try:
        if not issubclass(obj, simulation_module):
            return
    except TypeError:
        # obj is not a class
        return

    if not is_dataclass(obj):
        return

    namelist_entries = {}

    for field in fields(obj):  # type: ignore[arg-type]
        spec = get_namelist_spec(field)
        if spec is None:
            continue
        for section, key in spec.targets(field.name):
            namelist_entries.setdefault(str(section), []).append(
                {
                    "key": str(key),
                    "field": field.name,
                    "required": spec.required,
                    "description": spec.description,
                    "units": spec.units,
                }
            )

    if not namelist_entries:
        return

    # Add a blank line to separate from existing docstring content
    if lines and lines[-1].strip():
        lines.append("")

    lines.append("Namelist parameters")
    lines.append("--------------------")
    lines.append("")

    for section in sorted(namelist_entries):
        lines.append(f"*Section* ``{section}``:")
        for entry in sorted(namelist_entries[section], key=lambda e: e["key"]):
            suffix = " (required)" if entry["required"] else ""
            if entry["units"]:
                suffix += f" [{entry['units']}]"
            if entry["description"]:
                suffix += f": {entry['description']}"
            lines.append(f"  - ``{entry['key']}`` (field ``{entry['field']}``){suffix}")


def setup(app):
    """Sphinx extension hook to enhance autodoc output.

    Currently this adds explicit lists of namelist parameters for modules
    that inherit from ``simulation_module`` and declare ``nml_field`` bindings on
    their dataclass fields.
    """

    app.connect(
        "autodoc-process-docstring",
        lambda app, what, name, obj, options, lines: _append_namelist_field_docs(
            what, obj, lines
        ),
    )

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
