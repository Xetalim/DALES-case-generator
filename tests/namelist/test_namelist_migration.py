import ast
import importlib
import inspect
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from docs.conf import _append_namelist_field_docs
from MoDa.modular.dales_simulation import dales_simulation
from MoDa.modular.namelist import get_namelist_spec, nml_field
from MoDa.modular.simulation_module import simulation_module
from tests.namelist.test_namelist_fields import NamelistModule

ROOT = Path(__file__).resolve().parents[1]
LEGACY_KEYS = {"nml", "key", "required", "raise_conflict", "doc", "units"}


def migrated_modules():
    for path in sorted((ROOT / "MoDa").rglob("*.py")):
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "nml_field"
            for node in ast.walk(tree)
        ):
            yield ".".join(path.relative_to(ROOT).with_suffix("").parts)


def test_source_has_no_legacy_namelist_declarations():
    for path in (ROOT / "MoDa").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "metadata" or not isinstance(keyword.value, ast.Dict):
                    continue
                keys = {
                    key.value
                    for key in keyword.value.keys
                    if isinstance(key, ast.Constant)
                }
                assert not {"nml", "key"}.intersection(keys), f"{path}:{node.lineno}"


@pytest.mark.parametrize("module_name", list(migrated_modules()))
def test_active_bindings_are_typed(module_name):
    module = importlib.import_module(module_name)
    count = 0
    for name, definition in inspect.getmembers(module, inspect.isclass):
        if definition.__module__ != module_name or not is_dataclass(definition):
            continue
        for dataclass_field in fields(definition):
            assert not {"nml", "key"}.intersection(dataclass_field.metadata), name
            spec = get_namelist_spec(dataclass_field)
            if spec is not None:
                assert not LEGACY_KEYS.intersection(dataclass_field.metadata), name
                assert spec.targets(dataclass_field.name)
                count += 1
    assert count


@pytest.mark.parametrize(
    "section,key,targets",
    [
        ("A", None, [("A", "value")]),
        (("A", "B"), None, [("A", "value"), ("B", "value")]),
        ("A", ("x", "y"), [("A", "x"), ("A", "y")]),
        (("A", "B"), ("x", "y"), [("A", "x"), ("B", "y")]),
    ],
)
def test_docs_render_typed_bindings(section, key, targets):
    @dataclass
    class Documented(NamelistModule):
        value: float = nml_field(
            section=section,
            key=key,
            default=1.0,
            required=True,
            description="Surface pressure",
            units="Pa",
        )

    lines = ["Module documentation."]
    _append_namelist_field_docs("class", Documented, lines)
    assert lines[1] == ""
    for target_section, target_key in targets:
        assert f"*Section* ``{target_section}``:" in lines
        assert (
            f"  - ``{target_key}`` (field ``value``) (required) [Pa]: Surface pressure"
        ) in lines


def test_docs_reject_legacy_metadata():
    @dataclass
    class Invalid(NamelistModule):
        value: float = field(default=1.0, metadata={"nml": "TEST"})

    with pytest.raises(TypeError, match="must use nml_field or NamelistSpec"):
        _append_namelist_field_docs("class", Invalid, [])


@pytest.mark.parametrize("obj", [None, int, SimpleNamespace(), simulation_module])
def test_docs_ignore_non_dataclass_modules(obj):
    lines = []
    _append_namelist_field_docs("class", obj, lines)
    assert lines == []


def test_migrated_forcing_metadata(monkeypatch):
    from MoDa.Surface.surface import ConstantFluxesModule

    definition = next(
        item for item in fields(ConstantFluxesModule) if item.name == "ps"
    )
    assert definition.metadata["forcing_var"] == "psurf"
    module = ConstantFluxesModule(z0mav=0.1, z0hav=0.01)
    module._attach_to_sim(dales_simulation("test", {}))
    monkeypatch.setattr(
        module, "_external_forcing_start_value", lambda variable: 99000.0
    )
    module.apply_namelist_from_fields()
    for section, key in get_namelist_spec(definition).targets(definition.name):
        assert module.nml[section][key] == 99000.0
