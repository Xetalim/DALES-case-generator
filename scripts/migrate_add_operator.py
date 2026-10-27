#!/usr/bin/env python3
"""Migrate recognized deprecated builder ``+=`` calls to ``.add(...)``.

Run from the repository root. By default this prints a preview; pass --write
for in-place changes. Only receivers resolved to known deprecated collection
APIs are rewritten. Other and unresolved ``+=`` operations are reported.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEPRECATED_TYPES = {
    "dales_simulation",
    "EmissionModule",
    "IBMModule",
    "IBMModifications",
    "LandUseModifications",
    "NestingTopology",
    "SLURBModifications",
    "SLURBModule",
}
DEFAULT_PATHS = (Path("modular_dales"), Path("tests"), Path("sample_case.py"))


@dataclass(frozen=True)
class Rewrite:
    start: int
    end: int
    replacement: str
    line: int
    receiver_type: str


def _type_name(annotation: ast.expr | None, aliases: dict[str, str]) -> str | None:
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name):
        name = aliases.get(annotation.id, annotation.id)
        return name if name in DEPRECATED_TYPES else None
    if isinstance(annotation, ast.Attribute):
        name = aliases.get(annotation.attr, annotation.attr)
        return name if name in DEPRECATED_TYPES else None
    if isinstance(annotation, ast.Subscript):
        return _type_name(annotation.value, aliases)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _type_name(annotation.left, aliases) or _type_name(annotation.right, aliases)
    return None


def _call_type(
    expression: ast.expr,
    aliases: dict[str, str],
    return_types: dict[str, str],
) -> str | None:
    if not isinstance(expression, ast.Call):
        return None
    function = expression.func
    if isinstance(function, ast.Name):
        name = aliases.get(function.id, function.id)
    elif isinstance(function, ast.Attribute):
        name = aliases.get(function.attr, function.attr)
    else:
        return None
    return name if name in DEPRECATED_TYPES else return_types.get(name)


def _scopes(
    tree: ast.Module,
) -> Iterable[ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda]:
    yield tree
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            yield node


def _target_name(target: ast.expr) -> str | None:
    return target.id if isinstance(target, ast.Name) else None


def _scope_nodes(scope: ast.AST) -> Iterable[ast.AST]:
    def visit(node: ast.AST) -> Iterable[ast.AST]:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                continue
            yield child
            yield from visit(child)

    yield from visit(scope)


def _function_return_types(tree: ast.Module, aliases: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            resolved = _type_name(node.returns, aliases)
            if resolved:
                result[node.name] = resolved
    return result


def _imported_return_types(
    tree: ast.Module,
    filename: Path,
    aliases: dict[str, str],
    root: Path,
) -> dict[str, str]:
    imported: dict[str, str] = {}
    try:
        relative = filename.resolve().relative_to(root.resolve())
    except ValueError:
        return imported

    package_parts = list(relative.with_suffix("").parts[:-1])
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            module_parts = package_parts[: len(package_parts) - node.level + 1]
            if node.module:
                module_parts.extend(node.module.split("."))
        else:
            module_parts = node.module.split(".") if node.module else []
        module_path = root.joinpath(*module_parts)
        candidates = [module_path.with_suffix(".py"), module_path / "__init__.py"]
        source_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if source_path is None:
            continue
        try:
            imported_tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        imported_aliases = _import_aliases(imported_tree)
        for definition in ast.walk(imported_tree):
            if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            result_type = _type_name(definition.returns, imported_aliases)
            if result_type is None:
                continue
            for name in node.names:
                if name.name == definition.name:
                    imported[name.asname or name.name] = result_type
    return imported


def _known_types(
    scope: ast.AST,
    aliases: dict[str, str],
    return_types: dict[str, str],
) -> dict[str, str]:
    known: dict[str, str] = {}
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        arguments = [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs]
        if scope.args.vararg:
            arguments.append(scope.args.vararg)
        if scope.args.kwarg:
            arguments.append(scope.args.kwarg)
        for argument in arguments:
            resolved = _type_name(argument.annotation, aliases)
            if resolved:
                known[argument.arg] = resolved

    statements = sorted(
        (
            node
            for node in _scope_nodes(scope)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for statement in statements:
        if isinstance(statement, ast.AnnAssign):
            target = _target_name(statement.target)
            resolved = _type_name(statement.annotation, aliases)
            if target and resolved:
                known[target] = resolved
            elif target and statement.value:
                inferred = _call_type(statement.value, aliases, return_types)
                if inferred:
                    known[target] = inferred
            continue

        inferred = _call_type(statement.value, aliases, return_types)
        if inferred is None and isinstance(statement.value, ast.Name):
            inferred = known.get(statement.value.id)
        for target_node in statement.targets:
            target = _target_name(target_node)
            if target:
                if inferred:
                    known[target] = inferred
                else:
                    known.pop(target, None)
    return known


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for item in node.names:
                local = item.asname or item.name
                if item.name in DEPRECATED_TYPES:
                    aliases[local] = item.name
        elif isinstance(node, ast.Import):
            for item in node.names:
                short_name = item.name.rsplit(".", 1)[-1]
                local = item.asname or short_name
                if short_name in DEPRECATED_TYPES:
                    aliases[local] = short_name
    return aliases


def _absolute_offset(source: str, lineno: int, utf8_col: int) -> int:
    lines = source.splitlines(keepends=True)
    return sum(len(line) for line in lines[: lineno - 1]) + len(
        lines[lineno - 1][:utf8_col].decode("utf-8")
        if isinstance(lines[lineno - 1], bytes)
        else lines[lineno - 1].encode("utf-8")[:utf8_col].decode("utf-8")
    )


def _rewrites(source: str, filename: Path) -> tuple[list[Rewrite], list[tuple[int, str]]]:
    tree = ast.parse(source, filename=str(filename))
    aliases = _import_aliases(tree)
    return_types = _function_return_types(tree, aliases)
    return_types.update(_imported_return_types(tree, filename, aliases, Path.cwd()))
    rewrites: list[Rewrite] = []
    unresolved: list[tuple[int, str]] = []
    scopes = list(_scopes(tree))

    for scope in scopes:
        known = _known_types(scope, aliases, return_types)
        for node in _scope_nodes(scope):
            if not isinstance(node, ast.AugAssign) or not isinstance(node.op, ast.Add):
                continue
            if not isinstance(node.target, ast.Name):
                continue
            receiver = node.target.id
            receiver_type = known.get(receiver)
            if receiver_type not in DEPRECATED_TYPES:
                if receiver_type is None:
                    unresolved.append((node.lineno, receiver))
                continue
            start = _absolute_offset(source, node.target.lineno, node.target.col_offset)
            value_end = _absolute_offset(source, node.value.end_lineno, node.value.end_col_offset)
            target_source = ast.get_source_segment(source, node.target)
            value_source = ast.get_source_segment(source, node.value)
            if target_source is None or value_source is None:
                unresolved.append((node.lineno, receiver))
                continue
            rewrites.append(
                Rewrite(
                    start=start,
                    end=value_end,
                    replacement=f"{target_source}.add({value_source})",
                    line=node.lineno,
                    receiver_type=receiver_type,
                )
            )

    unique = {(item.start, item.end): item for item in rewrites}
    return sorted(unique.values(), key=lambda item: item.start), sorted(set(unresolved))


def _apply(source: str, rewrites: list[Rewrite]) -> str:
    for rewrite in reversed(rewrites):
        source = source[: rewrite.start] + rewrite.replacement + source[rewrite.end :]
    return source


def _python_files(paths: list[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.py"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_PATHS))
    parser.add_argument("--write", action="store_true", help="write migrated files in place")
    args = parser.parse_args()

    changed_files = 0
    rewritten_sites = 0
    unresolved_total = 0
    for path in _python_files(args.paths):
        source = path.read_text(encoding="utf-8")
        rewrites, unresolved = _rewrites(source, path)
        unresolved_total += len(unresolved)
        for rewrite in rewrites:
            print(f"{path}:{rewrite.line}: {rewrite.receiver_type} += ... -> .add(...)")
        for line, receiver in unresolved:
            print(f"{path}:{line}: skipped unresolved += receiver {receiver!r}")
        if rewrites:
            changed_files += 1
            rewritten_sites += len(rewrites)
            if args.write:
                path.write_text(_apply(source, rewrites), encoding="utf-8")

    action = "Migrated" if args.write else "Would migrate"
    print(
        f"{action} {rewritten_sites} call site(s) in {changed_files} file(s); "
        f"skipped {unresolved_total} unresolved site(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
