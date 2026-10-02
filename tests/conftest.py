from __future__ import annotations

import html
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest
import matplotlib

matplotlib.use("Agg")

from .helpers import load_machine_conf

_REPORT_PAYLOAD_ATTR = "_simulation_report_payloads"


def _flatten_payload(content: Any) -> str:
    if isinstance(content, dict):
        blocks: list[str] = []
        for key, value in content.items():
            blocks.append(f"{key}:\n{value}")
        return "\n\n".join(blocks)
    return str(content)


def _render_payload_html(title: str, content: Any) -> str:
    if not isinstance(content, dict):
        return (
            "<details open><summary>{title}</summary><pre>{content}</pre></details>"
        ).format(title=html.escape(title), content=html.escape(str(content)))

    info = html.escape(str(content.get("info_and_location", "<empty>")))
    crash = html.escape(str(content.get("crash_message", "<empty>")))
    logs = html.escape(str(content.get("log_messages", "<empty>")))
    case_dir = html.escape(str(content.get("case_dir", "<empty>")))
    comparison_case_dir = html.escape(
        str(content.get("comparison_case_dir", "<empty>"))
    )
    return "".join(
        [
            "<section style='margin:0.75rem 0;padding:0.75rem;border:1px solid #d0d7de;border-radius:6px;'>",
            f"<div style='font-weight:600;margin-bottom:0.5rem;'>{html.escape(title)}</div>",
            "<details open><summary>Crash message</summary><pre>",
            crash,
            "</pre></details>",
            "<details open><summary>Case directories</summary><pre>",
            f"case_dir={case_dir}\ncomparison_case_dir={comparison_case_dir}",
            "</pre></details>",
            "<details><summary>Log messages</summary><pre>",
            logs,
            "</pre></details>",
            "<details open><summary>Info and location</summary><pre>",
            info,
            "</pre></details>",
            "</section>",
        ]
    )


def pytest_html_results_table_header(cells):
    cells.insert(2, "<th>Simulation</th>")


def pytest_html_results_table_row(report, cells):
    titles = getattr(report, "simulation_report_titles", ())
    if titles:
        summary = "<br>".join(html.escape(title) for title in titles)
    else:
        summary = "&nbsp;"
    cells.insert(2, f"<td>{summary}</td>")


@pytest.fixture
def simulation_report(request: pytest.FixtureRequest) -> Callable[[str, Any], None]:
    def _add_report(title: str, content: Any) -> None:
        payloads = getattr(request.node, _REPORT_PAYLOAD_ATTR, None)
        if payloads is None:
            payloads = []
            setattr(request.node, _REPORT_PAYLOAD_ATTR, payloads)
        payloads.append((title, content))

    return _add_report


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    _ = call
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return

    payloads = getattr(item, _REPORT_PAYLOAD_ATTR, ())
    if not payloads:
        return

    report.simulation_report_titles = [title for title, _ in payloads]
    report.simulation_report_html = "".join(
        _render_payload_html(title, content) for title, content in payloads
    )

    for title, content in payloads:
        report.sections.append((title, _flatten_payload(content)))

    if not item.config.pluginmanager.hasplugin("html"):
        return


def pytest_html_results_table_html(report, data):
    details_html = getattr(report, "simulation_report_html", "")
    if not details_html:
        return
    del data[:]
    data.append(details_html)


@pytest.fixture
def machine_conf(tmp_path: Path) -> Callable[[str], dict]:
    """Provide a machine configuration with a temporary BASE_OUTPUT_PATH.

    Returns a function that takes a case name and returns a machine
    configuration dict with BASE_OUTPUT_PATH pointing to a unique
    temporary directory for that case.
    """

    def _machine_conf(tmp_path: Path, casename: str) -> dict:
        conf = deepcopy(load_machine_conf())
        base = tmp_path / f"case_{casename}"
        base.mkdir(parents=True, exist_ok=True)
        conf["case_conf"]["BASE_OUTPUT_PATH"] = str(base)
        return conf

    return lambda casename: _machine_conf(tmp_path, casename)
