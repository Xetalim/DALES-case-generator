from pathlib import Path

import pytest

from docs.examples.first_case import build_first_case
from docs.examples.time_dependent_case import build_time_dependent_case


@pytest.mark.parametrize(
    "build_case",
    [build_first_case, build_time_dependent_case],
    ids=["first-case", "time-dependent-case"],
)
def test_documented_case_generates_expected_files(tmp_path: Path, build_case) -> None:
    sim = build_case(tmp_path / "generated")
    assert (sim.output_path / "job.001").is_file()
    assert (sim.output_path / "input/namoptions.001").is_file()
    assert (sim.output_path / "input/docs_namoptions.001.md").is_file()
    assert (sim.output_path / "input/simulation_config.001.yaml").is_file()
    assert (sim.output_path / "input/init.001.nc").is_file()
    if build_case is build_time_dependent_case:
        assert (sim.output_path / "input/forcings.001.nc").is_file()
