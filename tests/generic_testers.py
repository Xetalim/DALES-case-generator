from __future__ import annotations

import logging
import traceback

import pytest

from modular_dales.Configuration.output_modules import CheckSimulationModule
from modular_dales.Configuration.run_and_time import TimeModule
from modular_dales.modular import dales_simulation

from .helpers import SimulationReport, assert_dirs_equal, run_command_with_report
from .sim_builders.test_basic import _build_basic_sim

logger = logging.getLogger(__name__)


def assert_roundtrip_simulation_outputs_identical(
    sim_builder, machine_conf, add_report: SimulationReport | None = None
) -> None:
    """Run a round-trip YAML serialization test for a given simulation builder.

    Builds the simulation, runs preprocessing, serializes to YAML, loads a new
    simulation from YAML with a different BASE_OUTPUT_PATH, runs preprocessing
    again, and asserts the two output directories are bit-for-bit identical.
    """

    builder_name = getattr(sim_builder, "__name__", type(sim_builder).__name__)
    stage = "init"
    sim1 = None
    sim2 = None
    out1 = None
    out2 = None

    try:
        stage = "build_from_scratch"
        machine_conf_1 = machine_conf("from_scratch")
        sim1 = sim_builder(machine_conf_1)

        stage = "preprocess_from_scratch"
        sim1.sim_preprocessing_pipeline()
        out1 = sim1.output_path

        stage = "serialize_yaml"
        yaml_text = sim1.save_sim_to_yaml()

        stage = "build_from_yaml"
        machine_conf_2 = machine_conf("from_yaml")
        sim2 = dales_simulation.load_sim_from_yaml(
            yaml_text, machine_conf=machine_conf_2
        )

        stage = "preprocess_from_yaml"
        sim2.sim_preprocessing_pipeline()
        out2 = sim2.output_path

        stage = "validate_output_dirs"
        assert out1 != out2
        assert out1.is_dir() and out2.is_dir()

        stage = "roundtrip_output_compare"
        assert_dirs_equal(out1, out2)
    except Exception:
        case_name = getattr(sim1, "case_name", "<unavailable>")
        out1_str = str(out1) if out1 is not None else "<unavailable>"
        out2_str = str(out2) if out2 is not None else "<unavailable>"

        if add_report is not None:
            add_report(
                f"{case_name} roundtrip failed",
                {
                    "crash_message": traceback.format_exc(),
                    "log_messages": "Roundtrip simulation failed; inspect stage and case directories.",
                    "case_dir": out1_str,
                    "comparison_case_dir": out2_str,
                    "info_and_location": "\n".join(
                        [
                            f"stage={stage}",
                            f"sim_builder={builder_name}",
                            f"case_name={case_name}",
                            f"case_dir={out1_str}",
                            f"comparison_case_dir={out2_str}",
                            f"from_scratch_output={out1_str}",
                            f"from_yaml_output={out2_str}",
                        ]
                    ),
                },
            )

        pytest.fail(
            "\n".join(
                [
                    f"Roundtrip simulation failed at stage '{stage}'",
                    f"sim_builder={builder_name}",
                    f"case_name={case_name}",
                    f"case_dir={out1_str}",
                    f"comparison_case_dir={out2_str}",
                    f"error={traceback.format_exc()}",
                ]
            ),
            pytrace=False,
        )


def run_simulation_and_check_job(
    sim_builder, machine_conf, add_report: SimulationReport | None = None
) -> None:
    """Run a short simulation using a builder and then execute job.001.

    The simulation is configured with a small runtime and a tcheck value so
    the external job.001 can verify the produced input.
    """

    machine_conf_1 = machine_conf("from_scratch")

    post_run_checker = None
    if isinstance(sim_builder, tuple):
        sim_builder, post_run_checker = sim_builder

    sim1 = sim_builder(machine_conf_1)
    if sim1.nml.get("RUN") is None:
        sim1.nml["RUN"] = {}
    if sim1.nml["RUN"].get("nprocx") is None:
        sim1.nml["RUN"]["nprocx"] = 0
    if sim1.nml["RUN"].get("nprocy") is None:
        sim1.nml["RUN"]["nprocy"] = 0
    sim1 += CheckSimulationModule(
        check_interval=10, stop_on_invalid=False, check_tendencies=True
    )
    sim1.sim_preprocessing_pipeline()
    out1 = sim1.output_path

    builder_name = getattr(sim_builder, "__name__", type(sim_builder).__name__)
    run_command_with_report(
        ["./job.001"],
        stage="job_001",
        case_dir=out1,
        title=f"{sim1.case_name} job.001 crash",
        add_report=add_report,
        info_lines=[f"sim_builder={builder_name}"],
    )

    if post_run_checker is not None:
        post_run_checker(out1)


def test_diff_works(machine_conf) -> None:
    """Test that outputs differ if we change the time settings after reload.

    This reproduces the original test that ensured _assert_dirs_equal detects
    differences when the simulation configuration is modified after
    serialization.
    """

    machine_conf_1 = machine_conf("from_scratch")
    case_name = "012_TESTS_roundtrip_test_case"  # preserved for debugging parity

    sim1 = _build_basic_sim(machine_conf_1)
    sim1.sim_preprocessing_pipeline()
    out1 = sim1.output_path

    yaml_text = sim1.save_sim_to_yaml()

    machine_conf_2 = machine_conf("from_yaml")
    sim2 = dales_simulation.load_sim_from_yaml(yaml_text, machine_conf=machine_conf_2)
    for mod in sim2.modules:
        if isinstance(mod, TimeModule):
            mod.xday = 2
    logger.info("Set time to 2 for roundtrip diff test (case %s)", case_name)
    sim2.sim_preprocessing_pipeline()
    out2 = sim2.output_path

    assert out1 != out2
    assert out1.is_dir() and out2.is_dir()

    with pytest.raises(ValueError):
        assert_dirs_equal(out1, out2)


def assert_files_written(sim_builder_for_write_test, machine_conf) -> None:
    """Tests per case if the necessary files are written"""

    machine_conf_1 = machine_conf("assert_files_written")

    sim1, written_tester = sim_builder_for_write_test(machine_conf_1)
    sim1.sim_preprocessing_pipeline()
    written_tester(sim1.output_path)
