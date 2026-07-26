from pathlib import Path


FORBIDDEN = ("edu" + "-ai-prototype", "edu" + "_ai", "pipeline" + "_runner")


def test_clear_implementation_does_not_reference_old_project():
    root = Path(__file__).resolve().parents[1] / "clear"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    for token in FORBIDDEN:
        assert token not in text


def test_core_package_does_not_import_benchmark_or_internal_experiment_modules():
    root = Path(__file__).resolve().parents[1] / "clear"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    assert "clear.experiments" not in text
    assert "lss_bench" not in text
