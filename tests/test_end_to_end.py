from pathlib import Path

from keplerbench.cli import main


def test_generate_baseline_evaluate(tmp_path: Path, capsys):
    dataset = tmp_path / "world.csv"
    predictions = tmp_path / "predictions.csv"
    main(["generate", "--output", str(dataset), "--samples", "81", "--seed", "2"])
    main(["baseline", "--data", str(dataset), "--predictions", str(predictions), "--order", "4"])
    main(["evaluate-prediction", "--truth", str(dataset), "--predictions", str(predictions)])
    assert dataset.exists()
    assert dataset.with_suffix(".csv.metadata.json").exists()
    assert predictions.exists()
    assert "mean_arcminutes" in capsys.readouterr().out
