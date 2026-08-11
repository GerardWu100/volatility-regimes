"""Integration tests for the walk-forward research CLI.

These tests verify that the runner writes the expected research artifacts,
uses the walk-forward config at runtime, and respects a leakage-safe embargo
between the training labels and each forward-looking test chunk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import tomllib


def test_walkforward_config_defaults_use_demo_profile() -> None:
    """Default walk-forward config should stay on the lighter demo profile."""
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "walkforward.toml"

    with config_path.open("rb") as file_handle:
        config = tomllib.load(file_handle)

    assert config["evaluation"]["horizons"] == [20]
    assert config["evaluation"]["min_train_size"] == 2520
    assert config["regime"]["hmm_random_restarts"] == 5
    assert config["regime"]["fixed_k_values"] == []


def test_walkforward_cli_writes_summary_outputs_and_applies_embargo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runner should write outputs, expected model rows, and embargoed train sets."""
    import volatility_regimes.walkforward.engine as cli

    dates = pd.bdate_range("2024-01-02", periods=40)
    prices = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0 + float(index) for index in range(40)],
        }
    )
    feature_rows = pd.DataFrame(
        {
            "trade_date": dates,
            "atm_iv_near": [0.18 + 0.002 * index for index in range(40)],
            "atm_iv_mid": [0.20 + 0.002 * index for index in range(40)],
            "skew_near": [0.04] * 40,
            "skew_mid": [0.03] * 40,
            "butterfly_near": [0.01] * 40,
            "butterfly_mid": [0.01] * 40,
            "term_slope": [0.02] * 40,
        }
    ).set_index("trade_date")

    output_dir = tmp_path / "outputs" / "walkforward"
    output_dir.mkdir(parents=True, exist_ok=True)
    observed_train_windows: list[dict[str, object]] = []

    def fake_linear_features_batch(
        train_features: pd.DataFrame,
        train_target: pd.Series,
        test_features: pd.DataFrame,
    ) -> pd.Series:
        """Record the train boundary once for each row in the test block."""
        for test_date in test_features.index:
            observed_train_windows.append(
                {
                    "model_name": "linear_features",
                    "train_end": train_features.index.max(),
                    "train_size": len(train_features),
                    "test_date": test_date,
                }
            )
        return pd.Series(
            float(train_target.mean()),
            index=test_features.index,
            dtype=float,
        )

    def fake_regime_mean_batch(
        train_features: pd.DataFrame,
        train_target: pd.Series,
        test_features: pd.DataFrame,
        model_type: str,
        min_k: int,
        max_k: int,
        hmm_n_iter: int = 200,
        hmm_random_restarts: int = 10,
    ) -> pd.DataFrame:
        """Record the train boundary once for each regime test row."""
        _ = model_type
        _ = min_k
        _ = max_k
        _ = hmm_n_iter
        _ = hmm_random_restarts
        for test_date in test_features.index:
            observed_train_windows.append(
                {
                    "model_name": "gmm_regime_mean",
                    "train_end": train_features.index.max(),
                    "train_size": len(train_features),
                    "test_date": test_date,
                }
            )
        return pd.DataFrame(
            {
                "prediction": float(train_target.mean()),
                "selected_k": 2,
                "predicted_regime": 1,
            },
            index=test_features.index,
        )

    # Patch file-system and data loading dependencies so the test stays local.
    monkeypatch.setattr(cli, "_load_symbol_inputs", lambda *_: (feature_rows, prices))
    monkeypatch.setattr(cli, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        cli,
        "forecast_linear_features_batch",
        fake_linear_features_batch,
    )
    monkeypatch.setattr(
        cli,
        "forecast_regime_mean_batch",
        fake_regime_mean_batch,
    )

    cli.run_research(
        symbols=["SPX"],
        feature_sets=["atm_term"],
        horizons=[5],
        min_train_size=20,
        step_size=5,
        annualization=252,
        regime_min_k=2,
        regime_max_k=2,
    )

    assert (output_dir / "forecast_panel.csv").exists()
    assert (output_dir / "metric_summary.csv").exists()
    assert (output_dir / "research_summary.md").exists()

    forecast_panel = pd.read_csv(
        output_dir / "forecast_panel.csv", parse_dates=["date"]
    )
    metric_summary = pd.read_csv(output_dir / "metric_summary.csv")

    expected_model_names = {
        "atm_iv",
        "historical_mean",
        "trailing_realized_vol",
        "linear_features",
        "gmm_regime_mean",
    }
    assert set(forecast_panel["model_name"]) == expected_model_names
    assert set(metric_summary["model_name"]) == expected_model_names
    assert len(forecast_panel) == 25
    assert set(metric_summary["n_forecasts"]) == {5}

    date_to_position = {date: position for position, date in enumerate(dates)}
    for window in observed_train_windows:
        train_end = pd.Timestamp(window["train_end"])
        test_date = pd.Timestamp(window["test_date"])
        train_position = date_to_position[train_end]
        test_position = date_to_position[test_date]

        # realized_vol_t uses returns from t+1 through t+h, so train_end+h
        # must stay strictly before the current test date to avoid overlap.
        assert train_position + 5 < test_position

    observed_train_sizes = {
        int(window["train_size"]) for window in observed_train_windows
    }
    assert observed_train_sizes == {20}


def test_walkforward_fails_fast_when_safe_training_history_is_too_short(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a configuration that cannot leave one embargo-safe forecast."""
    import volatility_regimes.walkforward.engine as cli

    dates = pd.bdate_range("2024-01-02", periods=20)
    prices = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0 + float(index) for index in range(20)],
        }
    )
    features = pd.DataFrame(
        {
            "trade_date": dates,
            "atm_iv_near": [0.20] * 20,
            "atm_iv_mid": [0.22] * 20,
            "term_slope": [0.02] * 20,
        }
    ).set_index("trade_date")

    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "walkforward")

    with pytest.raises(
        ValueError,
        match=(
            r"aligned_rows=10, maximum_safe_train_rows=4, "
            r"requested_min_train_size=10"
        ),
    ):
        cli.run_research(
            symbols=["SPX"],
            feature_sets=["atm_term"],
            horizons=[5],
            min_train_size=10,
            step_size=5,
            annualization=252,
            regime_min_k=2,
            regime_max_k=2,
            symbol_input_loader=lambda *_: (features, prices),
        )


def test_walkforward_cli_main_uses_walkforward_config_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI entrypoint should read runtime defaults and output_dir from config."""
    import volatility_regimes.walkforward.engine as cli

    dates = pd.bdate_range("2024-01-02", periods=40)
    prices = pd.DataFrame(
        {
            "date": dates,
            "close": [100.0 + float(index) for index in range(40)],
        }
    )
    feature_rows = pd.DataFrame(
        {
            "trade_date": dates,
            "atm_iv_near": [0.18 + 0.002 * index for index in range(40)],
            "atm_iv_mid": [0.20 + 0.002 * index for index in range(40)],
            "skew_near": [0.04] * 40,
            "skew_mid": [0.03] * 40,
            "butterfly_near": [0.01] * 40,
            "butterfly_mid": [0.01] * 40,
            "term_slope": [0.02] * 40,
        }
    ).set_index("trade_date")
    project_root = tmp_path / "project_root"
    project_root.mkdir(parents=True, exist_ok=True)

    root_config_text = """
[data]
symbols = ["SPX"]
start_date = "2020-01-02"
end_date = "2020-12-31"

[cache]
required_dir = "data/raw"
metadata_version = 1
require_full_date_coverage = true
mode = "offline_first"

[features]
delta_min = 0.05
delta_max = 0.95
near_term_dte_min = 15
near_term_dte_target = 30
near_term_dte_max = 45
mid_term_dte_min = 45
mid_term_dte_target = 90
mid_term_dte_max = 120
wing_delta = 0.25
min_strikes_per_side = 5
"""
    walkforward_config_text = """
[sample]
symbols = ["SPX"]
start_date = "2024-01-02"
end_date = "2024-02-29"

[evaluation]
horizons = [5]
feature_sets = ["atm_term"]
min_train_size = 20
step_size = 5
annualization = 252

[regime]
min_k = 2
max_k = 2
fixed_k_values = []
hmm_n_iter = 50
hmm_random_restarts = 2

[output]
output_dir = "outputs/custom_walkforward"
"""
    (project_root / "config.toml").write_text(root_config_text, encoding="utf-8")
    (project_root / "walkforward.toml").write_text(
        walkforward_config_text,
        encoding="utf-8",
    )

    observed_loader_call: dict[str, object] = {}

    def fake_load_symbol_inputs(
        symbol: str,
        project_config: dict[str, object],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Record the effective sample window passed into the loader."""
        observed_loader_call["symbol"] = symbol
        observed_loader_call["start_date"] = project_config["data"]["start_date"]
        observed_loader_call["end_date"] = project_config["data"]["end_date"]
        return feature_rows, prices

    monkeypatch.setattr(cli, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(cli, "_load_symbol_inputs", fake_load_symbol_inputs)

    cli.main()

    output_dir = project_root / "outputs" / "custom_walkforward"
    assert (output_dir / "forecast_panel.csv").exists()
    assert (output_dir / "metric_summary.csv").exists()
    assert (output_dir / "research_summary.md").exists()
    assert observed_loader_call["symbol"] == "SPX"
    assert observed_loader_call["start_date"] == "2024-01-02"
    assert observed_loader_call["end_date"] == "2024-02-29"
