from hp_ml.auto_tune_baseline import (
    StrategyConfig,
    cumulative_return,
    default_strategy_configs,
    simulate_strategy,
    strategy_metrics,
    walkforward_boundaries,
)


def test_cumulative_return_compounds_daily_values():
    assert abs(cumulative_return([0.10, -0.05]) - 0.045) < 1e-12


def test_walkforward_boundaries_keep_exact_start_and_end():
    boundaries = walkforward_boundaries("2024-11-08", "2025-01-03")
    assert str(boundaries[0].date()) == "2024-11-08"
    assert str(boundaries[-1].date()) == "2025-01-03"
    assert "2024-11-30" in {str(item.date()) for item in boundaries}
    assert "2024-12-31" in {str(item.date()) for item in boundaries}


def test_strategy_simulation_reports_baseline_outperformance():
    import pandas as pd

    signals = pd.DataFrame(
        [
            {"date": "2024-01-01", "asset_code": "A", "daily_return": 0.01, "price": 2.0, "sma": 1.0, "sentiment_z": 0.0, "prob_up": 0.90},
            {"date": "2024-01-01", "asset_code": "B", "daily_return": -0.01, "price": 2.0, "sma": 1.0, "sentiment_z": 0.0, "prob_up": 0.10},
            {"date": "2024-01-02", "asset_code": "A", "daily_return": 0.02, "price": 2.0, "sma": 1.0, "sentiment_z": 0.0, "prob_up": 0.90},
            {"date": "2024-01-02", "asset_code": "B", "daily_return": -0.02, "price": 2.0, "sma": 1.0, "sentiment_z": 0.0, "prob_up": 0.10},
            {"date": "2024-01-03", "asset_code": "A", "daily_return": 0.03, "price": 2.0, "sma": 1.0, "sentiment_z": 0.0, "prob_up": 0.90},
            {"date": "2024-01-03", "asset_code": "B", "daily_return": -0.03, "price": 2.0, "sma": 1.0, "sentiment_z": 0.0, "prob_up": 0.10},
        ]
    )
    daily = simulate_strategy(
        signals,
        ["A", "B"],
        StrategyConfig(mode="discrete", top_k=1, prob_threshold=0.5, sent_break_threshold=1.0, fee_rate=0.0),
    )
    metrics = strategy_metrics(daily)
    assert metrics["strategy_cumulative_return"] > metrics["benchmark_cumulative_return"]
    assert metrics["success"] is True


def test_default_strategy_grid_contains_fast_success_candidate():
    labels = {config.label for config in default_strategy_configs(fee_rate=0.001)}
    assert "ema_ab0.6_as0.05_top1_p0.49_sent1.6_trend_fee0.001" in labels
