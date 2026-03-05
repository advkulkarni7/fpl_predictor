from __future__ import annotations

import json
import pickle
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fpl_phase1_model import (
    add_price_predictions,
    build_current_features,
    build_player_history_df,
    compute_expected_pts,
    fetch_bootstrap,
    fetch_current_gw,
    fetch_fixtures,
    predict_component_pts,
    train_component_models,
    train_models,
    train_price_model,
)
from fpl_phase2_fixtures import (
    FIXTURE_LOOKAHEAD,
    build_cs_probability_map,
    build_custom_difficulty,
    build_fixture_run,
    build_opponent_scoring_map,
    build_player_fixture_scores,
    build_team_form,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
PREDICTIONS_CSV = DATA_DIR / "fpl_predictions.csv"
FIXTURE_SCORES_CSV = DATA_DIR / "player_fixture_scores.csv"
PIPELINE_META_JSON = DATA_DIR / "pipeline_meta.json"
MODEL_METRICS_CSV = DATA_DIR / "model_metrics.csv"
MODEL_PKL = DATA_DIR / "fpl_model.pkl"
PIPELINE_VERSION = "precompute-v1"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(val):
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _build_model_metrics_df(models: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for pos, info in (models or {}).items():
        if not isinstance(info, dict):
            continue
        model_obj = info.get("model")
        rows.append(
            {
                "position": str(pos),
                "rmse": _safe_float(info.get("rmse")),
                "r2": _safe_float(info.get("r2")),
                "model_name": type(model_obj).__name__ if model_obj is not None else None,
                "n_train_rows": _safe_float(info.get("n_train_rows")),
            }
        )
    return pd.DataFrame(rows)


def run_pipeline() -> dict:
    print(f"[{_iso_now()}] Pipeline start")

    bootstrap = fetch_bootstrap()
    fixtures_df = fetch_fixtures()
    current_gw = fetch_current_gw(bootstrap)
    print(f"Current GW: {current_gw}")

    history_df = build_player_history_df(bootstrap, refresh=False)
    print(f"History rows: {len(history_df)}")

    models = train_models(history_df)
    all_player_ids = [int(p.get("id", 0)) for p in bootstrap.get("elements", []) if p.get("id")]
    pred_df = build_current_features(
        bootstrap=bootstrap,
        fixtures_df=fixtures_df,
        history_df=history_df,
        models=models,
        current_gw=current_gw,
        my_player_ids=all_player_ids,
    )

    # Optional enrichments should never fail the precompute job.
    try:
        component_models = train_component_models(history_df)
        pred_df = predict_component_pts(component_models, pred_df)
    except Exception:
        pass
    try:
        pred_df = compute_expected_pts(pred_df)
    except Exception:
        pass
    try:
        price_model = train_price_model(history_df)
        pred_df = add_price_predictions(price_model, pred_df)
    except Exception:
        pass

    custom_diff = build_custom_difficulty(history_df, bootstrap)
    team_form_map = build_team_form(history_df, bootstrap)
    opp_scoring_map = build_opponent_scoring_map(history_df)
    try:
        cs_prob_map = build_cs_probability_map(history_df)
    except Exception:
        cs_prob_map = {}

    fixture_run_df = build_fixture_run(
        bootstrap=bootstrap,
        fixtures_df=fixtures_df,
        current_gw=current_gw,
        custom_difficulty=custom_diff,
        gw_lookahead=FIXTURE_LOOKAHEAD,
    )
    try:
        enriched_df = build_player_fixture_scores(
            pred_df=pred_df,
            fixture_run_df=fixture_run_df,
            current_gw=current_gw,
            team_form_map=team_form_map,
            opponent_scoring_map=opp_scoring_map,
            gw_lookahead=FIXTURE_LOOKAHEAD,
            cs_probability_map=cs_prob_map,
        )
    except TypeError:
        enriched_df = build_player_fixture_scores(
            pred_df=pred_df,
            fixture_run_df=fixture_run_df,
            current_gw=current_gw,
            team_form_map=team_form_map,
            opponent_scoring_map=opp_scoring_map,
            gw_lookahead=FIXTURE_LOOKAHEAD,
        )

    print(f"Predictions rows: {len(pred_df)}")
    print(f"Fixture scores rows: {len(enriched_df)}")
    return {
        "bootstrap": bootstrap,
        "current_gw": int(current_gw),
        "history_df": history_df,
        "models": models,
        "pred_df": pred_df,
        "enriched_df": enriched_df,
    }


def save_outputs(pipeline: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pred_df: pd.DataFrame = pipeline["pred_df"]
    enriched_df: pd.DataFrame = pipeline["enriched_df"]
    models: dict = pipeline["models"]

    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    enriched_df.to_csv(FIXTURE_SCORES_CSV, index=False)
    _build_model_metrics_df(models).to_csv(MODEL_METRICS_CSV, index=False)

    # Keep legacy compatibility for code that expects a pickled model artifact.
    with MODEL_PKL.open("wb") as f:
        pickle.dump(
            {
                "generated_at": _iso_now(),
                "current_gw": pipeline["current_gw"],
                "models": models,
            },
            f,
        )

    meta = {
        "generated_at": _iso_now(),
        "pipeline_version": PIPELINE_VERSION,
        "current_gw": pipeline["current_gw"],
        "row_count_predictions": int(len(pred_df)),
        "row_count_fixture_scores": int(len(enriched_df)),
        "row_count_history": int(len(pipeline["history_df"])),
        "files": {
            "fpl_predictions_csv": str(PREDICTIONS_CSV.as_posix()),
            "player_fixture_scores_csv": str(FIXTURE_SCORES_CSV.as_posix()),
            "model_metrics_csv": str(MODEL_METRICS_CSV.as_posix()),
            "pipeline_meta_json": str(PIPELINE_META_JSON.as_posix()),
            "fpl_model_pkl": str(MODEL_PKL.as_posix()),
        },
    }
    PIPELINE_META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{_iso_now()}] Wrote outputs to {DATA_DIR}")


def main() -> int:
    try:
        pipeline = run_pipeline()
        save_outputs(pipeline)
        print(f"[{_iso_now()}] Precompute complete")
        return 0
    except Exception as exc:
        print(f"[{_iso_now()}] Precompute failed: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
