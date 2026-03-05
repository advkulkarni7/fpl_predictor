-- FPL AI Assistant - MVP DB Schema
-- Postgres / Neon
-- Idempotent for both fresh installs and older deployments.

-- 1) Snapshot metadata table (source of truth for each pipeline run)
CREATE TABLE IF NOT EXISTS model_snapshots (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL CHECK (status IN ('building', 'ready', 'failed')),
    season TEXT,
    current_gw INTEGER,
    pipeline_version TEXT,
    model_version TEXT,
    feature_version TEXT,
    build_duration_sec INTEGER,
    error_message TEXT,
    row_count_players INTEGER,
    row_count_player_fixture INTEGER,
    row_count_team_fixture INTEGER
);

-- Schema evolution for existing deployments
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'building';
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS season TEXT;
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS current_gw INTEGER;
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS pipeline_version TEXT;
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS model_version TEXT;
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS feature_version TEXT;
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS build_duration_sec INTEGER;
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS row_count_players INTEGER;
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS row_count_player_fixture INTEGER;
ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS row_count_team_fixture INTEGER;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'model_snapshots_status_check'
    ) THEN
        ALTER TABLE model_snapshots
            ADD CONSTRAINT model_snapshots_status_check
            CHECK (status IN ('building', 'ready', 'failed'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_model_snapshots_status_created
    ON model_snapshots (status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_snapshots_created
    ON model_snapshots (created_at DESC);


-- 2) Main player predictions snapshot table
CREATE TABLE IF NOT EXISTS player_predictions_snapshot (
    snapshot_id BIGINT NOT NULL,
    player_id INTEGER NOT NULL,
    player_name TEXT,
    team_id INTEGER,
    team_name TEXT,
    position TEXT,
    price NUMERIC(6, 1),

    predicted_pts NUMERIC,
    expected_pts NUMERIC,
    pts_low NUMERIC,
    pts_high NUMERIC,
    captain_ev NUMERIC,
    p_plays_full NUMERIC,
    predicted_price_change NUMERIC,

    combined_score NUMERIC,
    value_score NUMERIC,
    avg_difficulty NUMERIC,
    fixture_run_label TEXT,
    blank_gws INTEGER,
    double_gws INTEGER,
    is_blank_next_gw BOOLEAN,
    momentum_score NUMERIC,

    raw_json JSONB,

    PRIMARY KEY (snapshot_id, player_id),
    CONSTRAINT fk_player_predictions_snapshot_snapshot
        FOREIGN KEY (snapshot_id)
        REFERENCES model_snapshots (id)
        ON DELETE CASCADE
);

-- Schema evolution for existing deployments
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS player_name TEXT;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS team_id INTEGER;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS team_name TEXT;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS position TEXT;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS price NUMERIC(6, 1);
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS predicted_pts NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS expected_pts NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS pts_low NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS pts_high NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS captain_ev NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS p_plays_full NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS predicted_price_change NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS combined_score NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS value_score NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS avg_difficulty NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS fixture_run_label TEXT;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS blank_gws INTEGER;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS double_gws INTEGER;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS is_blank_next_gw BOOLEAN;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS momentum_score NUMERIC;
ALTER TABLE player_predictions_snapshot ADD COLUMN IF NOT EXISTS raw_json JSONB;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_player_predictions_snapshot_snapshot'
    ) THEN
        ALTER TABLE player_predictions_snapshot
            ADD CONSTRAINT fk_player_predictions_snapshot_snapshot
            FOREIGN KEY (snapshot_id)
            REFERENCES model_snapshots (id)
            ON DELETE CASCADE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_pred_snap_position
    ON player_predictions_snapshot (snapshot_id, position);

CREATE INDEX IF NOT EXISTS idx_pred_snap_team
    ON player_predictions_snapshot (snapshot_id, team_id);

CREATE INDEX IF NOT EXISTS idx_pred_snap_expected_pts
    ON player_predictions_snapshot (snapshot_id, expected_pts DESC);

CREATE INDEX IF NOT EXISTS idx_pred_snap_predicted_pts
    ON player_predictions_snapshot (snapshot_id, predicted_pts DESC);

CREATE INDEX IF NOT EXISTS idx_pred_snap_combined
    ON player_predictions_snapshot (snapshot_id, combined_score DESC);


-- 3) Team fixture planner snapshot table (for fixture planner heatmaps/runs)
CREATE TABLE IF NOT EXISTS team_fixture_run_snapshot (
    snapshot_id BIGINT NOT NULL,
    team_id INTEGER NOT NULL,
    gw INTEGER NOT NULL,

    opponent TEXT,
    opponent_team_id INTEGER,
    difficulty NUMERIC,
    is_home BOOLEAN,
    is_blank BOOLEAN,
    is_double BOOLEAN,

    PRIMARY KEY (snapshot_id, team_id, gw),
    CONSTRAINT fk_team_fixture_run_snapshot_snapshot
        FOREIGN KEY (snapshot_id)
        REFERENCES model_snapshots (id)
        ON DELETE CASCADE
);

-- Schema evolution for existing deployments
ALTER TABLE team_fixture_run_snapshot ADD COLUMN IF NOT EXISTS opponent TEXT;
ALTER TABLE team_fixture_run_snapshot ADD COLUMN IF NOT EXISTS opponent_team_id INTEGER;
ALTER TABLE team_fixture_run_snapshot ADD COLUMN IF NOT EXISTS difficulty NUMERIC;
ALTER TABLE team_fixture_run_snapshot ADD COLUMN IF NOT EXISTS is_home BOOLEAN;
ALTER TABLE team_fixture_run_snapshot ADD COLUMN IF NOT EXISTS is_blank BOOLEAN;
ALTER TABLE team_fixture_run_snapshot ADD COLUMN IF NOT EXISTS is_double BOOLEAN;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_team_fixture_run_snapshot_snapshot'
    ) THEN
        ALTER TABLE team_fixture_run_snapshot
            ADD CONSTRAINT fk_team_fixture_run_snapshot_snapshot
            FOREIGN KEY (snapshot_id)
            REFERENCES model_snapshots (id)
            ON DELETE CASCADE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_team_fixture_snapshot_gw
    ON team_fixture_run_snapshot (snapshot_id, gw);


-- 4) Player fixture features snapshot table (normalized per-player per-GW fixture detail)
CREATE TABLE IF NOT EXISTS player_fixture_features_snapshot (
    snapshot_id BIGINT NOT NULL,
    player_id INTEGER NOT NULL,
    gw INTEGER NOT NULL,

    opponent TEXT,
    opponent_team_id INTEGER,
    difficulty NUMERIC,
    is_home BOOLEAN,
    is_blank BOOLEAN,
    is_double BOOLEAN,
    fixture_pts_proxy NUMERIC,

    PRIMARY KEY (snapshot_id, player_id, gw),
    CONSTRAINT fk_player_fixture_features_snapshot_snapshot
        FOREIGN KEY (snapshot_id)
        REFERENCES model_snapshots (id)
        ON DELETE CASCADE
);

-- Schema evolution for existing deployments
ALTER TABLE player_fixture_features_snapshot ADD COLUMN IF NOT EXISTS opponent TEXT;
ALTER TABLE player_fixture_features_snapshot ADD COLUMN IF NOT EXISTS opponent_team_id INTEGER;
ALTER TABLE player_fixture_features_snapshot ADD COLUMN IF NOT EXISTS difficulty NUMERIC;
ALTER TABLE player_fixture_features_snapshot ADD COLUMN IF NOT EXISTS is_home BOOLEAN;
ALTER TABLE player_fixture_features_snapshot ADD COLUMN IF NOT EXISTS is_blank BOOLEAN;
ALTER TABLE player_fixture_features_snapshot ADD COLUMN IF NOT EXISTS is_double BOOLEAN;
ALTER TABLE player_fixture_features_snapshot ADD COLUMN IF NOT EXISTS fixture_pts_proxy NUMERIC;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_player_fixture_features_snapshot_snapshot'
    ) THEN
        ALTER TABLE player_fixture_features_snapshot
            ADD CONSTRAINT fk_player_fixture_features_snapshot_snapshot
            FOREIGN KEY (snapshot_id)
            REFERENCES model_snapshots (id)
            ON DELETE CASCADE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_player_fixture_snapshot_player
    ON player_fixture_features_snapshot (snapshot_id, player_id);

CREATE INDEX IF NOT EXISTS idx_player_fixture_snapshot_gw
    ON player_fixture_features_snapshot (snapshot_id, gw);


-- 5) Model diagnostics by position
CREATE TABLE IF NOT EXISTS model_metrics_snapshot (
    snapshot_id BIGINT NOT NULL,
    position TEXT NOT NULL,

    rmse NUMERIC,
    r2 NUMERIC,
    model_name TEXT,
    n_train_rows INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (snapshot_id, position),
    CONSTRAINT fk_model_metrics_snapshot_snapshot
        FOREIGN KEY (snapshot_id)
        REFERENCES model_snapshots (id)
        ON DELETE CASCADE
);

-- Schema evolution for existing deployments
ALTER TABLE model_metrics_snapshot ADD COLUMN IF NOT EXISTS rmse NUMERIC;
ALTER TABLE model_metrics_snapshot ADD COLUMN IF NOT EXISTS r2 NUMERIC;
ALTER TABLE model_metrics_snapshot ADD COLUMN IF NOT EXISTS model_name TEXT;
ALTER TABLE model_metrics_snapshot ADD COLUMN IF NOT EXISTS n_train_rows INTEGER;
ALTER TABLE model_metrics_snapshot ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_model_metrics_snapshot_snapshot'
    ) THEN
        ALTER TABLE model_metrics_snapshot
            ADD CONSTRAINT fk_model_metrics_snapshot_snapshot
            FOREIGN KEY (snapshot_id)
            REFERENCES model_snapshots (id)
            ON DELETE CASCADE;
    END IF;
END $$;


-- 6) Team-level FPL API cache (reduces repeated live calls on Streamlit)
CREATE TABLE IF NOT EXISTS fpl_team_cache (
    team_id INTEGER NOT NULL,
    event_gw INTEGER NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    picks_json JSONB,
    transfer_info_json JSONB,

    status TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'failed')),
    error_message TEXT,

    PRIMARY KEY (team_id, event_gw)
);

-- Schema evolution for existing deployments
ALTER TABLE fpl_team_cache ADD COLUMN IF NOT EXISTS fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE fpl_team_cache ADD COLUMN IF NOT EXISTS picks_json JSONB;
ALTER TABLE fpl_team_cache ADD COLUMN IF NOT EXISTS transfer_info_json JSONB;
ALTER TABLE fpl_team_cache ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ok';
ALTER TABLE fpl_team_cache ADD COLUMN IF NOT EXISTS error_message TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fpl_team_cache_status_check'
    ) THEN
        ALTER TABLE fpl_team_cache
            ADD CONSTRAINT fpl_team_cache_status_check
            CHECK (status IN ('ok', 'failed'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_team_cache_fetched_at
    ON fpl_team_cache (fetched_at DESC);
