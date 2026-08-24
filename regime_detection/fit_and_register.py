"""Fit a JM configuration through a given cutoff and register it as a model_version
with its full point-in-time-honest label set - the concrete "a recalibration event
happens" operation from the recalibration design (§2, approved).

CLI, so this doubles as both the demo/validation tool and the real utility a
scheduled or drift-triggered recalibration would eventually call.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

from data_agent.db import get_connection
from regime_detection.features import FEATURE_COLUMNS, build_feature_matrix
from regime_detection.jump_model_fit import prepare_fit_data
from regime_detection.regime_db import ensure_schema, save_model_version, save_regime_labels


def fit_and_register(conn, cutoff: date | None, k: int, jump_penalty: float, notes: str | None = None) -> int:
    df = build_feature_matrix(conn)
    X_raw_full, ret_ser_full = prepare_fit_data(df)
    cutoff_ts = None if cutoff is None else pd.Timestamp(cutoff)
    X_raw = X_raw_full if cutoff_ts is None else X_raw_full.loc[:cutoff_ts]
    ret_ser = ret_ser_full if cutoff_ts is None else ret_ser_full.loc[:cutoff_ts]

    clipper = DataClipperStd(mul=3.0)
    scaler = StandardScalerPD()
    X = scaler.fit_transform(clipper.fit_transform(X_raw))

    jm = JumpModel(n_components=k, jump_penalty=jump_penalty, cont=False, random_state=0)
    jm.fit(X, ret_ser=ret_ser, sort_by="cumret")
    labels = pd.Series(jm.labels_, index=X_raw.index)

    version_id = save_model_version(
        conn, model_kind="jm", k=k, jump_penalty=jump_penalty,
        fit_start_date=X_raw.index[0].date(), fit_end_date=X_raw.index[-1].date(),
        clipper=clipper, scaler=scaler, feature_columns=FEATURE_COLUMNS, notes=notes,
    )
    n_labels = save_regime_labels(conn, version_id, labels)
    return version_id, n_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and register a JM model_version")
    parser.add_argument("--cutoff", type=date.fromisoformat, default=None,
                         help="Fit through this date (default: all available data)")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--jump-penalty", type=float, default=50.0)
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()

    conn = get_connection()
    try:
        ensure_schema(conn)
        version_id, n_labels = fit_and_register(conn, args.cutoff, args.k, args.jump_penalty, args.notes)
    finally:
        conn.close()

    print(f"Registered model_version {version_id}: k={args.k}, jump_penalty={args.jump_penalty}, "
          f"cutoff={args.cutoff or 'full data'}, {n_labels} labels saved")


if __name__ == "__main__":
    main()
