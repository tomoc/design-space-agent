from __future__ import annotations

import pandas as pd


def duplicated_feature_target_rows(df: pd.DataFrame, columns: list[str]) -> int:
    """Return duplicated row count over selected columns.

    This placeholder keeps leakage checks explicit without claiming a complete
    leakage detector in the MVP.
    """
    if not columns:
        return 0
    return int(df.duplicated(subset=columns).sum())
