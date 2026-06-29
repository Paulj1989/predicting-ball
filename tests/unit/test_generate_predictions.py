import sys
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parent.parent.parent
for _path in [str(_root), str(_root / "scripts" / "modeling")]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_predictions import (  # noqa: E402  # type: ignore[unresolved-import]
    create_matches_dataframe,
)


def test_create_matches_dataframe_none_inputs_returns_empty():
    result = create_matches_dataframe(None, None, None)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_create_matches_dataframe_empty_predictions_returns_empty():
    result = create_matches_dataframe(pd.DataFrame(), pd.DataFrame(), None)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
