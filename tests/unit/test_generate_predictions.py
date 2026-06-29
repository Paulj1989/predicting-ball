import pandas as pd

from src.io.output_builders import create_matches_dataframe


def test_create_matches_dataframe_none_inputs_returns_empty():
    result = create_matches_dataframe(None, None, None)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_create_matches_dataframe_empty_predictions_returns_empty():
    result = create_matches_dataframe(pd.DataFrame(), pd.DataFrame(), None)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
