import pandas as pd
from .features import NUMERIC

def population_stability_index(reference: pd.Series, current: pd.Series, bins=10) -> float:
    edges = reference.quantile([i/bins for i in range(bins+1)]).drop_duplicates().values
    ref = pd.cut(reference, edges, include_lowest=True).value_counts(normalize=True, sort=False).clip(.0001)
    cur = pd.cut(current, edges, include_lowest=True).value_counts(normalize=True, sort=False).reindex(ref.index, fill_value=.0001).clip(.0001)
    return float(((cur-ref) * (cur/ref).apply(__import__('numpy').log)).sum())

def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    return {col: round(population_stability_index(reference[col], current[col]), 4) for col in NUMERIC}
