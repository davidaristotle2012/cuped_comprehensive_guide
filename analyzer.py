from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from calculators import Augmentor, Stats


class ABTestAnalyzer:

    def __init__(self):
        return

    def analyze(
        data: pd.DataFrame,
        success_metrics: List[str] = None,
        group_col: str = "group",
        customer_segments: Optional[List[str]] = None,
        mode: str = "cuped",
        X: Optional[List[str]] = None,
    ):

        if mode == "cuped":
            transformed_data = Augmentor.cuped_transform(
                data, X, success_metrics, group_col
            )

        return Stats.obtain_group_stats(
            data, success_metrics, group_col, transformed_data
        )

    def summary():
        """show results like R"""
        return 1
