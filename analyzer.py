from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from calculators import Augmentor, Stats


class ABTestAnalyzer:

    def __init__(self):
        self.raw_results = {}
        return

    def analyze(
        self,
        data: pd.DataFrame,
        success_metrics: List[str],
        group_col: str = "group",
        customer_segments: Optional[List[str]] = None,
        mode: str = "cuped",
        X: Optional[List[str]] = None,
    ):

        if mode == "cuped":
            transformed_data = Augmentor.cuped_transform(
                data, X, success_metrics, group_col
            )

        elif mode == "abtest":
            transformed_data = data

        group_stats = Stats.obtain_group_stats(
            data, success_metrics, group_col, transformed_data
        )
        # Add results of analysis to all results.
        analysis_number = len(self.raw_results) + 1
        self.raw_results.append({f"analysis_{analysis_number}": group_stats})

    def summary(self):
        """show results like R"""
        for analysis in self.raw_results:
            print(analysis)
        return 1
