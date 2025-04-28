from typing import Tuple, List, Optional

import pandas as pd

class RCTAnalyzer:
    def __init__(self,
                 data: pd.DataFrame,
                 pre_experiment_covariates: Optional[List[str]] = None,
                 intra_experiment_metrics: List[str],
                 treatment_col: str = 'treatment',
                 customer_segments: Optional[List[str]] = None,
                 mode: str = "no_enhancement"
                 ):
        """initialises RCTAnalyzer with data.
        
        - data: customer level df containing all columns defined below 
        - pre_experiment_covariates: list containing PE columns used for CUPED.
        - intra_experiment_metrics: list containing success metrics measured during exp.
        - treatment_col: column denoting treatment or control. columns should be binary
        - mode: ( None, "cuped", "gboost_cuped"). If not None, pre_experiment_covariates must not be None.
        """

    # validation (skip)
    assert mode in ["no_enhancement", "cuped", "gboost_cuped"], "Invalid mode"

        self.data = data
        self.pre_experiment_covariates = pre_experiment_covariates
        self.intra_experiment_metrics = intra_experiment_metrics
        self.treatment_col = treatment_col
        self.customer_segments = customer_segments
    
    def calculate_uplift():
        return 1
    
    def save_report():
        return 1