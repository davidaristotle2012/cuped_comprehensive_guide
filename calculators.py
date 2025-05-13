import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats


class Augmentor:

    def _cross_fitter():
        """
        performs cross_fitting algorithm to debias CUPED.
        """

    @staticmethod
    def cuped_transform(data, X, success_metrics, group_col):
        """
        Given intra and pre exp data,
        returns cuped transformed success metrics
        Used when mode = cuped

        input:
            - data: df containing X, success_metrics, and group_col
            - X: pre exp col
            - success_metrics: metrics col
        output:
            - transformed_data with success_metric
        """

        transformed_data = data[[group_col]]

        for success_metric in success_metrics:

            ### Obtain CUPED parameters ###
            # n: number of individuals, k: number of pre_exp dimensions

            pre_exp_df = data[X]  # n x k
            pre_exp_df_mean = np.mean(pre_exp_df)  # k x 1
            exp_df = data[success_metric]  # n x 1
            theta_model = LinearRegression(fit_intercept=False)
            theta_model.fit(pre_exp_df, exp_df)
            theta = theta_model.coef_.flatten()  # k x 1

            ### Compute CUPED transform ###

            # Center the pre-experiment covariates
            pre_exp_df_centered = pre_exp_df - pre_exp_df_mean  # n x k

            # Compute the CUPED adjustment
            cuped_adjustment = pre_exp_df_centered @ theta  # n x 1
            exp_cuped = exp_df.sub(cuped_adjustment, axis=0)  # n x 1

            transformed_data[success_metric] = exp_cuped

        return transformed_data

    @staticmethod
    def gboost_cuped_transform(clf):
        """
        Given intra and pre exp data,
        Calculates cuped-transformed success metrics and variance
        for control and test group.
        Used when mode = gboost-cuped
        """


class Stats:

    @staticmethod
    def obtain_group_stats(data, success_metrics, group_col, transformed_data):

        overall_results = {}

        for success_metric in success_metrics:
            grouped = transformed_data.groupby(group_col)[success_metric]
            means = grouped.mean().to_dict()
            vars = grouped.var().to_dict()
            uplift_abs = means["treatment"] - means["control"]
            uplift_pct = (
                uplift_abs * 100 / means["control"] if means["control"] > 0 else None
            )

            control_exp = transformed_data.loc[
                transformed_data[group_col] == "control", success_metric
            ]
            treatment_exp = transformed_data.loc[
                transformed_data[group_col] == "treatment", success_metric
            ]
            ttest = stats.ttest_ind(control_exp, treatment_exp, equal_var=False)

            ci95 = ttest.confidence_interval(0.95)
            ci90 = ttest.confidence_interval(0.90)
            ci80 = ttest.confidence_interval(0.80)

            before_var = data[success_metric].var()
            after_var = transformed_data[success_metric].var()
            var_reduction_ratio = (
                (before_var - after_var) / before_var if before_var > 0 else None
            )

            overall_results[success_metric] = {
                "Mean": {"treatment": means["treatment"], "control": means["control"]},
                "Variance": {
                    "treatment": vars["treatment"],
                    "control": vars["control"],
                },
                "Uplift": {"pcnt": uplift_pct, "absolute": uplift_abs},
                "Significance": {
                    "p-value": ttest.pvalue,
                    "ci_95": ci95,
                    "ci_90": ci90,
                    "ci_80": ci80,
                },
                "Variance reduction": {
                    "before": before_var,
                    "after": after_var,
                    "reduction_ratio": var_reduction_ratio,
                },
                "Skewness": {},
            }

        return overall_results
