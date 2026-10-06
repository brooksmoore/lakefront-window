# Holdout results

Label: Oak Street lakefront sensor hour with air temp 10-27 C, hourly max gust < 9 m/s, and no rain/snow detected; hours 07:00-19:00 local only.

Train 29,387 hours (2015-05-26 to 2023-12-31); test 11,397 hours (2024-01-01 to 2026-10-05). Good-hour base rate: train 48.0%, test 47.4%.

Inputs are actual Midway airport observations standing in for a forecast; real forecast error is not included.

| Method | Accuracy | Balanced acc. | F1 (good) | Brier | AUC | Best-window hit rate |
|---|---|---|---|---|---|---|
| naive_forecast_as_is | 0.846 | 0.842 | 0.827 | 0.154 | n/a | 0.510 |
| naive_plus_lake_temp_offset | 0.863 | 0.861 | 0.849 | 0.137 | n/a | 0.520 |
| climatology_month_hour | 0.760 | 0.760 | 0.752 | 0.171 | 0.825 | 0.466 |
| logistic_regression_all_train | 0.783 | 0.781 | 0.764 | 0.156 | 0.852 | 0.463 |
| gradient_boosting_all_train | 0.916 | 0.915 | 0.911 | 0.062 | 0.975 | 0.547 |
| tabpfn_v2_3000_rows | 0.914 | 0.913 | 0.907 | 0.067 | 0.973 | 0.556 |

Window check over 629 test days with all 13 hours: a day had at least one fully good 2-hour window 59.9% of the time (ceiling); a random window was good 41.5% of the time. On days that had a good window, the picked window was good: naive_forecast_as_is 85.1%, naive_plus_lake_temp_offset 86.7%, climatology_month_hour 77.7%, logistic_regression_all_train 77.2%, gradient_boosting_all_train 91.2%, tabpfn_v2_3000_rows 92.8%.

Per test day, TabPFN had fewer wrong hours than the as-is app reading on 274 days, more on 89 days, tied on 563.

## TabPFN calibration on the holdout

| Predicted bin | Hours | Mean predicted | Observed good |
|---|---|---|---|
| (-0.001, 0.1] | 4192 | 0.01 | 0.01 |
| (0.1, 0.3] | 1079 | 0.20 | 0.19 |
| (0.3, 0.5] | 921 | 0.39 | 0.38 |
| (0.5, 0.7] | 651 | 0.60 | 0.71 |
| (0.7, 0.9] | 2733 | 0.83 | 0.93 |
| (0.9, 1.0] | 1821 | 0.92 | 0.99 |

## Synthetic stress test (noise on inputs: temp sd 2 C, wind sd 1.5 m/s, RH sd 8)

| Method | Accuracy | Brier |
|---|---|---|
| naive_forecast_as_is | 0.817 | 0.183 |
| naive_plus_lake_temp_offset | 0.834 | 0.166 |
| gradient_boosting_all_train | 0.888 | 0.084 |
| tabpfn_v2_3000_rows | 0.885 | 0.085 |

TabPFN-v2 context: 3000 randomly sampled training hours; CPU time to score the holdout: 53.9 s.
