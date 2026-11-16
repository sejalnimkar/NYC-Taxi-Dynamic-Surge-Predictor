# Holdout results

Source: `duckdb`. Train 2026-01-01 00:00:00 to 2026-01-25 18:00:00, validation to 2026-01-31 23:00:00, holdout 2026-02-01 00:00:00 to 2026-02-28 23:00:00 (178,080 zone-hours).

| Horizon | Method | Holdout RMSE | Holdout MAE | Holdout WAPE |
|---|---|---|---|---|
| 1h | baseline: persistence | 37.28 | 21.63 | 19.4% |
| 1h | baseline: same_hour_yesterday | 64.91 | 30.76 | 27.6% |
| 1h | baseline: same_hour_last_week | 50.07 | 24.62 | 22.1% |
| 1h | lightgbm: recent | 26.15 | 14.70 | 13.2% |
| 1h | lightgbm: recent_plus_seasonal | 24.99 | 13.86 | 12.4% |
| 2h | baseline: persistence | 60.08 | 34.65 | 31.1% |
| 2h | baseline: same_hour_yesterday | 64.91 | 30.76 | 27.6% |
| 2h | baseline: same_hour_last_week | 50.07 | 24.62 | 22.1% |
| 2h | lightgbm: recent | 33.99 | 18.77 | 16.8% |
| 2h | lightgbm: recent_plus_seasonal | 32.00 | 17.25 | 15.5% |
| 3h | baseline: persistence | 78.41 | 45.58 | 40.9% |
| 3h | baseline: same_hour_yesterday | 64.91 | 30.76 | 27.6% |
| 3h | baseline: same_hour_last_week | 50.07 | 24.62 | 22.1% |
| 3h | lightgbm: recent | 39.16 | 21.44 | 19.2% |
| 3h | lightgbm: recent_plus_seasonal | 36.92 | 19.12 | 17.1% |

## LightGBM (recent + seasonal) by borough, holdout

| Horizon | Borough | Zone-hours | Mean trips/zone-hour | RMSE | MAE | WAPE |
|---|---|---|---|---|---|---|
| 1h | Manhattan | 46,368 | 149.1 | 28.28 | 17.52 | 11.7% |
| 1h | Queens | 46,368 | 93.7 | 27.48 | 12.68 | 13.5% |
| 1h | Brooklyn | 40,992 | 137.0 | 27.55 | 16.77 | 12.2% |
| 1h | Bronx | 28,896 | 92.1 | 15.92 | 10.92 | 11.9% |
| 1h | Staten Island | 13,440 | 24.7 | 6.94 | 4.71 | 19.1% |
| 1h | EWR | 672 | 0.0 | 0.53 | 0.35 | n/a |
| 1h | N/A | 672 | 1.5 | 1.39 | 1.05 | 71.0% |
| 1h | Unknown | 672 | 0.0 | 0.53 | 0.36 | n/a |
| 2h | Manhattan | 46,368 | 149.1 | 36.60 | 22.33 | 15.0% |
| 2h | Queens | 46,368 | 93.7 | 34.33 | 15.43 | 16.5% |
| 2h | Brooklyn | 40,992 | 137.0 | 35.77 | 21.20 | 15.5% |
| 2h | Bronx | 28,896 | 92.1 | 20.65 | 13.20 | 14.3% |
| 2h | Staten Island | 13,440 | 24.7 | 7.91 | 5.17 | 20.9% |
| 2h | EWR | 672 | 0.0 | 0.70 | 0.49 | n/a |
| 2h | N/A | 672 | 1.5 | 1.51 | 1.15 | 77.9% |
| 2h | Unknown | 672 | 0.0 | 0.70 | 0.49 | n/a |
| 3h | Manhattan | 46,368 | 149.1 | 41.82 | 25.01 | 16.8% |
| 3h | Queens | 46,368 | 93.7 | 40.64 | 17.06 | 18.2% |
| 3h | Brooklyn | 40,992 | 137.0 | 40.90 | 23.62 | 17.2% |
| 3h | Bronx | 28,896 | 92.1 | 23.10 | 14.21 | 15.4% |
| 3h | Staten Island | 13,440 | 24.7 | 8.76 | 5.53 | 22.4% |
| 3h | EWR | 672 | 0.0 | 0.90 | 0.60 | n/a |
| 3h | N/A | 672 | 1.5 | 1.63 | 1.22 | 82.3% |
| 3h | Unknown | 672 | 0.0 | 0.90 | 0.60 | n/a |
