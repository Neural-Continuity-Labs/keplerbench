# Local dynamical law, end to end

Initialization window up to t = 20; failure threshold 60 arcmin; divergence from two initialization windows ending at the same instant.

| World | Submission | State dim | Median error by horizon bucket (arcmin) | Failure horizon | Window divergence, median / max (arcmin) |
|---|---|---|---|---|---|
| inverse-square-0300 (T = 17.3) | dynamical | 8 | 0-5: 0.7, 5-10: 0.8, 10-20: 0.8, 20-40: 1.1, 40-60: 1.5 | never | 0.0 / 0.1 |
| inverse-square-0300 (T = 17.3) | trajectory_clock | 35 | 0-5: 4104.6, 5-10: 5224.4, 10-20: 3756.9, 20-40: 8347.7, 40-60: 8818.4 | 0.2 | 3683.8 / 5826.2 |
| inverse-square-0301 (T = 12.6) | dynamical | 8 | 0-5: 0.5, 5-10: 0.7, 10-20: 0.7, 20-40: 0.6, 40-60: 0.7 | never | 0.0 / 0.6 |
| inverse-square-0301 (T = 12.6) | trajectory_clock | 35 | 0-5: 3384.8, 5-10: 4927.4, 10-20: 3649.1, 20-40: 6318.8, 40-60: 8729.1 | 0.1 | 537.9 / 6157.8 |
| inverse-square-0302 (T = 19.4) | dynamical | 8 | 0-5: 0.8, 5-10: 0.7, 10-20: 0.8, 20-40: 0.8, 40-60: 1.1 | never | 0.2 / 0.2 |
| inverse-square-0302 (T = 19.4) | trajectory_clock | 35 | 0-5: 3477.7, 5-10: 6395.7, 10-20: 4997.2, 20-40: 9115.9, 40-60: 3979.6 | 0.1 | 9974.2 / 10785.3 |
