# Global-law transfer, end to end

Mean held-out angular error in arcmin across target worlds, by evidence budget k.

## Inverse-square targets (10 worlds)

| k | 4 | 8 | 16 | 32 | 64 | 128 | AUC (rad) |
|---|---|---|---|---|---|---|---|
| Shared law (frozen p) | 4040.56 | 3726.68 | 1271.49 | 116.28 | 10.76 | 3.51 | 0.4158 |
| Flexible baseline | 3998.47 | 3811.67 | 4549.58 | 3752.12 | 3834.01 | 4607.09 | 1.1781 |

## Hidden-exponent targets (10 worlds)

| k | 4 | 8 | 16 | 32 | 64 | 128 | AUC (rad) |
|---|---|---|---|---|---|---|---|
| Frozen p = 2 (misspecified) | 5349.81 | 5543.30 | 4205.41 | 3847.41 | 3143.01 | 928.32 | 1.1565 |
| Adapted exponent | 5703.91 | 6165.69 | 6776.97 | 1307.02 | 302.77 | 13.29 | 1.0129 |

## Data-blind verdicts (shared law, k = 32)

| World | Entry (arcmin) | Data-blind (arcmin) | Credit |
|---|---|---|---|
| inverse-square-0100 | 143.32 | 7453.49 | True |
| inverse-square-0101 | 3.36 | 8206.58 | True |
| inverse-square-0102 | 60.85 | 9204.15 | True |

Discovery froze p = 2.0001 from per-world fits 2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.001.
