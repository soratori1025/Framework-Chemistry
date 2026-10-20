# Paper 2 — collected results

## E-REC

_not run_

## E-SYNTH — ground-truth recovery

Recall of the planted informative features (mean over seeds).

| Scenario | Scorecard | SIREN (full) | SIREN $-$ stability | SIREN $-$ synergy | SIREN $-$ synergy $-$ stability |
|---|---|---|---|---|---|
| marginal | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| duplicate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| suppression | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| xor | 0.100 | 0.900 | 0.900 | 0.300 | 0.200 |
| masked_synergy | 0.800 | 0.900 | 0.900 | 0.800 | 1.000 |
| noise | -- | -- | -- | -- | -- |
| real_synergy | 0.400 | 0.600 | 0.600 | 0.500 | 0.500 |
| real_suppression | 0.300 | 0.700 | 0.700 | 0.600 | 0.600 |

## E-CASES — named suppressors on the molecular benchmarks

Descriptors whose association with the target is masked by the selected
context: marginal MI is near zero while conditional MI is not.

| Dataset | Descriptor | Marginal | Conditional | Gain |
|---|---|---|---|---|
| esol | `fr_phenol_noOrthoHbond` | 0.000 | 0.281 | 0.281 |
|  | `fr_Ar_OH` | 0.008 | 0.242 | 0.234 |
|  | `PEOE_VSA11` | 0.000 | 0.218 | 0.218 |
| freesolv | `SlogP_VSA10` | 0.021 | 0.419 | 0.398 |
|  | `fr_NH0` | 0.051 | 0.362 | 0.311 |
|  | `fr_Ar_N` | 0.059 | 0.347 | 0.288 |
| lipophilicity | `fr_C_O_noCOO` | 0.016 | 0.125 | 0.109 |
|  | `fr_unbrch_alkane` | 0.000 | 0.081 | 0.081 |
|  | `fr_C_O` | 0.009 | 0.073 | 0.064 |
| bbbp | `fr_ketone_Topliss` | 0.008 | 0.231 | 0.223 |
|  | `NumSaturatedCarbocycles` | 0.007 | 0.225 | 0.217 |
|  | `fr_aniline` | 0.032 | 0.240 | 0.208 |
| bace | `RingCount` | 0.000 | 0.265 | 0.265 |
|  | `PEOE_VSA5` | 0.017 | 0.255 | 0.238 |
|  | `fr_alkyl_halide` | 0.009 | 0.243 | 0.234 |

## E-LEAK

_not run_

## E-DOWNSTREAM

_not run_

## E-STABILITY

_not run_
