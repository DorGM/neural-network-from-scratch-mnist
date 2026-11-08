# Controlled-run diagnostics

These notes describe the finished He ablation. They do not change the training objective.

## L2 scaling

The penalty in `cost_terms` is `lambd / (2m) * sum(W^2)`, where `m` is the number of
columns in that call. The weight gradient adds `(lambd / m) * W`, which is the
derivative of that penalty. Minibatch costs use `m` near 64. The validation cost
uses `m = 12000`. The same weights therefore contribute a much larger penalty to the
logged minibatch cost than to the validation cost. Early stopping still uses the sum,
so this scale difference was left in place.

Train penalty below is the mean minibatch penalty on the restored epoch. Validation
penalty is the full validation matrix on that same checkpoint.

| Variant | Seed | Train CE | Train L2 | Val CE | Val L2 | Train L2 / Val L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 21 | 0.1417 | 0.0000 | 0.2401 | 0.000000 | n/a |
| l2 | 21 | 0.1148 | 0.0508 | 0.2124 | 0.000271 | 187.4 |
| batchnorm | 21 | 0.1572 | 0.0000 | 0.2018 | 0.000000 | n/a |
| batchnorm_l2 | 21 | 0.1459 | 0.0286 | 0.1973 | 0.000152 | 187.6 |
| baseline | 7 | 0.1168 | 0.0000 | 0.2322 | 0.000000 | n/a |
| l2 | 7 | 0.0947 | 0.0526 | 0.2124 | 0.000281 | 187.4 |
| batchnorm | 7 | 0.1286 | 0.0000 | 0.2000 | 0.000000 | n/a |
| batchnorm_l2 | 7 | 0.1189 | 0.0277 | 0.1885 | 0.000148 | 187.6 |
| baseline | 99 | 0.2133 | 0.0000 | 0.3304 | 0.000000 | n/a |
| l2 | 99 | 0.1988 | 0.0557 | 0.3175 | 0.000297 | 187.3 |
| batchnorm | 99 | 0.1498 | 0.0000 | 0.2090 | 0.000000 | n/a |
| batchnorm_l2 | 99 | 0.1318 | 0.0286 | 0.1976 | 0.000152 | 187.6 |

Validation accuracy is stored beside validation cost in `controlled_runs.csv`.
The two are not the same objective: a later epoch can have higher accuracy and worse cost,
and the restored checkpoint is the cost minimum.

## Batch norm

The forward order is affine, ReLU, then batch norm on hidden layers. It was not moved
before the ReLU. No separate affine-BN-ReLU ablation was run.
Training with a batch of at least 2 uses the batch mean and variance and updates the
running statistics. Evaluation, including accuracy, uses the running statistics.
A training batch of size 1 falls back to those running statistics and does not update them.
Gradient checks for batch norm on and off, with and without L2, are in
`results/gradient_check_decoupled.json`.

## Not run

A lambda grid and a legacy-versus-He factorial were not run. The controlled comparison
holds initialization fixed instead of estimating how much of the old gap was initialization.
