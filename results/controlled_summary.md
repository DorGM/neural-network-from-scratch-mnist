# Controlled MNIST ablation (He initialization fixed)

This is the primary comparison. Every variant uses `initialization="he"`.
Batch norm does not choose the initializer. The legacy confounded numbers
remain in `results/legacy_mnist_summary.md`.

## Seeds

Split seed 21 draws the 48,000 / 12,000 partition once and is shared by every run.
Seeds 21, 7, and 99 each seed weight initialization and the minibatch permutation.
They do not redraw the split. The same seed produces the same initial W and b
for the batch-norm and non-batch-norm variants. Gamma and beta start at ones and zeros.

## Selection

Each checkpoint is the first minimum of validation cost (cross-entropy plus the
L2 term when `lambd > 0`). The L2 term uses `lambd / (2m) * ||W||^2` with `m`
equal to the column count of that call, so a minibatch and the full validation
set do not apply the same penalty to the same weights. That scaling was not changed.

The learning rate, lambda 0.04, architecture, split, batch size, and patience match the already published legacy experiment, so those choices were not blind to the MNIST test set. Checkpoints in this controlled experiment were selected on validation cost before any test prediction for these runs was computed.

Selection lock SHA-256: `1cb2d7e27936355edf0b4d9e8b5e59373592eda5483ffb2d0bcdf0fd50b9ffdf`.

## Per-seed results

| Variant | Seed | Epochs | Best epoch | Val loss | Val accuracy | Test accuracy | Test correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 21 | 82 | 61 | 0.240084 | 0.9370 | 0.9383 | 9383/10000 |
| l2 | 21 | 112 | 91 | 0.212688 | 0.9424 | 0.9474 | 9474/10000 |
| batchnorm | 21 | 57 | 36 | 0.201788 | 0.9448 | 0.9490 | 9490/10000 |
| batchnorm_l2 | 21 | 60 | 39 | 0.197496 | 0.9472 | 0.9491 | 9491/10000 |
| baseline | 7 | 92 | 71 | 0.232194 | 0.9396 | 0.9492 | 9492/10000 |
| l2 | 7 | 133 | 112 | 0.212707 | 0.9409 | 0.9488 | 9488/10000 |
| batchnorm | 7 | 80 | 59 | 0.200020 | 0.9452 | 0.9528 | 9528/10000 |
| batchnorm_l2 | 7 | 93 | 72 | 0.188602 | 0.9473 | 0.9498 | 9498/10000 |
| baseline | 99 | 83 | 62 | 0.330390 | 0.9209 | 0.9250 | 9250/10000 |
| l2 | 99 | 103 | 82 | 0.317767 | 0.9239 | 0.9291 | 9291/10000 |
| batchnorm | 99 | 68 | 47 | 0.208990 | 0.9466 | 0.9510 | 9510/10000 |
| batchnorm_l2 | 99 | 90 | 69 | 0.197727 | 0.9482 | 0.9535 | 9535/10000 |

Means and sample standard deviations use divisor n-1 over the three seeds.

| Variant | Val loss mean | Val loss sample SD | Val acc mean | Val acc sample SD | Test mean | Test sample SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.267556 | 0.054558 | 0.9325 | 0.0101 | 0.9375 | 0.0121 |
| l2 | 0.247721 | 0.060662 | 0.9357 | 0.0103 | 0.9418 | 0.0110 |
| batchnorm | 0.203599 | 0.004751 | 0.9456 | 0.0009 | 0.9509 | 0.0019 |
| batchnorm_l2 | 0.194608 | 0.005203 | 0.9476 | 0.0006 | 0.9508 | 0.0024 |

## Paired differences

Difference = batch norm minus no batch norm, at the same lambda and seed.
A negative validation-loss difference means the batch-norm run has the lower validation cost.
Three seeds are not a significance test.

### lambd = 0.0

| Seed | Test accuracy | Val accuracy | Val loss |
| ---: | ---: | ---: | ---: |
| 21 | 0.0107 | 0.0078 | -0.038296 |
| 7 | 0.0036 | 0.0057 | -0.032174 |
| 99 | 0.0260 | 0.0257 | -0.121400 |

Mean test-accuracy difference 0.0134, sample SD 0.0114.
Mean validation-accuracy difference 0.0131, sample SD 0.0110.
Mean validation-loss difference -0.063957, sample SD 0.049841.

### lambd = 0.04

| Seed | Test accuracy | Val accuracy | Val loss |
| ---: | ---: | ---: | ---: |
| 21 | 0.0017 | 0.0048 | -0.015193 |
| 7 | 0.0010 | 0.0064 | -0.024105 |
| 99 | 0.0244 | 0.0243 | -0.120041 |

Mean test-accuracy difference 0.0090, sample SD 0.0133.
Mean validation-accuracy difference 0.0118, sample SD 0.0109.
Mean validation-loss difference -0.053113, sample SD 0.058132.

## What this sample can support

The table is the result for this fixed architecture, this one split, and these three seeds.
It does not by itself establish a general advantage for batch norm or for L2.
