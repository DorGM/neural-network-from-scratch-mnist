# Legacy MNIST experiment (confounded)

This file preserves the original published experiment. It is a historical record. It is **not** a controlled estimate of batch normalization.

## Why these numbers are confounded

`initialize_parameters` in `main.py` used `randn / fan_in` when batch norm was off and `randn * sqrt(2 / fan_in)` when batch norm was on. The BN-versus-no-BN contrast therefore changed normalization and weight initialization together. Do not relabel the table below as the isolated effect of batch norm.

The twelve checkpoints under `checkpoints/` were left in place and were not rewritten. The executed notebook `experiments.ipynb` was left in place, including its saved outputs.

## Provenance

- Project: `neural-network-from-scratch-mnist`
- Sources copied from: `README.md` (conclusions below), executed outputs in `experiments.ipynb`, and read-only loads of `checkpoints/*.pt`
- Inspection date: 2026-10-14
- Notebook runtime, from its first code cell: Windows-11-10.0.26200-SP0, Python 3.13.2, torch `2.14.1+cpu`, torchvision `0.29.1+cpu`, CUDA false, 14 threads, device cpu
- Protocol stated by the original README: widths `[784, 20, 7, 5, 10]`; ReLU hidden layers; softmax output; pixels `/255`; official MNIST 60,000 train / 10,000 test; fit/validation split seed 21 giving 48,000 / 12,000; init and shuffle seeds 21, 7, and 99; batch size 64; learning rate 0.009; maximum 300 epochs; patience 20; checkpoint chosen on validation cost; test evaluation afterward
- Variants: baseline (`lambd=0`), L2 (`lambd=0.04`), batch norm, batch norm + L2 (`lambd=0.04`)

## Original README results

Test accuracy is `correct / 10000` on the official test split, after restoring the validation checkpoint. Means and sample standard deviations are over the three seeds.

| Variant | Seed 21 test | Test mean | Test sample std | Val mean |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.9420 | 0.9332 | 0.0092 | 0.9295 |
| L2 (`lambd=0.04`) | 0.9464 | 0.9349 | 0.0117 | 0.9299 |
| Batch norm | 0.9490 | 0.9509 | 0.0019 | 0.9456 |
| Batch norm + L2 | 0.9491 | 0.9508 | 0.0024 | 0.9476 |

Original README interpretation, retained as originally written for this confounded run: batch norm is the change that raises accuracy and shrinks the seed spread; L2 alone does not separate from the baseline spread; with batch norm on, L2 leaves the test mean essentially unchanged. That interpretation mixes initialization with batch norm and is not reused as the controlled conclusion.

## Notebook per-seed outputs

Restored-checkpoint accuracies printed by the comparison cell. `epochs` is how many epochs ran. `best` is the restored epoch index, counting from 0. Train cost in the learning curves is the mean mini-batch cost and includes the L2 penalty when `lambd > 0`.

| Variant | Seed | Epochs | Best epoch | Train acc | Val acc | Test acc | Test correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 21 | 86 | 65 | 0.9658 | 0.9391 | 0.9420 | 9420/10000 |
| l2 | 21 | 105 | 84 | 0.9695 | 0.9429 | 0.9464 | 9464/10000 |
| batchnorm | 21 | 57 | 36 | 0.9706 | 0.9448 | 0.9490 | 9490/10000 |
| batchnorm_l2 | 21 | 60 | 39 | 0.9711 | 0.9472 | 0.9491 | 9491/10000 |
| baseline | 7 | 107 | 86 | 0.9638 | 0.9302 | 0.9340 | 9340/10000 |
| l2 | 7 | 128 | 107 | 0.9637 | 0.9302 | 0.9354 | 9354/10000 |
| batchnorm | 7 | 80 | 59 | 0.9757 | 0.9452 | 0.9528 | 9528/10000 |
| batchnorm_l2 | 7 | 93 | 72 | 0.9776 | 0.9473 | 0.9498 | 9498/10000 |
| baseline | 99 | 111 | 90 | 0.9505 | 0.9193 | 0.9236 | 9236/10000 |
| l2 | 99 | 131 | 110 | 0.9522 | 0.9164 | 0.9230 | 9230/10000 |
| batchnorm | 99 | 68 | 47 | 0.9717 | 0.9466 | 0.9510 | 9510/10000 |
| batchnorm_l2 | 99 | 90 | 69 | 0.9743 | 0.9482 | 0.9535 | 9535/10000 |

Notebook aggregate cell (`statistics.mean` / `statistics.stdev`):

| Variant | Val mean | Val sample std | Test mean | Test sample std |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.9295 | 0.0099 | 0.9332 | 0.0092 |
| l2 | 0.9299 | 0.0133 | 0.9349 | 0.0117 |
| batchnorm | 0.9456 | 0.0009 | 0.9509 | 0.0019 |
| batchnorm_l2 | 0.9476 | 0.0006 | 0.9508 | 0.0024 |

Seed 21 L2 weight norms printed by the notebook:

| Variant | W1 | W2 | W3 | W4 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 6.396697044372559 | 5.405289649963379 | 5.373213768005371 | 5.6964521408081055 |
| l2 | 6.3357625007629395 | 5.362521171569824 | 5.313912868499756 | 5.522260665893555 |
| batchnorm | 7.3411359786987305 | 3.821951389312744 | 2.8100802898406982 | 6.302443027496338 |
| batchnorm_l2 | 6.501905918121338 | 3.2552764415740967 | 2.3629708290100098 | 5.7464704513549805 |

Gradient check printed by the notebook on the 5-4-3 fixture (`check_gradients(seed=0)`), before any of these training runs:

| use_bnorm | lambd | max_abs_autograd | max_abs_numeric |
| --- | ---: | ---: | ---: |
| false | 0.0 | 2.7939677238464355e-08 | 0.0 |
| false | 0.1 | 2.7939677238464355e-08 | 0.00017652101814746857 |
| true | 0.0 | 1.1920928955078125e-07 | null |
| true | 0.05 | 1.1920928955078125e-07 | null |

## Checkpoint history at the restored epoch

Read on 2026-10-14 from the existing `checkpoints/*.pt` files with `torch.load(..., map_location="cpu", weights_only=False)`. Files were not saved again. Costs are the history values at `best_epoch`, rounded to 6 decimal places. Train cost is the mean mini-batch cost from that epoch. Validation cost is the full validation set in eval mode. Both include the L2 penalty when `lambd > 0`.

| Checkpoint | Epochs | Best epoch | Train cost | Val cost | Train acc | Val acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_seed21.pt | 86 | 65 | 0.131372 | 0.220114 | 0.965813 | 0.939083 |
| baseline_seed7.pt | 107 | 86 | 0.143791 | 0.295713 | 0.963750 | 0.930167 |
| baseline_seed99.pt | 111 | 90 | 0.197374 | 0.339657 | 0.950521 | 0.919250 |
| l2_seed21.pt | 105 | 84 | 0.160794 | 0.206029 | 0.969479 | 0.942917 |
| l2_seed7.pt | 128 | 107 | 0.193062 | 0.277508 | 0.963729 | 0.930250 |
| l2_seed99.pt | 131 | 110 | 0.236447 | 0.333597 | 0.952229 | 0.916417 |
| batchnorm_seed21.pt | 57 | 36 | 0.157163 | 0.201788 | 0.970563 | 0.944833 |
| batchnorm_seed7.pt | 80 | 59 | 0.128601 | 0.200020 | 0.975708 | 0.945250 |
| batchnorm_seed99.pt | 68 | 47 | 0.149789 | 0.208990 | 0.971729 | 0.946583 |
| batchnorm_l2_seed21.pt | 60 | 39 | 0.174484 | 0.197496 | 0.971146 | 0.947167 |
| batchnorm_l2_seed7.pt | 93 | 72 | 0.146567 | 0.188602 | 0.977583 | 0.947333 |
| batchnorm_l2_seed99.pt | 90 | 69 | 0.160378 | 0.197727 | 0.974292 | 0.948250 |

Every saved history has `restored_best: true`. Non-BN checkpoints store `W1`–`W4` and `b1`–`b4`. BN checkpoints also store `gamma`, `beta`, `running_mean`, and `running_var` for hidden layers 1–3.

## Original notebook conclusion

The closing markdown cell of `experiments.ipynb` states that the gradient check matched autograd to about `1e-7`, finite differences on the non-batch-norm fixture stayed under `2e-4`, seed-21 test accuracies were 0.9420, 0.9464, 0.9490, and 0.9491, and the three-seed test means were 0.9332, 0.9349, 0.9509, and 0.9508 with sample standard deviations 0.0092, 0.0117, 0.0019, and 0.0024. It attributes the accuracy gain and tighter seed spread to batch norm. That attribution is kept here only as the original claim about this confounded design.
