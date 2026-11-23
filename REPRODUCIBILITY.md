# Protocol and known manuscript wording discrepancies

The frozen results reproduce the final `paper_v12` study. No manuscript is modified by this release. All reported accuracies use the **7,962 source-disjoint** UrbanSound8K clips, not its full 8,732-clip set. Means, support bases and TextSpan fits use all 5,929 Clotho audio recordings; their captions are not used.

Two implementation details differ from the wording in that manuscript:

1. **Gaussian vectors are not unit-normalized.** The saved baseline is `default_rng(4503).standard_normal((800,512)).astype(float32)`. Its mean row norm is about 22.61. Candidate magnitudes affect raw-variance TextSpan selection. The release preserves the saved experiment; changing row normalization would be a different baseline.
2. **The class prompts are the ten explicit sentences in `assets/protocol.json`**, such as “A dog barking.” The manuscript's generic “This is a sound of {label}.” template does not describe the cached experiment. The frozen sentences must be used to reproduce its numbers.

## What is being changed?

The full-depth component and cumulative-removal experiments replace **direct contributions in the final embedding** by Clotho means. They do not rerun the intervening network. Accounting terms are extracted independently, including the initial residual, attention output biases, complete transformer MLP writes, patch-merging offsets, and readout offsets. Independent sums must reconstruct the intact embedding within the stated tolerance; this is an algebraic check, not evidence of causal faithfulness.

The final-stage reconstruction keeps each head mean plus its signed projection onto K TextSpan directions. The remainder is independently extracted and replaced by its Clotho mean. At K=0 every term is constant. The “uncompressed head contributions” reference keeps all 64 head contributions with the mean remainder. It is dictionary-independent.

The class-group experiment is different: it **internally** replaces selected heads' attended-value slices, before the attention output projection, by their position-specific Clotho means. All remaining final-stage computation and the original nonlinear readout are rerun. The output projection bias is not assigned to individual heads.

## Frozen decisions and uncertainty

`assets/groups.json` contains the exact ten unique random complement selections per nonempty target group (seed 4520). Zero-head classes use one empty no-op selection. Random-edit bars span the 5th–95th percentiles of these draws, not a confidence interval. The cumulative random-order curve instead enumerates all 4,096 subsets of twelve blocks; each K averages all C(12,K) subsets, with 5th–95th percentile shading.

The six example heads are a qualitative, deliberately selected illustration, not an unbiased estimate of explanation quality. Their selection and summaries are frozen in `assets/selected-heads.json`. Class mappings, including empty lists, were generated before internal interventions. No new mapping is silently generated if a refit changes the selected descriptions.

## Numerical checks

Use float32 model inference, float64 fitting/scoring, TF32 disabled. Real checkpoint smoke checks compare intact versus replayed embeddings (absolute 1e-6, relative 1e-5), independent contribution sums, single versus batch contributions, CPU versus GPU contributions (absolute 1e-6), and bit-exact no-op intervention. Every frozen text bank is independently re-encoded and checked (absolute 1e-6, relative 1e-5) before full audio extraction. Each waveform is checked against its SHA-256 before use. Shards preserve sample IDs, have checksums and independent accounting diagnostics, and are resumable only with identical recorded inputs, code and environment.

`check` compares every numeric result field to its frozen reference at 1e-10. Different hardware/library reductions can change predictions near ties; a failed comparison must be inspected and reported, not hidden by widening this tolerance. Validation of the public implementation is recorded in `reference/validation.json` when complete.

## Normalized-Gaussian supplement

At the author's request, `normalized-gaussian` repeats fitting after normalizing each row of the **same** seed-4503 Gaussian bank in float32. It uses the identical contributions, split, support rank and reconstruction recipe. This is a separate experiment, not a replacement of the paper baseline. The supplementary result is in `reference/normalized-gaussian.csv` and `docs/figures/normalized-gaussian.pdf`: K=3 accuracy is 30.24%, versus 32.03% for the saved unnormalized bank and 45.42% for the LLM bank; K=32 is 67.19%. These supplementary fits were evaluated using the batch-8 extraction, whose paper reconstruction accuracies matched every reference value.
