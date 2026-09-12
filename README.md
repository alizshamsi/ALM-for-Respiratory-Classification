# RespARIA — Auditing Audio-Language Models for Pediatric Respiratory Sound Classification

Code, per-item predictions, bootstrap resamples, logs and pre-registration for
the paper. Every number in the paper is reproducible from this repository; the
analysis stage needs no GPU.

> **Headline.** On the held-out SPRSound test2023 split, one 57 M-parameter
> LoRA adapter on Qwen2-Audio and four task-specific 86 M AST classifiers reach
> the same mean MCC (0.575 vs 0.585, paired difference −0.010 [−0.063, +0.026])
> but fail differently: AST is better on events, the adapter on five-class
> record labels, and the adapter never predicts four rare event types that the
> validation split did not contain.

## What is here

| | |
|---|---|
| `src/` | 46 scripts, unmodified from the run that produced the paper |
| `run/` | shell wrappers for the GPU stages |
| `manifests/`, `qa/` | official splits and the multiple-choice items |
| `results/` | every metric JSON, per-item prediction file, and the 66 shared bootstrap resamples |
| `results/step5/PREREGISTRATION.md` | hypotheses registered before the test set was opened, with hashes and timestamps. Two of them failed and are reported as failures |
| `logs/` | every run, including the ones that went wrong |
| `ckpt/*/effective_config.json`, `history.json` | the exact configuration each model received |
| `paper/` | the LaTeX tables, generated from JSON |

**Not here:** raw or derived audio, base model weights, trainer optimizer
state. All are regenerated or downloaded by the `Makefile`. Fine-tuned weights
are on the Hugging Face Hub — see below.

## Quickstart

```bash
git clone https://github.com/alizshamsi/ALM-for-Respiratory-Classification/REPO.git && cd REPO
cp configs/paths.env.example configs/paths.env   # edit the two paths
make setup
make check                                       # prints torch / CUDA / paths
make analysis                                    # CPU only, 
```

`make analysis` recomputes every table and interval in the paper from the
committed predictions. It does not need a GPU, the dataset, or the model.

## Full pipeline, from a bare machine

```bash
make data        # clone SPRSound, build manifests, audit splits      
make audio       # materialise the shared 16 kHz tree, 2.5 GB        
make qa          # build 17,210 / 3,568 / 2,766 / 7,990 items     
make model       # download Qwen2-Audio-7B-Instruct, 17 GB          
make train       # LoRA fine-tuning                                
make train-ast   # four AST baselines                                
make evaluate    # score both models on test2023                     
make analysis    # all statistics                                    
```

Reference hardware: one Quadro RTX 8000 (48 GB, Turing, no bfloat16), shared
with other users. `make train` needs one line filled in first — see
[REPRODUCE.md](REPRODUCE.md).

## Fine-tuned weights

| Artifact | Size | Location |
|---|---|---|
| Qwen2-Audio LoRA adapter, `checkpoint-1000` | 228 MB | `https://huggingface.co/USER/resparia-qwen2audio-lora` |
| AST t11 / t12 / t21 / t22 | 4 × 345 MB | `https://huggingface.co/USER/resparia-ast-baselines` |

They exceed GitHub's 100 MB per-file limit. `adapter_config.json` and
`effective_config.json` are committed here so you can inspect the configuration
without downloading anything.

## Two caveats that will save you an hour

**Absolute paths in `manifests/` and `qa/`.** The committed copies embed the
`wav` paths of the original machine. They are here as the reference artifacts
that produced the paper. `make data qa` regenerates them against your own
`RESP_RAW_SPRSOUND`; only the `wav` field differs, and everything downstream
reads the regenerated copies.

**cuDNN must be disabled on Turing.** The scripts set
`torch.backends.cudnn.enabled = False` before any CUDA context exists. Without
it, Whisper's first convolution fails with `CUDNN_STATUS_NOT_INITIALIZED` on
this hardware.

## Data

SPRSound is distributed by SJTU under its own terms:
<https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound>. Please cite the corpus
paper and observe its licence.

**This repository contains no audio.** Filenames in `manifests/`, `qa/` and the
prediction files carry SPRSound's own pseudonymous patient identifier, age, sex
and chest location, as published in the corpus. Forty-seven test2023 records
carry no identifier; the paper discloses this, and the bootstrap treats them as
a single cluster, which is conservative.

## Evaluation protocol

The hazard checklist (Table II of the paper) is implemented as runnable
controls:

| Hazard | Script |
|---|---|
| Majority-class accuracy | `src/step5_01_floors.py` |
| Letter-to-class mapping | `src/test_metrics_mc.py` |
| Option-position preference | `src/step5_05a_build_shift.py`, `src/step5_05c_shift_analysis.py` |
| Answering without audio | `src/step5_04b_blind_{qwen,ast}.py`, `src/step5_04c_blind_analysis.py` |
| Loudness and split-dependent normalisation | `src/audit_audio.py` |
| Duration cues | `src/shortcut_baseline.py` |
| Window-clamp collapse | `src/audit_audio.py` |
| Derivable labels | `src/compose_check.py`, `src/step5_03_composition.py` |
| Clustered inference | `src/step5_stats.py` |
| Recalibration on a small split | `src/step4_04b_recal_cv.py` |
| Selection blind spots | `src/audit.py` |

`src/audit.py` is dataset-agnostic. Point it at any manifest with an id field,
a group field and label fields and it runs the leakage, duplicate, class-shift,
derivability and trivial-baseline checks.

## Citation

See [CITATION.cff](CITATION.cff).

## Licence

Code: MIT (see [LICENSE](LICENSE)). The SPRSound corpus is not covered by it.
