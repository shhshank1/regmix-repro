# RegMix Reproduction on Simulated Thin-Film Spectra

This repository compares RegMix with Mixup, Manifold Mixup, Global kNN, Global Distance, and a no-augmentation baseline for thin-film thickness regression. The implementation uses JAX and a physics-simulated reflectance dataset.

**Important dataset note:** the original DACON competition dataset was not used because it could not be downloaded in this study. The supplied CSV files are a simulated stand-in. Therefore, absolute RMSE values should not be compared directly with the original RegMix paper. See [`paper.pdf`](paper.pdf) for the full methodology, results, and limitations.

## Repository contents

- `regmix.py` — shared models, data loading, augmentation policies, and training helpers.
- `controller.py` — RegMix controller training with PPO.
- `tune.py`, `tune_extra.py` — baseline hyperparameter searches.
- `final.py` — final evaluation across five seeds.
- `report.py` — generates the results table and figure from saved experiment outputs.
- `thinfilm.py` — thin-film reflectance simulation utilities.
- `smoke.py` — basic runtime/training smoke test.
- `data/` — supplied training, validation, and test CSV files.
- `results/` — saved outputs, logs, the results table, and figure from the reported runs.
- `paper.pdf` — research paper.
- `requirements.txt` — Python package dependencies.

## Requirements

- Python 3.10 or newer is recommended.
- A CPU-only setup is sufficient for the supplied workflow; a GPU is not required.
- Git is needed if you are cloning this repository from GitHub.

The dependencies are listed in `requirements.txt`. Package versions are not pinned yet, so the exact environment may change as upstream packages release updates.

## Setup (Windows PowerShell)

Open PowerShell in the repository folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, you can use Command Prompt instead:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the main experiment

Run commands from the repository root, in this order. The tuning and controller stages can take time, particularly on CPU.

```bash
python tune.py
python tune_extra.py
python controller.py 300
python final.py 100
python report.py
```

These commands write or update files in `results/`. The current `results/` directory also includes saved artifacts from the experiments described in the paper. Re-running the pipeline can overwrite some of those artifacts. If you want to preserve the supplied results, make a backup of `results/` first.

To run the basic smoke test:

```bash
python smoke.py
```

## Main results

The table below reports the five-seed results from the paper. Test RMSE is in nanometres; lower is better. Standard deviations are sample standard deviations (denominator `n - 1`).

| Method | Test RMSE (nm), mean ± sample SD | Validation RMSE (nm) | RegMix better by (nm) | Paired t-test p-value |
|---|---:|---:|---:|---:|
| RegMix | 13.772 ± 0.124 | 13.508 | — | — |
| Global Distance (`d = 0.1`) | 13.824 ± 0.141 | 13.622 | 0.052 | 0.616 |
| Global kNN (`k = 8`) | 13.876 ± 0.244 | 13.692 | 0.103 | 0.464 |
| Vanilla (no augmentation) | 13.908 ± 0.157 | 13.887 | 0.136 | 0.108 |
| Mixup (`α = 1`) | 13.986 ± 0.165 | 13.860 | 0.214 | 0.147 |
| Manifold Mixup (`α = 0.02`) | 14.004 ± 0.193 | 13.937 | 0.231 | 0.157 |
| Mixup (`α = 0.02`, tuned) | 14.136 ± 0.153 | 13.908 | 0.363 | 0.013 |

The reported main-run table uses five paired seeds. The paper also presents a ten-seed analysis and additional controls; consult the paper for those results and their interpretation. The repository's existing saved artifacts should be checked against the paper before treating them as a fresh, end-to-end reproduction.

Figure: [`results/results.png`](results/results.png). The horizontal axis is truncated at 12.5 nm, so use the printed values and individual seed dots rather than bar lengths alone.

## Interpretation and limitations

- RegMix has the lowest mean test RMSE in the five-seed main comparison, but its improvement over no augmentation is small and not statistically significant in that comparison.
- RegMix performs better than the two Mixup settings in the reported ten-seed analysis after the paper's multiple-comparison correction; it is not distinguishable from the tuned Global Distance baseline.
- The data are simulated and noise-free, and the original competition dataset was not used. Conclusions should not be generalized directly to real measurement data.
- The implementation uses on-the-fly mixing and tunes baselines with a grid search on a smaller proxy model. These choices differ from parts of the original paper's procedure.
- AdaMixup and C-Mixup are not implemented. The paper explains these omissions and other limitations.

## Citation

If you use this code, cite the accompanying paper and the original RegMix work:

- Hwang, S.-H. and Whang, S. E. (2021). *RegMix: Data mixing augmentation for regression*. arXiv:2106.03374.
- Telang, N., Shelke, S., and Suryawanshi, Y. *Data Augmentation for Thin-Film Thickness Regression: Comparing Mixup, kNN-Limited Mixing and RegMix on Simulated Spectra*. See [`paper.pdf`](paper.pdf) for the manuscript.

## License

No license has been added yet. Until the authors choose and add a license, do not assume that this repository grants permission to reuse, modify, or redistribute its code or data.
