"""Build results table (markdown/CSV), significance numbers and figures from results/*.json|npy."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

r = json.load(open("results/final.json"))
hist = json.load(open("results/regmix_hist.json"))
probs = np.load("results/regmix_probs.npy")
opts = [0] + [2 ** i for i in range(10)]

# ---- table + paired tests vs RegMix (same seeds => paired by seed index)
names = list(r)
reg = np.array(r["RegMix"]["test_runs"])
lines = ["| Method | Test RMSE (nm), mean ± std over 5 seeds | Val RMSE (nm) | RegMix better by (nm) | paired t-test p |",
         "|---|---|---|---|---|"]
for n in names:
    t = np.array(r[n]["test_runs"])
    diff = t.mean() - reg.mean()
    p = stats.ttest_rel(t, reg).pvalue if n != "RegMix" else float("nan")
    lines.append(f"| {n} | {r[n]['test_mean']:.3f} ± {r[n]['test_std']:.3f} | {r[n]['val_mean']:.3f} | "
                 f"{'-' if n == 'RegMix' else f'{diff:+.3f}'} | {'-' if n == 'RegMix' else f'{p:.3f}'} |")
open("results/results_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- figure
INK, MUTED, ACC, GRAY = "#1f2328", "#6b7280", "#d4541a", "#9aa1ab"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})

order = sorted(names, key=lambda n: r[n]["test_mean"])
y = np.arange(len(order))
for i, n in enumerate(order):
    col = ACC if n == "RegMix" else GRAY
    ax[0].barh(i, r[n]["test_mean"], xerr=r[n]["test_std"], color=col, height=0.55,
               error_kw=dict(ecolor=INK, lw=1.2, capsize=3))
    ax[0].scatter(r[n]["test_runs"], [i] * 5, s=10, color=INK, zorder=3)
    ax[0].text(r[n]["test_mean"] + 0.33, i, f"{r[n]['test_mean']:.2f}", va="center", color=INK, fontsize=9)
ax[0].set_yticks(y); ax[0].set_yticklabels(order, fontsize=8.5); ax[0].invert_yaxis()
ax[0].set_xlim(12.5, 14.6); ax[0].set_xlabel("Test RMSE (nm) - lower is better; dots = 5 seeds")
ax[0].set_title("Final results (simulated Synthetic dataset)", loc="left", color=INK, fontsize=11)

w = 0.65
ax[1].bar(range(len(opts)), probs.mean(0), color=ACC, width=w)
ax[1].axhline(1 / len(opts), color=MUTED, lw=1, ls="--"); ax[1].text(len(opts) - 0.5, 1 / len(opts) + 0.002, "uniform", ha="right", color=MUTED, fontsize=8)
ax[1].set_xticks(range(len(opts))); ax[1].set_xticklabels(opts, fontsize=8)
ax[1].set_xlabel("kNN option"); ax[1].set_ylabel("Mean policy probability")
ax[1].set_title("Learned kNN policy (cf. paper Fig. 6)", loc="left", color=INK, fontsize=11)

it = [h["it"] for h in hist]; m = np.array([h["mean_val_mse"] for h in hist])
sm = np.convolve(m, np.ones(10) / 10, mode="valid")
ax[2].plot(it, m, color=GRAY, lw=1, alpha=0.7, label="mean val MSE of 20 sampled policies")
ax[2].plot(it[9:], sm, color=ACC, lw=2, label="10-iteration average")
ax[2].set_xlabel("Controller iteration"); ax[2].set_ylabel("Validation MSE (standardised)")
ax[2].set_title("Controller training (cf. paper Fig. 7b/8b)", loc="left", color=INK, fontsize=11)
ax[2].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("results/results.png", dpi=150)
