"""Stage 3: final evaluation with the paper's regression model (Table 2, Synthetic):
MLP [2048, 1024, 512, 256], LayerNorm, Adam lr 1e-3, batch 64. Every method is trained 5 times
(different seeds, same seeds across methods); the checkpoint with the best validation MSE is
evaluated on the held-out test set. RMSE is reported in nm (sqrt of mean squared error)."""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from regmix import *

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEEDS = 5
FULL = [226, 2048, 1024, 512, 256, 4]
data = load_data("data")
order, dist = neighbor_table(data["X"])
N = data["X"].shape[0]
tune = json.load(open("results/tune.json"))
probs = np.load("results/regmix_probs.npy")
regmix_k = KNN_OPTIONS[probs.argmax(1)]                       # greedy policy from the trained controller

methods = {
    "Vanilla": dict(k=np.zeros(N), alpha=100.0),
    "Mixup (alpha=1, original default)": dict(k=np.full(N, N - 1), alpha=1.0),
    f"Mixup (alpha={tune['mixup']['alpha']}, tuned)": dict(k=np.full(N, N - 1), alpha=tune["mixup"]["alpha"]),
    f"Manifold Mixup (alpha={tune['manifold']['alpha']}, tuned)": dict(manifold=True, alpha=tune["manifold"]["alpha"]),
    f"Global kNN (k={tune['global_knn']['k']})": dict(k=np.full(N, tune["global_knn"]["k"]), alpha=100.0),
    f"Global Dist. (d={tune['global_dist']['d']})": dict(k=policy_global_dist(dist, tune["global_dist"]["d"]), alpha=100.0),
    "RegMix": dict(k=regmix_k, alpha=100.0),
}
results = {}
for name, cfg in methods.items():
    if cfg.get("manifold"):
        tr = jax.jit(make_manifold_trainer(data, FULL, EPOCHS, cfg["alpha"]))
        run = lambda key: tr(key)
    else:
        tr = jax.jit(make_aug_trainer(data, order, FULL, EPOCHS, cfg["alpha"], return_params=True))
        kk = jnp.asarray(cfg["k"], dtype=jnp.int32)
        run = lambda key, tr=tr, kk=kk: tr(key, kk)
    te, va = [], []
    for s in range(SEEDS):
        best, params, curve = run(jax.random.PRNGKey(1000 + s))
        te.append(rmse_nm(params, data["Xt"], data["Yt"], data["sy"]))
        va.append(rmse_nm(params, data["Xv"], data["Yv"], data["sy"]))
        print(f"{name:28s} seed {s}: val RMSE {va[-1]:.3f}  test RMSE {te[-1]:.3f}", flush=True)
    results[name] = dict(test_mean=float(np.mean(te)), test_std=float(np.std(te)),
                         val_mean=float(np.mean(va)), test_runs=te, val_runs=va,
                         mixed_fraction=float((np.asarray(cfg.get("k", [0])) > 0).mean()),
                         mean_k=float(np.mean(cfg.get("k", [0]))))
    json.dump(results, open("results/final.json", "w"), indent=1)
print("DONE final", flush=True)
