"""Stage 1: tune baseline hyper-parameters on the validation set with the (cheap) proxy model.

Paper tunes Global kNN's k and Global Distance's d with Bayesian Optimisation; here a dense grid
is used instead (the search spaces are 1-D). Mixup / Manifold Mixup alpha is picked from a small
grid since the paper does not state it. 20 proxy models (different inits / mixing randomness,
same keys for every candidate) are averaged per candidate to reduce noise.
"""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from regmix import *

PROXY = [226, 256, 128, 4]
E = 40
data = load_data("data")
order, dist = neighbor_table(data["X"])
N = data["X"].shape[0]
keys = jax.random.split(jax.random.PRNGKey(123), 20)
out = {}


def log(*a):
    print(*a, flush=True)


def eval_policy(k_vec, alpha):
    tr = make_aug_trainer(data, order, PROXY, E, alpha)
    f = jax.jit(jax.vmap(lambda key, k: tr(key, k)[0]))
    ks = jnp.tile(jnp.asarray(k_vec, dtype=jnp.int32)[None], (20, 1))
    v = np.asarray(f(keys, ks))
    return float(v.mean()), float(v.std())


# ---- vanilla reference
v = eval_policy(np.zeros(N), 100.0)
out["vanilla"] = dict(val_mse=v[0], std=v[1]); log("vanilla", v)

# ---- Mixup alpha grid (all pairs mixed)
res = {}
for a in [0.2, 0.5, 1.0, 2.0]:
    res[a] = eval_policy(np.full(N, N - 1), a); log("mixup alpha", a, res[a])
best_a = min(res, key=lambda a: res[a][0])
out["mixup"] = dict(alpha=best_a, val_mse=res[best_a][0], grid={str(a): res[a] for a in res})

# ---- Manifold Mixup alpha grid
res = {}
for a in [0.2, 0.5, 1.0, 2.0]:
    tr = make_manifold_trainer(data, PROXY, E, a)
    f = jax.jit(jax.vmap(lambda key: tr(key)[0]))
    v = np.asarray(f(keys)); res[a] = (float(v.mean()), float(v.std())); log("manifold alpha", a, res[a])
best_a = min(res, key=lambda a: res[a][0])
out["manifold"] = dict(alpha=best_a, val_mse=res[best_a][0], grid={str(a): res[a] for a in res})

# ---- Global kNN (alpha=100 as in RegMix)
res = {}
for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
    res[k] = eval_policy(np.full(N, k), 100.0); log("global kNN k", k, res[k])
best_k = min(res, key=lambda k: res[k][0])
out["global_knn"] = dict(k=best_k, val_mse=res[best_k][0], grid={str(k): res[k] for k in res})

# ---- Global Distance (normalised threshold d in (0, 1])
res = {}
for d in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]:
    kv = policy_global_dist(dist, d)
    res[d] = eval_policy(kv, 100.0); log("global dist d", d, "mean k", kv.mean(), res[d])
best_d = min(res, key=lambda d: res[d][0])
out["global_dist"] = dict(d=best_d, val_mse=res[best_d][0], grid={str(d): res[d] for d in res})

json.dump(out, open("results/tune.json", "w"), indent=1)
log("DONE tune")
