"""Extend the Mixup / Manifold Mixup alpha grids downwards (best alpha was on the grid edge)."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from regmix import *

PROXY, E = [226, 256, 128, 4], 40
data = load_data("data")
order, dist = neighbor_table(data["X"])
N = data["X"].shape[0]
keys = jax.random.split(jax.random.PRNGKey(123), 20)   # same keys as tune.py
t = json.load(open("results/tune.json"))

for a in [0.02, 0.05, 0.1]:
    tr = make_aug_trainer(data, order, PROXY, E, a)
    f = jax.jit(jax.vmap(lambda key, k: tr(key, k)[0]))
    v = np.asarray(f(keys, jnp.full((20, N), N - 1, dtype=jnp.int32)))
    t["mixup"]["grid"][str(a)] = (float(v.mean()), float(v.std())); print("mixup", a, v.mean(), flush=True)
    tr = make_manifold_trainer(data, PROXY, E, a)
    v = np.asarray(jax.jit(jax.vmap(lambda key: tr(key)[0]))(keys))
    t["manifold"]["grid"][str(a)] = (float(v.mean()), float(v.std())); print("manifold", a, v.mean(), flush=True)

for name in ("mixup", "manifold"):
    g = t[name]["grid"]
    best = min(g, key=lambda a: g[a][0])
    t[name]["alpha"], t[name]["val_mse"] = float(best), g[best][0]
    print(name, "best alpha", best, g[best])
json.dump(t, open("results/tune.json", "w"), indent=1)
