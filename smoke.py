import time
import jax, jax.numpy as jnp, numpy as np
from regmix import *

data = load_data("data")
order, dist = neighbor_table(data["X"])
print("N", data["X"].shape, "k options", KNN_OPTIONS, "max dist", dist.max())

# 1) batched proxy trainer timing: 20 models, vanilla vs mixup-style
proxy = [226, 256, 128, 4]
E = 40
tr = make_aug_trainer(data, order, proxy, E, alpha=100.0)
batched = jax.jit(jax.vmap(lambda key, k: tr(key, k)[0]))
keys = jax.random.split(jax.random.PRNGKey(0), 20)
for name, k in [("vanilla", 0), ("k=4", 4), ("k=64", 64)]:
    ks = jnp.full((20, 600), k, dtype=jnp.int32)
    t = time.time(); out = np.asarray(batched(keys, ks)); dt = time.time() - t
    print(f"{name:8s} proxy val MSE mean={out.mean():.4f} std={out.std():.4f}  ({dt:.1f}s incl compile)")
t = time.time(); out = np.asarray(batched(keys, jnp.full((20, 600), 4, dtype=jnp.int32))); print("second call", time.time() - t)

# 2) full-size final model timing
full = [226, 2048, 1024, 512, 256, 4]
trf = jax.jit(make_aug_trainer(data, order, full, 30, alpha=100.0, return_params=True))
t = time.time(); b, p, c = trf(jax.random.PRNGKey(1), jnp.zeros(600, dtype=jnp.int32)); b.block_until_ready()
print("full model 30 epochs vanilla:", time.time() - t, "s; best val mse", float(b),
      "test RMSE nm", rmse_nm(p, data["Xt"], data["Yt"], data["sy"]))
