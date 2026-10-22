"""RegMix (Hwang & Whang, arXiv:2106.03374) re-implementation in JAX, plus the baselines.

Everything that mixes examples (Mixup, Global kNN, Global Distance, RegMix) is expressed as a
"kNN mixing policy": a vector k of length N where k[i] is how many nearest neighbours example i
may be mixed with (0 = never mixed). This is exactly the paper's formulation:
  * Mixup           -> k[i] = N-1 for all i
  * Global kNN      -> k[i] = const
  * Global Distance -> k[i] = number of neighbours of i within distance d
  * RegMix          -> k learned per example by an RL controller (PPO)

Deviation from the paper (for CPU feasibility): instead of materialising the whole mixed set
Mix(D, P) once, each epoch every example i draws ONE random partner among its k[i] nearest
neighbours and one lambda ~ Beta(alpha, alpha). The expected augmentation distribution is the
same; the training set per epoch is D plus N fresh mixed examples.
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

LABELS = [f"layer_{i}" for i in range(1, 5)]
FEATS = [str(i) for i in range(226)]
KNN_OPTIONS = np.array([0] + [2 ** i for i in range(10)], dtype=np.int32)  # {0} U {2^i | i in [0,9]}


# ----------------------------------------------------------------------------- data
def load_data(data_dir):
    tr, va, te = [pd.read_csv(os.path.join(data_dir, f"{n}.csv")) for n in ("train", "val", "test")]
    mx, sx = tr[FEATS].values.mean(0), tr[FEATS].values.std(0) + 1e-8
    my, sy = tr[LABELS].values.mean(0), tr[LABELS].values.std(0)
    fx = lambda d: ((d[FEATS].values - mx) / sx).astype(np.float32)
    fy = lambda d: ((d[LABELS].values - my) / sy).astype(np.float32)
    return dict(X=fx(tr), Y=fy(tr), Xv=fx(va), Yv=fy(va), Xt=fx(te), Yt=fy(te), sy=sy.astype(np.float32))


def neighbor_table(X):
    """nn_order[i] = other examples sorted by Euclidean distance; dist[i] the matching distances."""
    sq = (X ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    d = np.sqrt(d2)
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, 1)[:, :-1]
    return order.astype(np.int32), np.take_along_axis(d, order, 1).astype(np.float32)


def policy_global_dist(dist, d_norm):
    """k[i] = number of neighbours within d_norm * (max pairwise distance)."""
    return (dist <= d_norm * dist.max()).sum(1).astype(np.int32)


# ----------------------------------------------------------------------------- model
def init_mlp(key, sizes, ln=True):
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        p = {"w": jax.random.normal(keys[i], (a, b)) * jnp.sqrt(2.0 / a), "b": jnp.zeros(b)}
        if ln and i < len(sizes) - 2:
            p["g"], p["be"] = jnp.ones(b), jnp.zeros(b)
        params.append(p)
    return params


def forward(params, x, mix=None):
    """MLP with optional LayerNorm. mix=(m, lam, perm) implements Manifold Mixup at layer m."""
    h, L = x, len(params)
    for l, p in enumerate(params):
        if mix is not None:
            m, lam, perm = mix
            h = jnp.where(m == l, lam * h + (1 - lam) * h[perm], h)
        h = h @ p["w"] + p["b"]
        if l < L - 1:
            if "g" in p:
                h = (h - h.mean(-1, keepdims=True)) / jnp.sqrt(h.var(-1, keepdims=True) + 1e-5)
                h = h * p["g"] + p["be"]
            h = jax.nn.relu(h)
    return h


def rmse_nm(params, X, Y, sy):
    """RMSE in original (nm) units: sqrt of mean squared error over all samples and 4 labels."""
    pred = forward(params, jnp.asarray(X))
    return float(jnp.sqrt((((pred - jnp.asarray(Y)) * jnp.asarray(sy)) ** 2).mean()))


# ----------------------------------------------------------------------------- trainers
def _select(cond, a, b):
    return jax.tree_util.tree_map(lambda u, v: jnp.where(cond, u, v), a, b)


def make_aug_trainer(data, nn_order, sizes, epochs, alpha, lr=1e-3, batch=64, ln=True,
                     return_params=False):
    """train(key, k) -> (best validation MSE [standardised units], best params or None).

    Trains a fresh MLP on D U Mix(D, k) with Adam; keeps the best-validation epoch.
    Can be jax.vmap-ed over (key, k) to train many models (with different policies) at once.
    """
    X, Y, Xv, Yv = (jnp.asarray(data[n]) for n in ("X", "Y", "Xv", "Yv"))
    N = X.shape[0]
    P = 2 * N
    S = -(-P // batch)
    pad = S * batch - P
    nn = jnp.asarray(nn_order)
    ar = jnp.arange(N)
    opt = optax.adam(lr)

    def build_pool(key, k):
        k1, k2, k3 = jax.random.split(key, 3)
        kk = jnp.maximum(k, 1)
        pos = jnp.minimum((jax.random.uniform(k1, (N,)) * kk).astype(jnp.int32), kk - 1)
        partner = nn[ar, pos]
        lam = jax.random.beta(k2, alpha, alpha, (N,))[:, None]
        xm = lam * X + (1 - lam) * X[partner]
        ym = lam * Y + (1 - lam) * Y[partner]
        valid = jnp.concatenate([jnp.ones(N), (k > 0).astype(jnp.float32), jnp.zeros(pad)])
        Xp = jnp.concatenate([X, xm, jnp.zeros((pad, X.shape[1]))])
        Yp = jnp.concatenate([Y, ym, jnp.zeros((pad, Y.shape[1]))])
        order = jnp.argsort(jax.random.uniform(k3, (S * batch,)) + 10.0 * (1 - valid))  # valid first
        return Xp[order], Yp[order], valid[order]

    def loss_fn(p, xb, yb, wb):
        se = ((forward(p, xb) - yb) ** 2).mean(-1)
        return (se * wb).sum() / jnp.maximum(wb.sum(), 1.0)

    def train(key, k):
        k0, key = jax.random.split(key)
        params = init_mlp(k0, sizes, ln)
        ostate = opt.init(params)

        def epoch(carry, ekey):
            params, ostate, best, bestp = carry
            Xp, Yp, valid = build_pool(ekey, k)
            nvalid = valid.sum()

            def step(c, s):
                p, o = c
                xb = jax.lax.dynamic_slice_in_dim(Xp, s * batch, batch)
                yb = jax.lax.dynamic_slice_in_dim(Yp, s * batch, batch)
                wb = jax.lax.dynamic_slice_in_dim(valid, s * batch, batch)
                g = jax.grad(loss_fn)(p, xb, yb, wb)
                upd, o2 = opt.update(g, o, p)
                p2 = optax.apply_updates(p, upd)
                act = s * batch < nvalid
                return (_select(act, p2, p), _select(act, o2, o)), None

            (params, ostate), _ = jax.lax.scan(step, (params, ostate), jnp.arange(S))
            vl = ((forward(params, Xv) - Yv) ** 2).mean()
            better = vl < best
            return (params, ostate, jnp.where(better, vl, best), _select(better, params, bestp)), vl

        init = (params, ostate, jnp.inf, params)
        (params, ostate, best, bestp), curve = jax.lax.scan(epoch, init, jax.random.split(key, epochs))
        return (best, bestp, curve) if return_params else (best, None, curve)

    return train


def make_manifold_trainer(data, sizes, epochs, alpha, lr=1e-3, batch=64, ln=True):
    """Manifold Mixup (Verma et al. 2019) baseline adapted to regression: per batch, mix the
    representations at a random layer (input included) and mix the targets the same way."""
    X, Y, Xv, Yv = (jnp.asarray(data[n]) for n in ("X", "Y", "Xv", "Yv"))
    N = X.shape[0]
    S = -(-N // batch)
    L = len(sizes) - 1
    opt = optax.adam(lr)

    def train(key):
        k0, key = jax.random.split(key)
        params = init_mlp(k0, sizes, ln)
        ostate = opt.init(params)

        def epoch(carry, ekey):
            params, ostate, best, bestp = carry
            ka, kb = jax.random.split(ekey)
            perm = jax.random.permutation(ka, N)
            idx = jnp.concatenate([perm, perm[: S * batch - N]])

            def step(c, s):
                p, o = c
                sk = jax.random.fold_in(kb, s)
                k1, k2, k3 = jax.random.split(sk, 3)
                ib = jax.lax.dynamic_slice_in_dim(idx, s * batch, batch)
                lam = jax.random.beta(k1, alpha, alpha)
                m = jax.random.randint(k2, (), 0, L)
                bperm = jax.random.permutation(k3, batch)
                xb, yb = X[ib], Y[ib]
                ymix = lam * yb + (1 - lam) * yb[bperm]

                def lf(pp):
                    return ((forward(pp, xb, (m, lam, bperm)) - ymix) ** 2).mean()

                g = jax.grad(lf)(p)
                upd, o = opt.update(g, o, p)
                return (optax.apply_updates(p, upd), o), None

            (params, ostate), _ = jax.lax.scan(step, (params, ostate), jnp.arange(S))
            vl = ((forward(params, Xv) - Yv) ** 2).mean()
            better = vl < best
            return (params, ostate, jnp.where(better, vl, best), _select(better, params, bestp)), vl

        init = (params, ostate, jnp.inf, params)
        (params, ostate, best, bestp), curve = jax.lax.scan(epoch, init, jax.random.split(key, epochs))
        return best, bestp, curve

    return train


# ----------------------------------------------------------------------------- RL controller
def ctrl_init(key, n, n_opt, hidden=(100, 100, 100, 100)):
    sizes = [n, *hidden, n * n_opt]
    ks = jax.random.split(key, len(sizes) - 1)
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        scale = 0.01 if i == len(sizes) - 2 else jnp.sqrt(2.0 / a)  # near-uniform initial policy
        ps.append({"w": jax.random.normal(ks[i], (a, b)) * scale, "b": jnp.zeros(b)})
    return ps


def ctrl_logits(params, n, n_opt):
    """MLP controller: constant input of 1's, one softmax output group per training example."""
    h = jnp.ones(n)
    for p in params[:-1]:
        h = jax.nn.relu(h @ p["w"] + p["b"])
    return (h @ params[-1]["w"] + params[-1]["b"]).reshape(n, n_opt)


def ppo_loss(params, n, n_opt, actions, old_logp, adv, clip=0.2, ent_w=0.01):
    logp_all = jax.nn.log_softmax(ctrl_logits(params, n, n_opt), -1)          # (N, n_opt)
    logp = jnp.take_along_axis(logp_all[None].repeat(actions.shape[0], 0), actions[..., None], -1)[..., 0]
    ratio = jnp.exp(logp - old_logp)                                            # (T, N) per-example ratios
    a = adv[:, None]
    surr = jnp.minimum(ratio * a, jnp.clip(ratio, 1 - clip, 1 + clip) * a).mean()
    ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
    return -(surr + ent_w * ent)


def train_regmix_controller(data, nn_order, sizes, epochs, alpha=100.0, T=20, iters=150,
                            ctrl_lr=2e-3, ppo_epochs=4, seed=0, log=print):
    """Algorithm 1 of the paper: sample T kNN policies, train T regression models on D U Mix(D,P)
    (in parallel via vmap), reward = 1/validation loss, update the controller with PPO."""
    N = data["X"].shape[0]
    n_opt = len(KNN_OPTIONS)
    opts = jnp.asarray(KNN_OPTIONS)
    train = make_aug_trainer(data, nn_order, sizes, epochs, alpha)
    batched = jax.jit(jax.vmap(lambda key, k: train(key, k)[0]))

    key = jax.random.PRNGKey(seed)
    key, kc = jax.random.split(key)
    cparams = ctrl_init(kc, N, n_opt)
    copt = optax.adam(ctrl_lr)
    cstate = copt.init(cparams)
    grad_fn = jax.jit(jax.value_and_grad(ppo_loss), static_argnums=(1, 2))
    logits_fn = jax.jit(ctrl_logits, static_argnums=(1, 2))
    delta, hist = 0.0, []

    for it in range(iters):
        t0 = time.time()
        key, ka, kt = jax.random.split(key, 3)
        logits = logits_fn(cparams, N, n_opt)
        logp_all = jax.nn.log_softmax(logits, -1)
        actions = jax.random.categorical(ka, logits, shape=(T, N))
        old_logp = jnp.take_along_axis(logp_all[None].repeat(T, 0), actions[..., None], -1)[..., 0]
        ks = opts[actions]
        losses = np.asarray(batched(jax.random.split(kt, T), ks))
        rewards = np.empty(T)
        for i, ls in enumerate(losses):                       # Alg. 1 lines 9-13
            rewards[i] = 1.0 / ls - delta
            delta = 0.95 * delta + 0.05 * (1.0 / ls)
        adv = jnp.asarray((rewards - rewards.mean()) / (rewards.std() + 1e-8), dtype=jnp.float32)
        for _ in range(ppo_epochs):                           # PPO update
            _, g = grad_fn(cparams, N, n_opt, actions, old_logp, adv)
            upd, cstate = copt.update(g, cstate, cparams)
            cparams = optax.apply_updates(cparams, upd)
        probs = np.asarray(jax.nn.softmax(logits_fn(cparams, N, n_opt), -1))
        hist.append(dict(it=it, mean_val_mse=float(losses.mean()), best_val_mse=float(losses.min()),
                         mean_k=float((probs * KNN_OPTIONS).sum(1).mean()),
                         entropy=float(-(probs * np.log(probs + 1e-12)).sum(1).mean())))
        if it % 5 == 0 or it == iters - 1:
            log(f"[iter {it:3d}] val_mse mean={losses.mean():.4f} min={losses.min():.4f} "
                f"E[k]={hist[-1]['mean_k']:.1f} H={hist[-1]['entropy']:.2f} ({time.time()-t0:.1f}s)")
    probs = np.asarray(jax.nn.softmax(logits_fn(cparams, N, n_opt), -1))
    return probs, hist
