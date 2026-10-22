"""Stage 2: train the RegMix MLP controller with PPO (Algorithm 1)."""
import json
import sys

import numpy as np

from regmix import *

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
data = load_data("data")
order, dist = neighbor_table(data["X"])
probs, hist = train_regmix_controller(
    data, order, sizes=[226, 256, 128, 4], epochs=40, alpha=100.0, T=20, iters=ITERS,
    log=lambda s: print(s, flush=True))
np.save("results/regmix_probs.npy", probs)
json.dump(hist, open("results/regmix_hist.json", "w"))
print("DONE controller", flush=True)
