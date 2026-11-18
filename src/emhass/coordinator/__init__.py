"""The coordinated backend (optimization_backend: dantzig_wolfe).

Each device, or group of devices, is planned on its own by EMHASS's own model
(a participant); a Dantzig-Wolfe coordinator makes the plans agree at the
meter and at every node and limit of `site`, and splits the saving.

- `interface`: the query a participant answers, and its answer.
- `backend`: EMHASS's devices as participants, and `optimize`, which
  perform_optimization calls.
- `master`: the coordinator (the master LP, pricing, recovery).
- `model`, `submeter`, `meter`, `costs`: the site and how it is scored.
- `settlement`: each player's share of the saving.
- `site`: the `site` list, checked and compiled into EMHASS's own keys.
- `lpsolver`: the master's numpy LP.

numpy and scipy only. Adapted, MIT-licensed, from home-energy-optimizer
(https://github.com/ameetdesh/home-energy-optimizer), by its author.
"""
