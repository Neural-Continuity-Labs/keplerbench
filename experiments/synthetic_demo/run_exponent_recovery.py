"""Record the exponents the adapted entry recovers at the largest budget.

Writes exponent_recovery.json: per hidden-exponent target world, the true
exponent and the exponent fitted from the first 128 training observations by
the free-exponent entry of run_transfer.py. Run from the repository root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from keplerbench.synthetic import generate_observations, sample_world  # noqa: E402
from keplerbench.transfer import _exposed_and_heldout  # noqa: E402
from learners import fit_central_force  # noqa: E402

HERE = Path(__file__).resolve().parent

records = []
for seed in range(200, 210):
    config = sample_world("hidden-exponent", seed=seed, sample_count=450, integration_substeps=120)
    rows = generate_observations(config, seed=seed)
    exposed, _ = _exposed_and_heldout(rows, 128)
    fit = fit_central_force(exposed, p=None, seed=0)
    records.append({"world_id": config.world_id, "p_true": config.force_exponent, "p_fitted": fit.p,
                    "abs_error": abs(fit.p - config.force_exponent)})
    print(records[-1], flush=True)

(HERE / "exponent_recovery.json").write_text(json.dumps({"k": 128, "worlds": records}, indent=2) + "\n", encoding="utf-8")
