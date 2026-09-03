#!/usr/bin/env python3
"""Build the AI Feynman input files for the historical Mars pilot.

Reads the locally rebuilt epoch table (run `keplerbench fetch-historical` first)
and writes, for the training epochs only, the two inputs used in the paper:

* inputs/pilot_t_to_dec.txt        time in Julian years since the first training epoch -> declination (deg)
* inputs/pilot_phases_to_dec.txt   sidereal phase, synodic phase (radians, unwrapped) -> declination (deg)

and inputs/pilot_epochs.csv, the epoch table with those columns added, which the
scorer uses to evaluate every Pareto expression on the held-out epochs.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Mars's periods, the same prior the epicycle and ridge rows receive. The pilot's
# lattice uses 779.9361 d for the synodic period; the committed runs used 779.94 d.
P_SIDEREAL_DAYS, P_SYNODIC_DAYS = 686.980, 779.94
HERE = Path(__file__).resolve().parent
table = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "data" / "historical" / "mars_declinations.csv"

d = pd.read_csv(table)
d["t_years"] = (d.jd - d.jd.min()) / 365.25
d["phase_sid"] = 2 * np.pi * d.t_years / (P_SIDEREAL_DAYS / 365.25)
d["phase_syn"] = 2 * np.pi * d.t_years / (P_SYNODIC_DAYS / 365.25)
out = HERE / "inputs"; out.mkdir(exist_ok=True)
d.to_csv(out / "pilot_epochs.csv", index=False)
train = d[d.split == "train"]
np.savetxt(out / "pilot_t_to_dec.txt", np.column_stack([train.t_years, train.declination_deg]), fmt="%.8f")
np.savetxt(out / "pilot_phases_to_dec.txt", np.column_stack([train.phase_sid, train.phase_syn, train.declination_deg]), fmt="%.8f")
print(f"{len(train)} training epochs written to {out}")
