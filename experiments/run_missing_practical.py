"""Complete missing practical-width all-grid levels for the current setups."""
import json
from pathlib import Path
import numpy as np
from experiments.matched_allgrid import run, spacing
from experiments.swapped_low_order import LAPLACE_KITE, STOKES_STARFISH

for label, problem in (("laplace", LAPLACE_KITE), ("stokes", STOKES_STARFISH)):
    for m in (64, 768):
        dest = Path(f"results/paper_revision/allgrid_practical/{label}")
        dest.mkdir(parents=True, exist_ok=True)
        h = (problem.box[1] - problem.box[0]) / (m + 1)
        n = spacing(problem.geometry, h)["n_gamma"]
        report, fields = run(problem, m, kappa=2, delta_override=2*h, n_override=n)
        error_names = ("error",) if label == "laplace" else ("pressure_error", "velocity_error")
        for key in error_names:
            values = fields[key]
            mag = np.linalg.norm(values, axis=-1) if values.ndim == 3 else abs(values)
            ij = np.unravel_index(np.argmax(mag), mag.shape)
            report[f"{key}_maximum_node"] = dict(index=list(map(int, ij)),
                point=[float(fields["xx"][ij]), float(fields["yy"][ij])],
                error=float(mag[ij]), in_direct_halo=bool(fields["halo"][ij]))
        (dest / f"M{m}.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(dest / f"fields_M{m}.npz", **fields)
        print(label, m, {key: report[key] for key in (("laplace",) if label == "laplace" else ("pressure", "velocity"))}, flush=True)
