"""Machine-readable result schemas shared by drivers and plotting scripts."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path


LOCAL_FIELDS = ("geometry", "solution", "n_gamma", "h_gamma", "order", "m",
                "q", "delta", "kappa", "lambda", "side", "error_inf",
                "error_l2", "oversampled_error_inf")
ALLGRID_FIELDS = ("geometry", "solution", "m_grid", "h_c", "n_gamma",
                  "order", "m", "q", "delta", "kappa", "direct_targets",
                  "error_inf", "error_l2", "gradient_error_inf",
                  "gradient_error_l2", "oversampled_error_inf",
                  "bie_seconds", "dst_seconds", "total_seconds")


def write_rows(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_metadata(path, *, experiment, configuration, error_floor=None,
                   commit=None):
    payload = {"experiment": experiment, "configuration": configuration,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "code_commit": commit, "plot_error_floor": error_floor}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))
