#!/usr/bin/env python3

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


AUDIT_ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_ROOT",
        Path(__file__).resolve().parents[1]
    )
)
OUTPUT_JSON = (
    AUDIT_ROOT / "protocol/"
    "DETECTOR_SCORING_EQUIVALENCE_v1.json"
)

OUTPUT_TXT = (
    AUDIT_ROOT / "protocol/"
    "DETECTOR_SCORING_EQUIVALENCE_v1.txt"
)

CONFIGS = [
    {
        "backbone": "OpenCLIP",
        "input_dim": 512,
        "embeddings": Path(
            os.environ["TAMVLM_OPENCLIP_EMBEDDINGS"]
        ),
        "results_csv": Path(
            os.environ["TAMVLM_OPENCLIP_RESULTS_CSV"]
        ),
        "checkpoint_dir": Path(
            os.environ["TAMVLM_OPENCLIP_CHECKPOINT_DIR"]
        ),
        "checkpoint_pattern": "clip_mlp_seed{seed}.pt",
        "prediction_dir": Path(
            os.environ["TAMVLM_OPENCLIP_PREDICTION_DIR"]
        ),
    },
    {
        "backbone": "Qwen3-VL-Embedding-2B",
        "input_dim": 2048,
        "embeddings": Path(
            os.environ["TAMVLM_QWEN_EMBEDDINGS"]
        ),
        "results_csv": Path(
            os.environ["TAMVLM_QWEN_RESULTS_CSV"]
        ),
        "checkpoint_dir": Path(
            os.environ["TAMVLM_QWEN_CHECKPOINT_DIR"]
        ),
        "checkpoint_pattern": "qwen3vl_mlp_seed{seed}.pt",
        "prediction_dir": Path(
            os.environ["TAMVLM_QWEN_PREDICTION_DIR"]
        ),
    },
]

MAX_ALLOWED_ABSOLUTE_ERROR = 1e-6
BATCH_SIZE = 4096


class Detector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features)


def atomic_json_write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    os.replace(temporary, path)


def load_thresholds(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    return {
        int(row["seed"]): float(
            row["threshold_val_1pct_fpr"]
        )
        for row in rows
    }


def score_indices(
    model,
    embeddings,
    indices,
    device,
):
    indices = np.asarray(indices, dtype=np.int64)

    output = np.empty(
        len(indices),
        dtype=np.float32,
    )

    model.eval()

    with torch.inference_mode():
        for start in range(
            0,
            len(indices),
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                len(indices),
            )

            part = indices[start:stop]

            batch = torch.as_tensor(
                np.asarray(embeddings[part]),
                dtype=torch.float32,
                device=device,
            )

            output[start:stop] = (
                torch.sigmoid(
                    model(batch).squeeze(-1)
                )
                .cpu()
                .numpy()
            )

    return output


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

device = torch.device("cuda")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print("=" * 112)
print("Frozen detector scoring-equivalence audit")
print("=" * 112)
print("GPU:", torch.cuda.get_device_name(0))
print("Batch size:", BATCH_SIZE)
print("Comparator: score > frozen threshold")

records = []
failures = []

start_time = time.perf_counter()

for config in CONFIGS:
    print("\n" + "-" * 112)
    print(config["backbone"])
    print("-" * 112)

    for required_path in [
        config["embeddings"],
        config["results_csv"],
        config["checkpoint_dir"],
        config["prediction_dir"],
    ]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    features = np.load(
        config["embeddings"],
        mmap_mode="r",
    )

    if features.shape != (
        48633,
        config["input_dim"],
    ):
        raise RuntimeError(
            f"{config['backbone']}: unexpected "
            f"embedding shape {features.shape}"
        )

    thresholds = load_thresholds(
        config["results_csv"]
    )

    if set(thresholds) != set(range(5)):
        raise RuntimeError(
            f"{config['backbone']}: thresholds "
            "do not contain seeds 0–4"
        )

    for seed in range(5):
        checkpoint_path = (
            config["checkpoint_dir"] /
            config["checkpoint_pattern"].format(
                seed=seed
            )
        )

        prediction_path = (
            config["prediction_dir"] /
            f"main_predictions_seed{seed}.npz"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                checkpoint_path
            )

        if not prediction_path.is_file():
            raise FileNotFoundError(
                prediction_path
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = checkpoint.get(
            "state_dict"
        )

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Missing state_dict: "
                f"{checkpoint_path}"
            )

        model = Detector(
            input_dim=config["input_dim"],
            hidden_dim=128,
            dropout=0.2,
        ).to(device)

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        saved = np.load(prediction_path)

        val_indices = saved["val_indices"]
        test_indices = saved["test_indices"]

        saved_val_scores = saved[
            "val_scores"
        ].astype(np.float32)

        saved_test_scores = saved[
            "test_scores"
        ].astype(np.float32)

        reproduced_val_scores = score_indices(
            model,
            features,
            val_indices,
            device,
        )

        reproduced_test_scores = score_indices(
            model,
            features,
            test_indices,
            device,
        )

        val_error = np.abs(
            reproduced_val_scores -
            saved_val_scores
        )

        test_error = np.abs(
            reproduced_test_scores -
            saved_test_scores
        )

        maximum_absolute_error = float(
            max(
                val_error.max(),
                test_error.max(),
            )
        )

        mean_absolute_error = float(
            np.concatenate(
                [val_error, test_error]
            ).mean()
        )

        threshold = thresholds[seed]

        saved_alerts = np.concatenate([
            saved_val_scores > threshold,
            saved_test_scores > threshold,
        ])

        reproduced_alerts = np.concatenate([
            reproduced_val_scores > threshold,
            reproduced_test_scores > threshold,
        ])

        alert_mismatches = int(
            np.count_nonzero(
                saved_alerts !=
                reproduced_alerts
            )
        )

        passed = (
            maximum_absolute_error <=
            MAX_ALLOWED_ABSOLUTE_ERROR
            and alert_mismatches == 0
        )

        if not passed:
            failures.append(
                f"{config['backbone']} seed {seed}: "
                f"max error={maximum_absolute_error}, "
                f"alert mismatches={alert_mismatches}"
            )

        record = {
            "backbone": config["backbone"],
            "seed": seed,
            "input_dimension": (
                config["input_dim"]
            ),
            "threshold": threshold,
            "validation_rows": int(
                len(val_indices)
            ),
            "test_rows": int(
                len(test_indices)
            ),
            "maximum_absolute_error": (
                maximum_absolute_error
            ),
            "mean_absolute_error": (
                mean_absolute_error
            ),
            "alert_decision_mismatches": (
                alert_mismatches
            ),
            "status": (
                "PASS" if passed else "FAIL"
            ),
        }

        records.append(record)

        print(
            f"seed {seed} | "
            f"max_abs={maximum_absolute_error:.3e} | "
            f"mean_abs={mean_absolute_error:.3e} | "
            f"alert mismatches={alert_mismatches} | "
            f"{record['status']}"
        )

status = "PASS" if not failures else "FAIL"

payload = {
    "status": status,
    "audit": (
        "Frozen checkpoint scoring reproduction "
        "against saved validation/test predictions"
    ),
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0),
    "batch_size": BATCH_SIZE,
    "score_function": (
        "sigmoid(Detector(features))"
    ),
    "detector_architecture": (
        "input_dim -> 128 -> ReLU -> "
        "Dropout(0.2) -> 1"
    ),
    "inference_mode": "model.eval()",
    "alert_comparator": (
        "score > seed-specific frozen threshold"
    ),
    "maximum_allowed_absolute_error": (
        MAX_ALLOWED_ABSOLUTE_ERROR
    ),
    "elapsed_seconds": (
        time.perf_counter() - start_time
    ),
    "records": records,
    "failures": failures,
}

atomic_json_write(
    OUTPUT_JSON,
    payload,
)

report_lines = [
    "=" * 112,
    "Frozen detector scoring-equivalence audit",
    "=" * 112,
    f"Status                    : {status}",
    f"Detector heads checked    : {len(records)}",
    f"Total alert mismatches    : "
    f"{sum(r['alert_decision_mismatches'] for r in records)}",
    f"Maximum absolute error    : "
    f"{max(r['maximum_absolute_error'] for r in records):.8e}",
    f"Elapsed seconds           : "
    f"{payload['elapsed_seconds']:.2f}",
    f"Saved JSON                : {OUTPUT_JSON}",
]

if failures:
    report_lines.append("Failures:")

    for failure in failures:
        report_lines.append(
            f"  - {failure}"
        )

OUTPUT_TXT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("\n" + "\n".join(report_lines))

if status != "PASS":
    raise SystemExit(
        "Detector scoring-equivalence audit failed."
    )
