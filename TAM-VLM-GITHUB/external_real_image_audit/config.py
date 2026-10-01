from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent


DATA_ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_DATA",
        PROJECT_ROOT / "data"
    )
)


RESULTS_ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_RESULTS",
        PROJECT_ROOT / "results"
    )
)


MODEL_ROOT = Path(
    os.environ.get(
        "TAMVLM_MODEL_ROOT",
        PROJECT_ROOT / "models"
    )
)


NUSCENES_ROOT = Path(
    os.environ.get(
        "TAMVLM_NUSCENES_ROOT",
        DATA_ROOT / "nuscenes"
    )
)


OPENCLIP_EMB_ROOT = Path(
    os.environ.get(
        "TAMVLM_OPENCLIP_EMB_ROOT",
        DATA_ROOT / "openclip_embeddings"
    )
)


QWEN_EMB_ROOT = Path(
    os.environ.get(
        "TAMVLM_QWEN_EMB_ROOT",
        DATA_ROOT / "qwen_embeddings"
    )
)
