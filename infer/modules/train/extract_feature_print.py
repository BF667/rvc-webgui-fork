import os
import sys
import traceback
from pathlib import Path
from typing import Literal

from tap import Tap
from loguru import logger

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

BoolString = Literal["True", "False", "true", "false", "1", "0"]


def parse_bool(value: BoolString) -> bool:
    return value.lower() in {"true", "1"}


class ExtractFeatureCpuArgs(Tap):
    # Requested device.
    device: str
    # Experiment directory.
    exp_dir: Path
    # Model version.
    version: str
    # Whether to use half precision.
    is_half: BoolString

    def configure(self) -> None:
        self.add_argument("device")
        self.add_argument("exp_dir")
        self.add_argument("version")
        self.add_argument("is_half")


class ExtractFeatureGpuArgs(Tap):
    # Requested device.
    device: str
    # GPU ID assigned to this worker.
    i_gpu: str
    # Experiment directory.
    exp_dir: Path
    # Model version.
    version: str
    # Whether to use half precision.
    is_half: BoolString

    def configure(self) -> None:
        self.add_argument("device")
        self.add_argument("i_gpu")
        self.add_argument("exp_dir")
        self.add_argument("version")
        self.add_argument("is_half")


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parsed_args = ExtractFeatureGpuArgs().parse_args()
elif len(sys.argv) == 5:
    parsed_args = ExtractFeatureCpuArgs().parse_args()
elif len(sys.argv) == 6:
    parsed_args = ExtractFeatureGpuArgs().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed_args.i_gpu)
else:
    raise ValueError("Expected positional arguments: device [i_gpu] exp_dir version is_half")
exp_dir = parsed_args.exp_dir
version = parsed_args.version
is_half = parse_bool(parsed_args.is_half)
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

logger.remove()
logger.add(
    exp_dir / "extract_f0_feature.log",
    level="INFO",
    serialize=True,
    enqueue=True,
    backtrace=False,
    diagnose=False,
)


logger.bind(event="feature_args", argv=sys.argv[1:]).info("Received feature extraction args")
model_path = "assets/hubert/hubert_base.pt"

logger.info(f"Feature extraction output directory: {exp_dir}")
wavPath = exp_dir / "1_16k_wavs"
outPath = exp_dir / "3_feature256" if version == "v1" else exp_dir / "3_feature768"
outPath.mkdir(parents=True, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
logger.info(f"Loading HuBERT model from {model_path}")
# if hubert model is exist
if not os.access(model_path, os.F_OK):
    logger.error(
        f"Feature extraction stopped because {model_path} does not exist. Download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
    )
    exit(0)

from fairseq.data.dictionary import Dictionary
from torch.serialization import safe_globals

with safe_globals([Dictionary]):
    # torch.serialization.add_safe_globals([Dictionary])
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
model = models[0]
model = model.to(device)
logger.info(f"Moved HuBERT model to {device}")
if is_half:
    if device not in ["mps", "cpu"]:
        model = model.half()
model.eval()

todo = sorted(wavPath.iterdir(), key=lambda p: p.name)
if len(todo) == 0:
    logger.bind(event="feature_empty", total=0).info("No features to extract")
else:
    logger.bind(event="feature_started", total=len(todo), version=version).info(
        "Starting feature extraction"
    )
    if saved_cfg is None:
        raise RuntimeError("HuBERT checkpoint did not include a saved config")
    normalize = saved_cfg.task.normalize
    for idx, file in enumerate(todo):
        try:
            if file.suffix == ".wav":
                wav_path = wavPath / file.name
                out_path = outPath / file.with_suffix(".npy").name

                if out_path.exists():
                    continue

                feats = readwave(wav_path, normalize=normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (
                        feats.half().to(device)
                        if is_half and device not in ["mps", "cpu"]
                        else feats.to(device)
                    ),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9 if version == "v1" else 12,  # layer 9
                }
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats = (
                        model.final_proj(logits[0]) if version == "v1" else logits[0]
                    )

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    logger.warning(f"{file.name} contains NaN values")
                logger.bind(
                    event="ui_progress",
                    detail_event="feature_progress",
                    stage="extract_feature",
                    current=idx + 1,
                    total=len(todo),
                    fraction=(idx + 1) / max(len(todo), 1),
                    message=f"Extracting features {idx + 1}/{len(todo)}: {file.name}",
                    file=file.name,
                    output_shape=list(feats.shape),
                ).info(f"Extracted features for {file.name}")
        except Exception:
            logger.bind(
                event="ui_progress",
                detail_event="feature_failed",
                stage="extract_feature",
                current=idx + 1,
                total=len(todo),
                fraction=(idx + 1) / max(len(todo), 1),
                message=f"Feature extraction failed at {idx + 1}/{len(todo)}: {file.name}",
                file=file.name,
                traceback=traceback.format_exc(),
            ).exception(f"Failed feature extraction for {file.name}")
    logger.bind(event="feature_finished", total=len(todo)).info(
        "Finished feature extraction"
    )
