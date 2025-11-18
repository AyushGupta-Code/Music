"""Download the Cardiff NLP emotion classifier locally."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "cardiffnlp/twitter-roberta-base-emotion"
DEFAULT_TARGET = Path(__file__).resolve().parents[1] / "hf_models" / "twitter-roberta-base-emotion"


def download(target: Path = DEFAULT_TARGET, repo_id: str = REPO_ID) -> Path:
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        local_dir_use_symlinks=False,
        resume_download=True,
        repo_type="model",
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Directory to place the model (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help="Hugging Face repo to download",
    )
    args = parser.parse_args()
    path = download(args.target, args.repo_id)
    print(f"✅ Downloaded {args.repo_id} to {path}")


if __name__ == "__main__":
    main()
