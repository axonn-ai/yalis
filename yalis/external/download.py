# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

from config import configs

# from litgpt.scripts.convert_hf_checkpoint import convert_hf_checkpoint
from convert_hf_checkpoint import convert_hf_checkpoint

from lightning_utilities.core.imports import RequirementCache
import sys

_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")
_HF_TRANSFER_AVAILABLE = RequirementCache("hf_transfer")


def download_from_hub(
    repo_id: str,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    tokenizer_only: bool = False,
    convert_checkpoint: bool = True,
    dtype: Optional[str] = None,
    checkpoint_dir: Path = Path(os.getenv("YALIS_CACHE", "~/.cache/yalis"))
    / "checkpoints",
    model_name: Optional[str] = None,
) -> None:
    """Download weights or tokenizer data from the Hugging Face Hub.

    Arguments:
        repo_id: The repository ID in the format ``org/name`` or ``user/name`` as shown in Hugging Face.
            If "list" is provided as input, a list of the currently supported models in LitGPT and quits.
        access_token: Optional API token to access models with restrictions.
        tokenizer_only: Whether to download only the tokenizer files.
        convert_checkpoint: Whether to convert the checkpoint files to the LitGPT format after downloading.
        dtype: The data type to convert the checkpoint files to. If not specified, the weights will remain in the
            dtype they are downloaded in.
        checkpoint_dir: Where to save the downloaded files.
        model_name: The existing config name to use for this repo_id. This is useful to download alternative weights of
            existing architectures.
    """
    options = [
        f"{config['hf_config']['org']}/{config['hf_config']['name']}"
        for config in configs
    ]

    if repo_id == "list":
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(sorted(options, key=lambda x: x.lower())))
        return

    if model_name is None and repo_id not in options:
        print(
            f"Unsupported `repo_id`: {repo_id}."
            "\nIf you are trying to download alternative "
            "weights for a supported model, please specify the corresponding model via the `--model_name` option, "
            "for example, `litgpt download NousResearch/Hermes-2-Pro-Llama-3-8B --model_name Llama-3-8B`."
            "\nAlternatively, please choose a valid `repo_id` from the list of supported models, which can be obtained via "
            "`litgpt download list`."
        )
        return

    from huggingface_hub import snapshot_download

    if importlib.util.find_spec("hf_transfer") is None:
        print(
            "It is recommended to install hf_transfer for faster checkpoint download speeds: `pip install hf_transfer`"
        )

    download_files = ["tokenizer*", "generation_config.json", "config.json"]
    if not tokenizer_only:
        bins, safetensors = find_weight_files(repo_id, access_token)
        if bins:
            # covers `.bin` files and `.bin.index.json`
            download_files.append("*.bin*")
        elif safetensors:
            if not _SAFETENSORS_AVAILABLE:
                raise ModuleNotFoundError(str(_SAFETENSORS_AVAILABLE))
            download_files.append("*.safetensors*")
        else:
            raise ValueError(f"Couldn't find weight files for {repo_id}")

    import huggingface_hub._snapshot_download as download
    import huggingface_hub.constants as constants

    previous = constants.HF_HUB_ENABLE_HF_TRANSFER
    if _HF_TRANSFER_AVAILABLE and not previous:
        print("Setting HF_HUB_ENABLE_HF_TRANSFER=1")
        constants.HF_HUB_ENABLE_HF_TRANSFER = True
        download.HF_HUB_ENABLE_HF_TRANSFER = True

    directory = checkpoint_dir / repo_id
    with gated_repo_catcher(repo_id, access_token):
        snapshot_download(
            repo_id,
            local_dir=directory,
            allow_patterns=download_files,
            token=access_token,
        )

    constants.HF_HUB_ENABLE_HF_TRANSFER = previous
    download.HF_HUB_ENABLE_HF_TRANSFER = previous

    def _find_weights_dir(root: Path) -> Path:
        """Find the directory under `root` that contains HF weight files.

        The function checks the root, then immediate subdirectories, then one
        level deeper for files like `*.bin`, `*.safetensors` or index jsons
        (e.g., `pytorch_model.bin.index.json`, `model.safetensors.index.json`).
        """
        def _has_weights(p: Path) -> bool:
            if (p / "pytorch_model.bin.index.json").is_file() or (
                p / "model.safetensors.index.json"
            ).is_file():
                return True
            # look for .bin or .safetensors files (ignore training_args.bin)
            for ext in ("*.bin", "*.safetensors"):
                for f in p.glob(ext):
                    if f.is_file() and f.name != "training_args.bin":
                        return True
            return False

        if _has_weights(root):
            return root

        # check immediate subdirectories
        for child in sorted(root.iterdir()):
            if child.is_dir() and _has_weights(child):
                return child

        # check one level deeper
        for child in sorted(root.iterdir()):
            if child.is_dir():
                for subchild in sorted(child.iterdir()):
                    if subchild.is_dir() and _has_weights(subchild):
                        return subchild

        # not found; return the original root so the converter can raise a
        # meaningful error if necessary
        return root

    if convert_checkpoint and not tokenizer_only:
        print("Converting checkpoint files to LitGPT format.")
        resolved_dir = _find_weights_dir(directory)
        if resolved_dir != directory:
            print(f"Detected weight files in nested directory: {resolved_dir}")
        # If user did not provide an explicit model_name, prefer the top-level
        # directory name (e.g., 'gpt-oss-20b') rather than the nested folder name
        # (e.g., 'metal') which is not a model config name.
        local_model_name = model_name or directory.name
        print(f'Using model_name = "{local_model_name}"')
        convert_hf_checkpoint(
            checkpoint_dir=resolved_dir, dtype=dtype, model_name=local_model_name
        )

def find_weight_files(
    repo_id: str, access_token: Optional[str]
) -> Tuple[List[str], List[str]]:
    from huggingface_hub import repo_info
    from huggingface_hub.utils import filter_repo_objects

    with gated_repo_catcher(repo_id, access_token):
        info = repo_info(repo_id, token=access_token)
    filenames = [f.rfilename for f in info.siblings]
    bins = list(
        filter_repo_objects(items=filenames, allow_patterns=["*model*.bin*"])
    )
    safetensors = list(
        filter_repo_objects(items=filenames, allow_patterns=["*.safetensors*"])
    )
    return bins, safetensors


@contextmanager
def gated_repo_catcher(repo_id: str, access_token: Optional[str]):
    try:
        yield
    except OSError as e:
        err_msg = str(e)
        if "Repository Not Found" in err_msg:
            raise ValueError(
                f"Repository at https://huggingface.co/api/models/{repo_id} not found."
                " Please make sure you specified the correct `repo_id`."
            ) from None
        elif "gated repo" in err_msg:
            if not access_token:
                raise ValueError(
                    f"https://huggingface.co/{repo_id} requires"
                    "authentication, please set the `HF_TOKEN=your_token`"
                    " environment variable or pass "
                    "`--access_token=your_token`. You can find your token by visiting"
                    " https://huggingface.co/settings/tokens."
                ) from None
            else:
                raise ValueError(
                    f"https://huggingface.co/{repo_id} requires authentication. The access token provided by `HF_TOKEN=your_token`"
                    " environment variable or `--access_token=your_token` may not have sufficient access rights. Please"
                    f" visit https://huggingface.co/{repo_id} for more information."
                ) from None
        raise e from None


if __name__ == "__main__":
    repo_id = sys.argv[1]
    download_from_hub(repo_id)
