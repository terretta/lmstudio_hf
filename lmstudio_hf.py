# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "huggingface_hub>=0.20",
# ]
# ///
import argparse
import json
import os
from pathlib import Path
import sys
import shutil

def select_models(model_choices):
    n = len(model_choices)
    if n == 0:
        return []
    selected = [False] * n
    idx = 0  # row 0 = "Select all", rows 1..n = model rows
    total_rows = n + 1
    window_size = os.get_terminal_size().lines - 5

    while True:
        print("\033[H\033[J", end="")
        print("❯ lm-studio - Hugging Face Manage models \nAvailable models (↑/↓ to navigate, SPACE to select, ENTER to confirm, Ctrl+C to quit):")

        all_state = all(selected)
        window_start = max(0, min(idx - window_size + 3, total_rows - window_size))
        window_end = min(window_start + window_size, total_rows)

        for i in range(window_start, window_end):
            cursor = '>' if i == idx else ' '
            if i == 0:
                marker = '◉' if all_state else '○'
                label = 'Select all'
            else:
                marker = '◉' if selected[i - 1] else '○'
                label = model_choices[i - 1][0]
            print(f"{cursor} {marker} {label}")

        key = get_key()
        if key == "\x1b[A":  # Up arrow
            idx = max(0, idx - 1)
        elif key == "\x1b[B":  # Down arrow
            idx = min(total_rows - 1, idx + 1)
        elif key == " ":
            if idx == 0:
                new_state = not all(selected)
                selected = [new_state] * n
            else:
                selected[idx - 1] = not selected[idx - 1]
        elif key == "\r":  # Enter key
            break
        elif key == "\x03":  # Ctrl+C
            print("\nImport is cancelled. Do nothing.")
            sys.exit(0)

    return [choice for choice, is_selected in zip(model_choices, selected) if is_selected]

def get_key():
    """Get a single keypress from the user."""
    import tty, termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def manage_models():
    "Import MLX models from the Hugging Face cache."
    cache_dir = Path(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    )
    lm_studio_dir = Path(os.path.expanduser("~/.cache/lm-studio/models"))

    found_models = set()
    for root, dirs, _ in os.walk(cache_dir):
        for d in dirs:
            if "mlx-community" in d:
                model_dir = Path(root) / d
                snapshots_dir = model_dir / "snapshots"
                if not snapshots_dir.exists():
                    continue

                # Search for config.json in any subfolder under snapshots
                config_found = False
                snapshot_path = None
                for config_root, _, config_files in os.walk(snapshots_dir):
                    if "config.json" in config_files:
                        config_path = Path(config_root) / "config.json"
                        try:
                            with open(config_path) as f:
                                config = json.load(f)
                                model_type = config.get("model_type", "").lower()
                                config_found = True
                                snapshot_path = Path(config_root)
                                break
                        except (json.JSONDecodeError, FileNotFoundError):
                            continue

                if not config_found or not snapshot_path:
                    continue

                parts = d.split("--")
                model_name = "/".join(parts[1:])
                if model_name:
                    # Store model_type, model_name, and snapshot_path
                    found_models.add((model_type, model_name, snapshot_path))

    if not found_models:
        print("No MLX models found in Hugging Face cache")
        return

    # Create list of models with their current import status
    model_choices = []
    sorted_models = sorted(found_models)
    type_w = max((len(f"({mt})") for mt, _, _ in sorted_models), default=0)
    for model_type, model, snapshot_path in sorted_models:
        target_path = lm_studio_dir / f"{model}"
        is_imported = target_path.exists()
        status = " (already imported)" if is_imported else ""
        display_name = f"{f'({model_type})'.ljust(type_w)} {model}{status}"
        model_choices.append((display_name, model, is_imported, snapshot_path))

    # Show interactive selection menu
    selected = select_models(model_choices)
    print("\nImporting models...\n")

    for display_name, model_name, is_imported, snapshot_path in selected:
        target_path = lm_studio_dir / f"{model_name}"

        if is_imported:
            # Remove existing directory or symlink
            if target_path.is_symlink() or target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            print(f"Removed {model_name}")

        else:
            # Create parent directories and target directory
            target_path.mkdir(parents=True, exist_ok=True)

            # Create symbolic links for all files in the snapshot directory
            for item in snapshot_path.iterdir():
                link_path = target_path / item.name
                os.symlink(item, link_path)

            print(f"Imported {model_name} (symlinked files)")

def detect_model_type(model_dir):
    """Return 'mlx', 'gguf', or 'other' based on weight files in the directory."""
    has_safetensors = False
    has_gguf = False
    for entry in model_dir.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.suffix == ".safetensors":
            has_safetensors = True
        elif entry.suffix == ".gguf":
            has_gguf = True
    if has_safetensors:
        return "mlx"
    if has_gguf:
        return "gguf"
    return "other"

def is_already_symlinked(model_dir):
    """True if every non-dotfile entry in model_dir is a symlink."""
    entries = [e for e in model_dir.iterdir() if not e.name.startswith(".")]
    if not entries:
        return False
    return all(e.is_symlink() for e in entries)

def resolve_hf_snapshot(hub_dir, publisher, name):
    """Return the active HF snapshot path for <publisher>/<name>, or None if not cached."""
    model_root = hub_dir / f"models--{publisher}--{name}"
    ref = model_root / "refs" / "main"
    if not ref.exists():
        return None
    sha = ref.read_text().strip()
    snapshot_path = model_root / "snapshots" / sha
    return snapshot_path if snapshot_path.exists() else None

def scan_lmstudio_models(lm_studio_dir):
    """Yield (publisher, name, model_dir, model_type) for each MLX/GGUF model in LM Studio."""
    results = []
    if not lm_studio_dir.exists():
        return results
    for publisher_dir in sorted(lm_studio_dir.iterdir()):
        if not publisher_dir.is_dir() or publisher_dir.name.startswith("."):
            continue
        for model_dir in sorted(publisher_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            mtype = detect_model_type(model_dir)
            if mtype == "other":
                continue
            results.append((publisher_dir.name, model_dir.name, model_dir, mtype))
    return results

def replace_with_symlink_tree(model_dir, snapshot_path):
    """Replace model_dir's contents with per-file symlinks pointing into snapshot_path."""
    backup = model_dir.with_name(model_dir.name + ".old")
    if backup.exists():
        raise RuntimeError(
            f"Backup directory {backup} already exists from a prior failed run; "
            f"remove it manually before re-running."
        )
    model_dir.rename(backup)
    try:
        model_dir.mkdir(parents=True, exist_ok=False)
        for item in snapshot_path.iterdir():
            if item.name.startswith("."):
                continue
            os.symlink(item, model_dir / item.name)
    except Exception:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        backup.rename(model_dir)
        raise
    shutil.rmtree(backup)

def mirror_to_huggingface(types):
    """Scan LM Studio, ensure each model is in the HF cache, replace with symlinks.

    `types` is a set of model type strings to include, e.g. {"mlx"} or {"mlx", "gguf"}.
    """
    cache_dir = Path(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    )
    hub_dir = cache_dir / "hub"
    lm_studio_dir = Path(os.path.expanduser("~/.cache/lm-studio/models"))

    if not lm_studio_dir.exists():
        print(f"No LM Studio models directory at {lm_studio_dir}")
        return

    candidates = [m for m in scan_lmstudio_models(lm_studio_dir) if m[3] in types]
    if not candidates:
        print(f"No LM Studio models matching types: {sorted(types)}")
        return

    from huggingface_hub import HfApi
    from huggingface_hub.utils import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )
    api = HfApi()

    classified = []
    print(f"Checking {len(candidates)} model(s) against Hugging Face Hub...")
    for publisher, name, model_dir, mtype in candidates:
        if is_already_symlinked(model_dir):
            classified.append((publisher, name, model_dir, mtype, "already_symlinked", None))
            continue
        snapshot_path = resolve_hf_snapshot(hub_dir, publisher, name)
        if snapshot_path is not None:
            classified.append((publisher, name, model_dir, mtype, "in_hf_cache", snapshot_path))
            continue
        repo_id = f"{publisher}/{name}"
        try:
            api.model_info(repo_id)
            classified.append((publisher, name, model_dir, mtype, "needs_download", None))
        except (RepositoryNotFoundError, GatedRepoError):
            classified.append((publisher, name, model_dir, mtype, "not_on_hub", None))
        except HfHubHTTPError as e:
            print(f"  Hub error for {repo_id}: {e}")
            classified.append((publisher, name, model_dir, mtype, "not_on_hub", None))
        except Exception as e:
            print(f"  Unexpected error checking {repo_id}: {e}")
            classified.append((publisher, name, model_dir, mtype, "not_on_hub", None))

    counts = {}
    for c in classified:
        counts[c[4]] = counts.get(c[4], 0) + 1
    if counts.get("already_symlinked"):
        print(f"{counts['already_symlinked']} already mirrored — skipped")
    if counts.get("not_on_hub"):
        print(f"{counts['not_on_hub']} not on Hugging Face Hub — skipped:")
        for c in classified:
            if c[4] == "not_on_hub":
                print(f"  - {c[0]}/{c[1]}")

    actionable = [c for c in classified if c[4] in ("in_hf_cache", "needs_download")]
    if not actionable:
        print("\nNo actionable models. Exiting.")
        return

    model_choices = []
    type_w = max((len(f"({c[3]})") for c in actionable), default=0)
    for publisher, name, model_dir, mtype, status, snapshot_path in actionable:
        label = "in HF cache, will symlink" if status == "in_hf_cache" else "will download from HF"
        display_name = f"{f'({mtype})'.ljust(type_w)} {publisher}/{name} [{label}]"
        model_choices.append((display_name, publisher, name, model_dir, status, snapshot_path))

    selected = select_models(model_choices)
    if not selected:
        print("\nNo models selected. Exiting.")
        return

    print("\nMirroring models...\n")
    from huggingface_hub import snapshot_download
    for _, publisher, name, model_dir, status, snapshot_path in selected:
        repo_id = f"{publisher}/{name}"
        try:
            if status == "needs_download":
                print(f"Downloading {repo_id} ...")
                snapshot_path = Path(snapshot_download(repo_id=repo_id, cache_dir=str(hub_dir)))
            replace_with_symlink_tree(model_dir, snapshot_path)
            print(f"Mirrored {repo_id}")
        except Exception as e:
            print(f"  Failed to mirror {repo_id}: {e}")
            continue

def main():
    """Entry point: dispatch the import (default) or mirror subcommand."""
    parser = argparse.ArgumentParser(prog="lmstudio-hf")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("import", help="Import MLX models from HF cache into LM Studio (default).")
    m = sub.add_parser("mirror", help="Mirror LM Studio models into HF cache and replace with symlinks.")
    m.add_argument(
        "--type",
        choices=["mlx", "gguf", "both"],
        default="mlx",
        help="Model types to mirror (default: mlx, matching the import flow).",
    )
    args = parser.parse_args()
    if args.cmd == "mirror":
        types = {"mlx", "gguf"} if args.type == "both" else {args.type}
        mirror_to_huggingface(types)
    else:
        manage_models()

if __name__ == "__main__":
    main()
