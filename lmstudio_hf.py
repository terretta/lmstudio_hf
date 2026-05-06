# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "huggingface_hub>=0.20",
# ]
# ///
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
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
    """Get a single keypress from the user.

    Handles single-byte keys plus common ANSI escape sequences:
      \\x1b[A/B/C/D   arrow keys (3 bytes)
      \\x1b[3~        Delete / Forward Delete (4 bytes)
      \\x1b[5~/[6~    PageUp / PageDown (4 bytes)
      \\x1b[H / [F    Home / End (3 bytes)

    Note: bare ESC is NOT supported as a distinct key. Disambiguating bare
    ESC from the start of an escape sequence requires a timeout, which
    risks misclassifying a slow-arriving arrow-key sequence as bare ESC
    on SSH or under load. Arrow keys must be 100% reliable, so we always
    read at least 2 follow-up bytes after ESC. Use Q (or Ctrl-C) as the
    cancel key in interactive pickers.

    After ESC + '[', if the third byte is a digit, keep reading until we
    hit a non-digit terminator (~, A-Z).
    """
    import tty, termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
            if len(ch) == 3 and ch[-1].isdigit():
                while True:
                    nxt = sys.stdin.read(1)
                    ch += nxt
                    if not nxt.isdigit():
                        break
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

def resolve_lm_studio_models_dir():
    """Locate LM Studio's models/ directory cross-platform.

    Honors the ~/.lmstudio-home-pointer redirect file (a plaintext one-liner
    holding the LM Studio home dir) used on macOS, Linux, and Windows alike.
    Falls back to the platform-default ~/.lmstudio/ when no pointer exists.
    """
    pointer = Path(os.path.expanduser("~/.lmstudio-home-pointer"))
    if pointer.exists():
        home = Path(pointer.read_text().strip()).expanduser()
    else:
        home = Path(os.path.expanduser("~/.lmstudio"))
    return home / "models"

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

def sha256_file(path, chunk_size=1 << 20):
    """Stream sha256 of a file in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _lfs_sha256(sibling):
    """Extract sha256 from a sibling's LFS metadata, tolerating dict or object form."""
    lfs = getattr(sibling, "lfs", None)
    if not lfs:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)

def ingest_local_files_into_hf_cache(hub_dir, publisher, name, model_dir, repo_info):
    """Move LFS-eligible local files into the HF cache blob layout when sha256 matches.

    Returns (snapshot_path, ingested_count). Files whose hash doesn't match (or that
    have no LFS hash to verify against) are left in model_dir; a subsequent
    snapshot_download() will fetch what's missing. Mutates model_dir by removing
    successfully-ingested files (their canonical home is now the HF blob store).
    """
    sha = repo_info.sha
    target_root = hub_dir / f"models--{publisher}--{name}"
    blobs_dir = target_root / "blobs"
    snapshots_dir = target_root / "snapshots"
    refs_dir = target_root / "refs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshots_dir / sha
    snapshot_path.mkdir(parents=True, exist_ok=True)

    expected = {}
    for sib in repo_info.siblings or []:
        size = getattr(sib, "size", None)
        sha256 = _lfs_sha256(sib)
        expected[sib.rfilename] = (size, sha256)

    ingested = 0
    for entry in sorted(model_dir.iterdir()):
        if entry.is_dir() or entry.is_symlink() or entry.name.startswith("."):
            continue
        if entry.name not in expected:
            continue
        exp_size, exp_sha = expected[entry.name]
        if exp_sha is None:
            continue  # not an LFS file; let snapshot_download fetch it
        local_size = entry.stat().st_size
        if exp_size is not None and exp_size != local_size:
            continue

        blob_path = blobs_dir / exp_sha
        snapshot_link = snapshot_path / entry.name
        rel_target = os.path.relpath(blob_path, snapshot_path)

        if blob_path.exists():
            if not snapshot_link.exists():
                os.symlink(rel_target, snapshot_link)
            entry.unlink()
            ingested += 1
            continue

        gb = local_size / 1e9
        print(f"  Hashing {entry.name} ({gb:.1f} GB)...", flush=True)
        actual = sha256_file(entry)
        if actual != exp_sha:
            print(f"    sha mismatch — will re-download", flush=True)
            continue
        entry.rename(blob_path)
        if snapshot_link.exists():
            snapshot_link.unlink()
        os.symlink(rel_target, snapshot_link)
        ingested += 1
        print(f"    matched, ingested as blob {exp_sha[:12]}", flush=True)

    (refs_dir / "main").write_text(sha)
    return snapshot_path, ingested

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

def mirror_to_huggingface(types, reuse_local=True):
    """Scan LM Studio, ensure each model is in the HF cache, replace with symlinks.

    `types` is a set of model type strings to include, e.g. {"mlx"} or {"mlx", "gguf"}.
    `reuse_local` enables sha256-verified ingestion of LM Studio's existing files
    into the HF cache blob layout before downloading, so matching files are reused
    instead of re-fetched.
    """
    cache_dir = Path(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    )
    hub_dir = cache_dir / "hub"
    lm_studio_dir = resolve_lm_studio_models_dir()

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
        print(f"\n[{repo_id}]", flush=True)
        try:
            if status == "needs_download":
                if reuse_local:
                    try:
                        repo_info = api.model_info(repo_id, files_metadata=True)
                        snapshot_path, ingested = ingest_local_files_into_hf_cache(
                            hub_dir, publisher, name, model_dir, repo_info
                        )
                        if ingested:
                            print(f"  Ingested {ingested} local file(s) into HF blob store", flush=True)
                    except Exception as e:
                        print(f"  Could not ingest locally ({e}); falling back to full download", flush=True)
                print(f"  Downloading {repo_id} (skipping any blobs already cached)...", flush=True)
                snapshot_path = Path(snapshot_download(repo_id=repo_id, cache_dir=str(hub_dir)))
            replace_with_symlink_tree(model_dir, snapshot_path)
            print(f"\n  Mirrored {repo_id}", flush=True)
        except Exception as e:
            print(f"\n  Failed to mirror {repo_id}: {e}", flush=True)
            continue

# ---------------------------------------------------------------------------
# discover: read-only inventory of HF-compatible models cached by other apps
# ---------------------------------------------------------------------------

# Minimum file size for a "real" model weight; smaller files are placeholders,
# stubs (iCloud/Dropbox cloud-only), or metadata.
_DISCOVER_MIN_BYTES = 10 * 1024 * 1024

# Bundle ids of sandboxed apps whose container paths we want to label nicely
# when they show up in mdfind results.
_KNOWN_BUNDLE_IDS = {
    "com.liuliu.draw-things": "Draw Things",
    "com.numericcal.privatellm": "Private LLM",
}

# Substrings used by the path-pattern classifier to recognize tool-named dirs.
_TOOL_NAME_HINTS = ("comfyui", "ollama", "mlx", "llama.cpp", "lm-studio", "lmstudio", "huggingface")

# File extensions that we treat as HF-compatible model weights.
_MODEL_EXTENSIONS = (".safetensors", ".gguf", ".ckpt")


def _human_size(n):
    """Render byte count as a short human-readable string."""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def _friendly(path):
    """Render a Path with $HOME collapsed to '~' for compact display."""
    s = str(path)
    home = str(Path.home())
    if s == home:
        return "~"
    if s.startswith(home + "/"):
        return "~" + s[len(home):]
    return s

def _dir_size(path):
    """Sum file sizes under path, deduplicated by inode.

    HF cache snapshots are symlinks into the same blobs/, so naive
    rglob+stat double-counts every file. Track (st_dev, st_ino) and
    only count each unique inode once.
    """
    total = 0
    seen = set()
    try:
        for p in path.rglob("*"):
            try:
                if not p.is_file():
                    continue
                st = p.stat()
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                total += st.st_size
            except OSError:
                continue
    except Exception:
        pass
    return total

_QUANT_TIER = {
    # Full precision baseline
    "bf16": "fp", "fp16": "fp",
    # 8-bit tier
    "8bit": "8bit", "q8_0": "8bit", "q8p": "8bit",
    "mxfp8": "8bit", "mixed-9bit": "8bit", "9bit": "8bit",
    # 6-bit tier
    "6bit": "6bit", "q6_k": "6bit", "q6p": "6bit",
    # 5-bit tier
    "5bit": "5bit", "q5_k_m": "5bit", "mixed-5bit": "5bit",
    # 4-bit tier
    "4bit": "4bit", "q4_k_m": "4bit", "q4_k_s": "4bit",
    "iq4": "4bit", "nvfp4": "4bit", "mixed-4bit": "4bit",
    # 3-bit tier
    "3bit": "3bit", "iq3_m": "3bit",
}

def _quant_tier(quant):
    """Bucket a fine-grained quantization into a coarse tier for keeper slots."""
    if not quant:
        return None
    return _QUANT_TIER.get(quant, "other")

def _size_numeric(size_str):
    """Approximate parameter count in billions, or None if unparseable.

    Handles the standard '<n>B' form ('7B', '12B', '122B') and Gemma's edge
    variants ('E4B' → 4). Used both for sort ordering and bucket assignment.
    """
    if not size_str:
        return None
    s = size_str.upper().rstrip("B")
    if s.startswith("E"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return None

# Size buckets aligned with how people actually talk about model scale.
# Boundaries chosen at the gaps between natural clusters in the LLM
# ecosystem rather than at round-number bins, so that "the 4B class" is
# distinct from "the 8B class" and "the 30B class" doesn't share a bucket
# with "the 35-40B class" (different deployment niches, different inference
# cost). Used for slot-grouping (keeper determination), feature-coverage
# supersession, and in-family sort within the curation grid.
_SIZE_CLASS_BUCKETS = [
    ("tiny",   0,    3),     # 0-3B   (0.5B, 1B, 2B — TinyLlama, Phi-3.5 mini, Gemma 2B)
    ("xs",     3,    7),     # 3-7B   (3B, 4B class — Gemma 4B, Phi-4, Llama 3.2 3B)
    ("small",  7,    12),    # 7-12B  (8B class — Llama 8B, Gemma 9B, Granite 8B)
    ("medium", 12,   16),    # 12-16B (12-14B class — Gemma 12B, Mistral Nemo 12B)
    ("large",  16,   32),    # 16-32B (22-30B class — Gemma 27B, Qwen 27B, Granite 30B)
    ("xl",     32,   52),    # 32-52B (32-40B class — Qwen 32B/35B, Yi 34B)
    ("xxl",    52,   92),    # 52-92B (70B class — Llama 70B, Qwen 72B)
    ("huge",   92,   150),   # 92-150B (100-130B class — Qwen 122B, MiniMax-M2.7)
    ("mega",   150,  9999),  # 150B+  (200B+, 405B, 671B — DeepSeek-V3, Llama 405B)
]

def _size_class_label(size_str):
    n = _size_numeric(size_str)
    if n is None:
        return "?"
    for label, lo, hi in _SIZE_CLASS_BUCKETS:
        if lo <= n < hi:
            return label
    return "?"

_SIZE_CLASS_NUMERIC = {label: i for i, (label, _, _) in enumerate(_SIZE_CLASS_BUCKETS)}
_SIZE_CLASS_NUMERIC["?"] = 99

def _size_class_numeric(size_str):
    return _SIZE_CLASS_NUMERIC.get(_size_class_label(size_str), 99)

def _arch_label(enr):
    """'dense' or 'MoE A4B' (with active-params suffix when present)."""
    if enr.get("moe_active"):
        return f"MoE {enr['moe_active']}"
    return "dense"

def _modality_class(pipeline_tag):
    """Coarsen a pipeline_tag into one of {text, image, video, audio}.

    Classifies by *output* modality, not input. HF tags 'image-text-to-text'
    is a vision-LLM (image input, text output) — it belongs with text-LLMs
    for curation, not with image-generation models like FLUX. The split
    point is the last '-to-' in the tag:

      text-generation          → text
      image-text-to-text       → text  (multimodal input, text out)
      text-to-image            → image
      image-to-image           → image
      text-to-video            → video
      automatic-speech-recognition → audio (domain-based fallback)
    """
    if not pipeline_tag:
        return "text"
    pt = pipeline_tag.lower()
    output = pt.rsplit("-to-", 1)[1] if "-to-" in pt else pt
    if output.startswith("image"):
        return "image"
    if output.startswith("video"):
        return "video"
    if "speech" in pt or "audio" in pt:
        return "audio"
    return "text"

def _family_group_key(entry):
    """Hashable group key for curation: (family, modality).

    Broad: every variant of the same family + modality lands in one group
    so cross-version comparisons within an architectural niche can fire.
    The fine-grained niche (size class, dense vs MoE, alignment, quant tier)
    is encoded in the slot key — supersession only triggers when both group
    AND slot match.
    """
    enr = (entry.get("enrichment") or {})
    family = enr.get("family")
    if not family:
        return None
    modality = _modality_class(enr.get("pipeline_tag"))
    return (family, modality)

def _slot_key(entry):
    """Slot identity within a family group. Two entries with the same slot
    key compete for one keeper; different slots are independent niches.

    Components:
      purpose      instruct / coder / reasoning / embedding / asr / ...
      size_class   tiny / xs / small / medium / large / xl / xxl / huge / mega
      arch         dense / moe
      alignment    vanilla / aligned (binary; specific tags don't subdivide)
      quant_tier   fp / 8bit / 6bit / 5bit / 4bit / 3bit / other
      caps         additive capabilities (vision-input / tool-calling / ...)

    Capabilities matter because a vision-LLM and a text-only of the same
    family/version/size/arch fill different niches — both should be
    keepers. Without caps in the slot, e.g. Qwen 3.5 REAP-97B (no vision)
    would falsely collide with Qwen 3.5 122B-A10B nvfp4 (vision-input)
    even though they're different models.
    """
    enr = entry.get("enrichment") or {}
    purpose = enr.get("purpose") or "instruct"
    size_class = _size_class_label(enr.get("size") or "")
    arch = "moe" if enr.get("moe_active") else "dense"
    alignment = enr.get("alignment") or "vanilla"
    tier = _quant_tier(enr.get("quantization")) or "other"
    caps = "+".join(sorted(enr.get("capabilities") or [])) or "none"
    return f"{purpose}/{size_class}/{arch}/{alignment}/{tier}/{caps}"

def _version_float(enr):
    if not enr:
        return None
    v = enr.get("version")
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None

def _last_modified_days(enr):
    """Days since 2000-01-01, or 0 if last_modified is missing."""
    if not enr or not enr.get("last_modified"):
        return 0
    try:
        from datetime import date
        return (date.fromisoformat(enr["last_modified"]) - date(2000, 1, 1)).days
    except Exception:
        return 0

def _runner_rank(runner):
    """Lower is more preferred on Apple Silicon. mlx / mflux are native;
    diffusers and transformers run but typically slower; gguf via llama.cpp
    is fast for text. Unknown runners last."""
    return {
        "mlx": 0, "mflux": 0,
        "gguf": 1,
        "diffusers": 2, "transformers": 2,
    }.get(runner, 3)

def _quant_tier_rank(tier):
    return {"fp": 0, "8bit": 1, "6bit": 2, "5bit": 3, "4bit": 4, "3bit": 5, "other": 6, None: 7}[tier]

def _curation_sort_key(entry):
    """Return a tuple that sorts entries most-recommended-first.

    Order:
      1. Newer version first.
      2. Vanilla (no alignment) before alignment-tagged (so the unmodified
         model leads the list and the alignment variant follows in its slot).
      3. Higher precision first (bf16 → 8bit → 6bit → 5bit → 4bit → 3bit).
      4. MLX > GGUF > safetensors.
      5. More recently updated first.
      6. Alphabetical by id as a stable tiebreaker.
    """
    enr = entry.get("enrichment") or {}
    version = _version_float(enr) or 0.0
    has_alignment = 1 if enr.get("alignment") else 0
    quant_tier = _quant_tier(enr.get("quantization"))
    return (
        -version,
        has_alignment,
        _quant_tier_rank(quant_tier),
        _runner_rank(enr.get("runner")),
        -_last_modified_days(enr),
        entry.get("id") or "",
    )

def curate_entries(entries):
    """Group entries by family and apply keep / supersede / redundant tags.

    Mutates each entry in place by adding a `curation` field:
        curation = {
            "group": "<family> <size> <moe>",   # human label
            "decision": "keep" | "supersede" | "redundant",
            "slot": "<alignment>/<quant_tier>", # e.g. "vanilla/fp", "heretic/4bit"
            "reason": str,
        }

    Decision rules:
      - Group entries by (family, size, moe_active). Entries without a
        parsed family are skipped (no curation tag added).
      - Within a group, identify the highest version present. Anything
        with a lower version is `supersede`.
      - Within the top version, partition by (alignment ?: 'vanilla',
        quant_tier). For each (alignment, tier) slot, the highest-ranked
        entry (per _curation_sort_key) is `keep`; the rest in the same
        slot are `redundant`.
    """
    by_group = {}
    for e in entries:
        key = _family_group_key(e)
        if key is None:
            continue
        by_group.setdefault(key, []).append(e)

    for key, members in by_group.items():
        family, modality = key
        modality_suffix = "" if modality == "text" else f" [{modality}]"
        label = f"{family}{modality_suffix}"

        # Initialize curation tags up front.
        for m in members:
            m["curation"] = {
                "group": label,
                "slot": _slot_key(m),
                "decision": None,
                "reason": None,
            }

        # Identify the highest version present in this family group.
        versions = [
            v for v in (_version_float(m.get("enrichment") or {}) for m in members)
            if v is not None
        ]
        v_max = max(versions) if versions else None

        # Build the set of "feature signatures" present at V_max — what
        # purposes, alignment statuses, capabilities, AND maximum size class
        # the latest family version actually covers. Each signature is
        # (purpose, aligned_bool, frozenset(capabilities), size_class_num).
        # Size class matters: a newer 35B can't substitute for an older 122B
        # (knowledge capacity, instruction-following at hard tasks, recall on
        # long-tail facts all degrade as parameters drop). Size class numeric:
        # tiny=0, small=1, medium=2, large=3, xl=4, xxl=5; larger = bigger.
        vmax_feats = []
        if v_max is not None:
            for m in members:
                v = _version_float(m.get("enrichment") or {})
                if v == v_max:
                    enr = m.get("enrichment") or {}
                    vmax_feats.append((
                        enr.get("purpose") or "instruct",
                        bool(enr.get("alignment")),
                        frozenset(enr.get("capabilities") or []),
                        _size_class_numeric(enr.get("size") or ""),
                    ))

        # Pass 1 — feature-coverage supersession.
        # An older-version entry is superseded by V_max iff some V_max entry
        # has the same purpose, the same alignment status (vanilla vs aligned),
        # a capability set ⊇ the older's, AND a size class >= the older's.
        # If V_max is missing any of those — different purpose, switching
        # alignment, losing a capability, OR not having a model at the older's
        # size class or larger — the older entry stays a keeper because the
        # newer family lacks something key (capability or scale).
        for m in members:
            enr = m.get("enrichment") or {}
            v = _version_float(enr)
            if v_max is None or v is None or v >= v_max:
                continue
            older_purpose = enr.get("purpose") or "instruct"
            older_aligned = bool(enr.get("alignment"))
            older_caps = frozenset(enr.get("capabilities") or [])
            older_sc = _size_class_numeric(enr.get("size") or "")
            covered = False
            for vm_p, vm_a, vm_c, vm_sc in vmax_feats:
                # Unknown sizes (sc==99) don't constrain coverage in either
                # direction — we just don't have the signal to compare.
                size_ok = (older_sc == 99) or (vm_sc == 99) or (vm_sc >= older_sc)
                if (vm_p == older_purpose and vm_a == older_aligned
                        and older_caps.issubset(vm_c) and size_ok):
                    covered = True
                    break
            if covered:
                m["curation"]["decision"] = "supersede"
                aligned_label = "aligned" if older_aligned else "vanilla"
                caps_label = "+".join(sorted(older_caps)) if older_caps else "no extra caps"
                m["curation"]["reason"] = (
                    f"v{v:g} role ({older_purpose}, {aligned_label}, {caps_label}, "
                    f"size_class<={_size_class_label(enr.get('size') or '')}) covered by v{v_max:g}"
                )

        # Pass 2 — slot-based keep/redundant among the un-superseded.
        # Includes V_max members AND older entries whose features V_max
        # didn't cover (those are keeper-eligible because they offer something
        # the newer family lacks). Within a slot, newest-version-and-best
        # wins; same-slot ties become redundant.
        by_slot = {}
        for m in members:
            if m["curation"]["decision"] is not None:
                continue
            by_slot.setdefault(m["curation"]["slot"], []).append(m)

        for slot, slot_members in by_slot.items():
            sorted_members = sorted(slot_members, key=_curation_sort_key)
            kept = False
            for sm in sorted_members:
                cur = sm["curation"]
                if not kept:
                    cur["decision"] = "keep"
                    cur["reason"] = "best in slot"
                    kept = True
                else:
                    cur["decision"] = "redundant"
                    cur["reason"] = f"same slot ({slot}) already filled by a higher-ranked keeper"

_HEX_SHA = re.compile(r"^[0-9a-f]{6,64}$")

def enrich_entry(entry, snapshot_path=None):
    """Augment a discovered-item record with parsed metadata.

    Mutates the entry in place by adding an `enrichment` field containing:
      - all fields from parse_model_id(entry['id'])
      - chat_template fingerprint (when snapshot_path is supplied)
      - a quantization fallback derived from config.json or GGUF filename
        when the name didn't carry a recognizable quant token
      - local_sha (the snapshot dir name, when it looks like a git sha)
        so curation can later compare it against the Hub's current sha
    """
    parsed = parse_model_id(entry.get("id") or "")
    if snapshot_path is not None:
        parsed["chat_template"] = chat_template_fingerprint(snapshot_path)
        if not parsed.get("quantization"):
            parsed["quantization"] = detect_quantization_from_files(snapshot_path)
        # HF cache snapshots are named after the commit sha. For dirs that
        # don't follow that convention (e.g. an LM Studio non-symlinked
        # model dir), skip — local_sha stays None.
        if snapshot_path.name and _HEX_SHA.match(snapshot_path.name):
            parsed["local_sha"] = snapshot_path.name
        else:
            parsed["local_sha"] = None
    else:
        parsed["chat_template"] = None
        parsed["local_sha"] = None
    entry["enrichment"] = parsed
    return entry

def scan_hf_cache(hub_dir):
    """Yield records for each repo under a Hugging Face cache hub/ directory."""
    out = []
    if not hub_dir.exists():
        return out
    for entry in sorted(hub_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        parts = entry.name.split("--", 2)
        if len(parts) != 3:
            continue
        publisher, repo = parts[1], parts[2]
        snapshot = resolve_hf_snapshot(hub_dir, publisher, repo)
        if snapshot is None:
            continue
        formats = set()
        for f in snapshot.rglob("*"):
            if f.name.startswith(".") or not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in _MODEL_EXTENSIONS:
                formats.add(ext.lstrip("."))
        fmt = "+".join(sorted(formats)) if formats else "other"
        rec = {
            "app": "Hugging Face cache",
            "id": f"{publisher}/{repo}",
            "path": snapshot,
            "format": fmt,
            "size": _dir_size(entry),
            "status": "canonical",
        }
        enrich_entry(rec, snapshot_path=snapshot)
        out.append(rec)
    return out

def _collect_hf_blob_hashes(hub_dir):
    """Return the set of sha256 hex strings present as blobs in the HF cache."""
    hashes = set()
    if not hub_dir.exists():
        return hashes
    for entry in hub_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        blobs = entry / "blobs"
        if not blobs.is_dir():
            continue
        for blob in blobs.iterdir():
            if blob.is_file():
                hashes.add(blob.name)
    return hashes

def _local_lfs_shas(hub_dir, publisher, repo):
    """Return the set of blob filenames (== LFS sha256 hex) present locally
    for one HF cache repo. Used to detect "Hub has files we don't have
    locally" — i.e. someone re-uploaded a model file with different bytes
    since the last time we synced."""
    blobs_dir = hub_dir / f"models--{publisher}--{repo}" / "blobs"
    if not blobs_dir.exists():
        return frozenset()
    return frozenset(b.name for b in blobs_dir.iterdir() if b.is_file())

def scan_ollama_models(ollama_dir, app_name="Ollama", hf_blob_hashes=None):
    """Walk Ollama-format manifests and yield records per tag.

    Resolves each tag to its model blob by reading the manifest and looking up
    the application/vnd.ollama.image.model layer. Captures the source registry
    (registry.ollama.ai vs hf.co / huggingface.co) from the manifest path,
    which determines whether the blob's bytes are eligible for HF-cache
    deduplication:

      registry.ollama.ai/library/...   Ollama-curated package, bytes unique
                                       to Ollama's repacking; not byte-equal
                                       to anything in HF cache. Tagged
                                       `ollama_blob`.

      hf.co/<user>/<repo> or
      huggingface.co/<user>/<repo>     Direct HF GGUF download; blob is the
                                       exact bytes of the upstream file.
                                       Tagged `mirror_candidate` and, if
                                       hf_blob_hashes is provided and the
                                       blob's sha256 is already in HF cache,
                                       tagged `dedup_confirmed`.

    `hf_blob_hashes`, if supplied, is a set of sha256 hex strings already
    present in the HF cache; any Ollama blob whose hash is in the set is a
    confirmed duplicate regardless of which registry it came from.
    """
    out = []
    manifests_root = ollama_dir / "manifests"
    if not manifests_root.exists():
        return out
    for tag_path in manifests_root.rglob("*"):
        if not tag_path.is_file():
            continue
        try:
            manifest = json.loads(tag_path.read_text())
        except Exception:
            continue
        layer = next(
            (l for l in manifest.get("layers", [])
             if l.get("mediaType") == "application/vnd.ollama.image.model"),
            None,
        )
        if not layer:
            continue
        digest = layer.get("digest", "")
        if not digest.startswith("sha256:"):
            continue
        blob_hex = digest[len("sha256:"):]
        blob = ollama_dir / "blobs" / f"sha256-{blob_hex}"
        try:
            size = blob.stat().st_size
        except OSError:
            continue

        rel_parts = tag_path.relative_to(manifests_root).parts
        registry = rel_parts[0] if rel_parts else "unknown"
        tag = rel_parts[-1]
        ns_parts = rel_parts[1:-1]
        # Pretty model id, registry-aware:
        #   registry.ollama.ai/library/gemma4/latest -> "gemma4:latest"
        #   hf.co/bartowski/Llama-3.2/Q4_K_M         -> "bartowski/Llama-3.2:Q4_K_M"
        if registry == "registry.ollama.ai" and ns_parts and ns_parts[0] == "library":
            namespace = "/".join(ns_parts[1:])
        else:
            namespace = "/".join(ns_parts)
        model_id = f"{namespace}:{tag}" if namespace else tag

        if hf_blob_hashes is not None and blob_hex in hf_blob_hashes:
            status = "dedup_confirmed"
        elif registry in ("hf.co", "huggingface.co"):
            status = "mirror_candidate"
        else:
            status = "ollama_blob"

        rec = {
            "app": app_name,
            "id": model_id,
            "path": blob,
            "format": "gguf",
            "size": size,
            "status": status,
            "extra": {"registry": registry, "blob_sha256": blob_hex},
        }
        # Ollama-format blobs are nameless; no co-located tokenizer. Name
        # parsing alone supplies what enrichment can.
        enrich_entry(rec, snapshot_path=None)
        out.append(rec)
    return out

_KNOWN_FAMILIES = (
    "gemma", "qwen", "llama", "mistral", "mixtral", "phi", "granite",
    "deepseek", "yi", "nous", "hermes", "olmo", "tinyllama", "falcon",
    "mpt", "stablelm", "codellama", "starcoder", "wizardlm", "qwq",
    "minimax", "ltx", "flux", "z-image", "sdxl", "glm", "parakeet",
    "moondream", "siglip", "clip", "t5",
)
_ALIGNMENT_TOKENS = (
    "heretic", "abliterated", "uncensored", "decensored", "disinhibited",
    "unrestricted", "neutered",
)
# Primary purpose of a model — what it's for. Single value per entry.
# Vision-INPUT capability is separate (in `capabilities`) since most modern
# instructs are vision-capable but you wouldn't keep two of them.
_PURPOSE_TOKENS = (
    ("reasoning", r"\b(reasoning|thinking|qwq)\b|[-_]r1[-_]"),  # explicit reasoning / R1 family
    ("coder",     r"\b(coder|coding)\b"),
    ("embedding", r"\b(embed|embedding|nomic-embed)\b"),
    ("asr",       r"\b(parakeet|whisper|asr|stt)\b"),
    ("base",      r"\bbase\b"),
    # instruct/chat are the default — handled implicitly when nothing else matches
)
_INSTRUCT_TOKENS = re.compile(r"\b(it|instruct|inst|chat)\b")

# Specificity ranks for purpose detection. Used when reconciling HF-tag-derived
# purpose against name-parsed purpose: take whichever is more specific. HF
# inconsistently labels e.g. arthurcollet/Qwen3-Coder-Next as 'instruct' while
# its name clearly says 'Coder' — defer to the more specific signal so the
# model lands in the right purpose group for curation.
_PURPOSE_SPECIFICITY = {
    None:            0,
    "instruct":      1,
    "base":          1,
    "coder":         2,
    "reasoning":     2,
    "embedding":     2,
    "asr":           2,
    "vision-output": 2,
    "video-output":  2,
}

# Capabilities — additive. A model can have multiple. Inferred from HF tags
# primarily; from name as fallback.
_CAPABILITY_NAME_TOKENS = (
    ("vision-input", re.compile(r"\b(vision|vl|visual|multimodal)\b")),
    ("tool-calling", re.compile(r"\b(tool[-_]?(use|calling)|function[-_]?calling)\b")),
    ("long-context", re.compile(r"\b(long[-_]?context|128k|256k|1m[-_]?context)\b")),
)
_CAPABILITY_TAG_KEYS = {
    "vision-input": {"vision", "multimodal"},
    "tool-calling": {"tool-use", "tool-calling", "function-calling"},
    "long-context": {"long-context"},
}
_QUANT_PATTERNS = [
    ("q4_k_m",     r"q4[_-]?k[_-]?m"),
    ("q4_k_s",     r"q4[_-]?k[_-]?s"),
    ("q5_k_m",     r"q5[_-]?k[_-]?m"),
    ("q6_k",       r"q6[_-]?k"),
    ("q8_0",       r"q8[_-]?0"),
    ("iq3_m",      r"iq3[_-]?m"),
    ("iq4",        r"iq4"),
    ("mxfp8",      r"mxfp8"),
    ("nvfp4",      r"nvfp4"),
    ("bf16",       r"bf16"),
    ("fp16",       r"fp16"),
    ("fp8",        r"fp8"),
    ("mixed-9bit", r"mixed[-_]?9[-_]?bit"),
    ("mixed-8bit", r"mixed[-_]?8[-_]?bit"),
    ("mixed-5bit", r"mixed[-_]?5[-_]?bit"),
    ("mixed-4bit", r"mixed[-_]?4[-_]?bit"),
    ("9bit",       r"\b9[-_]?bit\b"),
    ("8bit",       r"\b8[-_]?bit\b"),
    ("6bit",       r"\b6[-_]?bit\b"),
    ("5bit",       r"\b5[-_]?bit\b"),
    ("4bit",       r"\b4[-_]?bit\b"),
    ("3bit",       r"\b3[-_]?bit\b"),
    ("q8p",        r"q8p"),
    ("q6p",        r"q6p"),
]

def parse_model_id(repo_id):
    """Heuristically extract structured metadata from a publisher/name string.

    Returns a dict with keys: publisher, name, family, version, size,
    moe_active, variant, alignment, format, quantization, training. Any
    field whose token isn't recognized in the input is left as None.
    """
    parts = repo_id.split("/")
    publisher = parts[0] if len(parts) >= 2 else None
    name = parts[-1]
    # Search the full publisher/name. Publishers like "mlx-community" or
    # "lmstudio-community" carry format and licensing signal that the model
    # name alone wouldn't reveal.
    s = repo_id.lower()

    out = {
        "publisher":    publisher,
        "name":         name,
        "family":       None,
        "version":      None,
        "size":         None,
        "moe_active":   None,
        "purpose":      None,   # primary purpose: instruct, coder, reasoning, embedding, asr, base, vision-output
        "alignment":    None,
        "format":       None,   # on-disk file format: safetensors, gguf, ckpt, bin
        "runner":       None,   # intended runtime: mlx, mflux, transformers, diffusers, gguf, ollama, drawthings
        "capabilities": [],     # additive: vision-input, tool-calling, long-context, ...
        "quantization": None,
        "training":     None,
    }

    fam_pat = "|".join(re.escape(f) for f in _KNOWN_FAMILIES)
    # Allow ., -, _, or space between family and version (e.g. FLUX.2, Gemma-4,
    # Qwen3, qwen 3.5).
    m = re.search(rf"\b({fam_pat})[ \-_.]?(\d+(?:\.\d+)?)?", s)
    if m:
        out["family"] = m.group(1)
        if m.group(2):
            out["version"] = m.group(2)

    m = re.search(r"(?:^|[\-_ /])(\d+(?:\.\d+)?)\s*[bB](?:[\-_ ./]|$)", repo_id)
    if m:
        out["size"] = f"{m.group(1)}B"
    else:
        # Gemma-style "e4b" / "e2b" sub-billion edge variants — a
        # parameter-class hint where the leading 'e' is part of the token,
        # not a separator. Treat the whole thing as the size label.
        m_e = re.search(r"(?:^|[\-_ /])([eE]\d+[bB])(?:[\-_ ./]|$)", repo_id)
        if m_e:
            out["size"] = m_e.group(1).upper()

    m = re.search(r"[-_][Aa](\d+(?:\.\d+)?)[Bb](?:[\-_/]|$)", repo_id)
    if m:
        out["moe_active"] = f"A{m.group(1)}B"

    # Runner / library — intended runtime. Looked up across the full
    # publisher/name. mflux must be checked before mlx since "mflux" contains
    # "flux" which could match other patterns.
    if "mflux" in s:
        out["runner"] = "mflux"
    elif "mlx" in s:
        out["runner"] = "mlx"
    elif "diffusers" in s:
        out["runner"] = "diffusers"
    elif "gguf" in s:
        out["runner"] = "gguf"

    # File format — what's on disk. Independent of runner. .gguf files are
    # always GGUF; .safetensors and .ckpt are run by various runtimes.
    if "gguf" in s:
        out["format"] = "gguf"
    elif "safetensors" in s:
        out["format"] = "safetensors"
    elif "ckpt" in s:
        out["format"] = "ckpt"
    # If MLX runner detected but no explicit format token, MLX implies safetensors
    if out["runner"] == "mlx" and not out["format"]:
        out["format"] = "safetensors"

    for label, pat in _QUANT_PATTERNS:
        if re.search(pat, s):
            out["quantization"] = label
            break

    # Purpose: prefer the more specific category over generic instruct.
    for label, pat in _PURPOSE_TOKENS:
        if re.search(pat, s):
            out["purpose"] = label
            break
    if not out["purpose"]:
        if _INSTRUCT_TOKENS.search(s):
            out["purpose"] = "instruct"

    # Capabilities — additive
    caps = []
    for label, pat in _CAPABILITY_NAME_TOKENS:
        if pat.search(s):
            caps.append(label)
    out["capabilities"] = caps

    found_alignment = [kw for kw in _ALIGNMENT_TOKENS if kw in s]
    if found_alignment:
        out["alignment"] = "+".join(found_alignment)

    if "qat" in s:
        out["training"] = "qat"
    elif "distilled" in s:
        out["training"] = "distilled"
    elif "dpo" in s:
        out["training"] = "dpo"

    return out

def detect_quantization_from_files(snapshot_path):
    """Inspect files in snapshot_path to determine quantization when name parsing failed.

    Two fallbacks, in order:
      1. config.json — look for an explicit `quantization` / `quantization_config`
         dict (bitsandbytes / GPTQ / MLX / AWQ all set this) or for `torch_dtype`
         indicating fp precision (bfloat16 → bf16, float16 → fp16, etc.).
      2. *.gguf filenames — GGUF files carry the quant in the filename suffix
         (e.g. `gemma-4-26B-A4B-it-Q4_K_M.gguf`); match against _QUANT_PATTERNS.

    Returns the detected quant label or None if no signal found.
    """
    if not snapshot_path or not snapshot_path.exists() or not snapshot_path.is_dir():
        return None
    cfg_path = snapshot_path / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                config = json.load(f)
        except Exception:
            config = None
        if isinstance(config, dict):
            qcfg = config.get("quantization") or config.get("quantization_config")
            if isinstance(qcfg, dict):
                bits = qcfg.get("bits") or qcfg.get("nbits")
                if isinstance(bits, int) and bits > 0:
                    return f"{bits}bit"
                method = qcfg.get("quant_method") or qcfg.get("method")
                if isinstance(method, str) and method:
                    return method.lower()
            dtype = config.get("torch_dtype")
            if isinstance(dtype, str):
                d = dtype.lower()
                if d in ("bfloat16", "bf16"):
                    return "bf16"
                if d in ("float16", "fp16", "half"):
                    return "fp16"
                if d in ("float32", "fp32", "float"):
                    return "fp32"
    try:
        for f in snapshot_path.iterdir():
            if not f.is_file():
                continue
            name = f.name.lower()
            if name.endswith(".gguf"):
                for label, pat in _QUANT_PATTERNS:
                    if re.search(pat, name):
                        return label
    except OSError:
        pass
    return None

def chat_template_fingerprint(snapshot_path):
    """Return a short sha256 prefix of the tokenizer's chat_template, or None.

    Two models with the same chat_template fingerprint share the same prompt
    formatting (subject to whitespace exactness). Used to spot cases where a
    family has multiple checkpoints but only some have the latest template
    update (e.g. Heretic v2).
    """
    if snapshot_path is None or not snapshot_path.exists():
        return None
    candidates = [
        snapshot_path / "tokenizer_config.json",
        snapshot_path / "chat_template.jinja",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        template = None
        if path.suffix == ".json":
            try:
                obj = json.loads(text)
                template = obj.get("chat_template")
            except Exception:
                continue
        else:
            template = text
        if template:
            return hashlib.sha256(template.encode("utf-8")).hexdigest()[:8]
    return None

def find_macos_container_for_bundle(bundle_id):
    """Resolve a Mac App Store sandboxed app's on-disk container path.

    Older Mac App Store apps created their container at
    ~/Library/Containers/<bundle-id>/. Newer ones use a randomly-assigned UUID
    for the container directory and store the bundle-id mapping in the
    container's `.com.apple.containermanagerd.metadata.plist` under
    MCMMetadataIdentifier. Walk both shapes and return whichever matches.
    """
    if sys.platform != "darwin":
        return None
    containers_root = Path.home() / "Library" / "Containers"
    if not containers_root.exists():
        return None
    direct = containers_root / bundle_id
    if direct.exists():
        return direct
    for entry in containers_root.iterdir():
        if not entry.is_dir():
            continue
        meta = entry / ".com.apple.containermanagerd.metadata.plist"
        if not meta.exists():
            continue
        try:
            with open(meta, "rb") as f:
                plist = plistlib.load(f)
        except Exception:
            continue
        if plist.get("MCMMetadataIdentifier") == bundle_id:
            return entry
    return None

def find_comfyui_installs():
    """Locate ComfyUI installs anywhere on disk, regardless of conventional path.

    A ComfyUI install is identified by the characteristic shape: a directory
    that contains both `models/` and `custom_nodes/` subdirectories. This
    catches installs in iCloud Drive, external volumes, project workspaces,
    or any other non-default location. Falls back to an empty list on
    non-macOS or if mdfind isn't available.
    """
    if sys.platform != "darwin":
        return []
    if os.environ.get("LMSTUDIO_HF_NO_MDFIND"):
        return []
    try:
        result = subprocess.run(
            ["mdfind", 'kMDItemFSName == "custom_nodes" && kMDItemContentType == "public.folder"'],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return []
    seen = set()
    installs = []
    for line in result.stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        custom_nodes = Path(s)
        if not custom_nodes.is_dir():
            continue
        parent = custom_nodes.parent
        if parent in seen:
            continue
        if not (parent / "models").is_dir():
            continue
        seen.add(parent)
        installs.append(parent)
    return installs

def scan_drawthings_models(model_dir):
    """Inventory Draw Things models. Each model is a <name>.ckpt, optionally
    paired with a <name>.ckpt-tensordata; size is the sum of both. Skip
    in-flight downloads (.ckpt.partial / .ckpt.partial.map) and metadata json.
    """
    out = []
    if not model_dir.exists() or not model_dir.is_dir():
        return out
    seen = set()
    for path in sorted(model_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix != ".ckpt":
            continue
        if path.name.endswith(".partial") or path.name.endswith(".partial.map"):
            continue
        basename = path.stem
        if basename in seen:
            continue
        seen.add(basename)
        total = 0
        for sibling_name in (f"{basename}.ckpt", f"{basename}.ckpt-tensordata"):
            sibling = model_dir / sibling_name
            try:
                if sibling.exists() and sibling.is_file():
                    total += sibling.stat().st_size
            except OSError:
                continue
        if total < _DISCOVER_MIN_BYTES:
            continue
        rec = {
            "app": "Draw Things",
            "id": basename,
            "path": path,
            "format": "ckpt",
            "size": total,
            "status": "proprietary",
        }
        enrich_entry(rec, snapshot_path=None)
        out.append(rec)
    return out

def scan_flat_dir(root, app_name, extensions=_MODEL_EXTENSIONS):
    """Recursively list HF-compatible model files under root, with size filter."""
    out = []
    if not root.exists() or not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in extensions:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < _DISCOVER_MIN_BYTES:
            continue
        rec = {
            "app": app_name,
            "id": path.name,
            "path": path,
            "format": path.suffix.lower().lstrip("."),
            "size": size,
            "status": "mirror_candidate",
        }
        # ComfyUI / Jan / GPT4All stash files in flat dirs without a
        # co-located tokenizer config; only name parsing applies.
        enrich_entry(rec, snapshot_path=None)
        out.append(rec)
    return out

def scan_mdfind():
    """Run mdfind on macOS for safetensors/gguf paths above the size threshold.

    Returns (paths_with_sizes, note_or_None).
    """
    if sys.platform != "darwin":
        return [], "Tier 2 (mdfind) is macOS-only; skipped."
    if os.environ.get("LMSTUDIO_HF_NO_MDFIND"):
        return [], "Tier 2 (mdfind) skipped (LMSTUDIO_HF_NO_MDFIND set)."
    try:
        result = subprocess.run(
            ["mdfind", 'kMDItemFSName == "*.safetensors" || kMDItemFSName == "*.gguf"'],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        return [], f"Tier 2 (mdfind) failed: {e}"
    out = []
    for line in result.stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        p = Path(s)
        try:
            if not p.is_file() or p.is_symlink():
                continue
            size = p.stat().st_size
        except OSError:
            continue
        if size >= _DISCOVER_MIN_BYTES:
            out.append((p, size))
    return out, None

def _tool_hint_in(s_lower):
    """Return the first tool-name substring matching s_lower, or None."""
    for tool in _TOOL_NAME_HINTS:
        if f"/{tool}/" in s_lower or f"/.{tool}/" in s_lower:
            return tool
    return None

def infer_source_from_path(path):
    """Examine an mdfind hit's path and return a structured origin guess."""
    s = str(path)
    home = str(Path.home())

    # Sandboxed apps — most specific first.
    m = re.match(rf"^{re.escape(home)}/Library/Containers/([^/]+)/", s)
    if m:
        bundle = m.group(1)
        label = _KNOWN_BUNDLE_IDS.get(bundle, bundle)
        return {"kind": "sandbox", "label": label, "evidence": bundle}

    # iCloud Drive — both legacy ("Mobile Documents") and modern ("CloudStorage")
    # mount points. Append a tool-name hint when one appears in the relative path,
    # so "iCloud Drive / .../ComfyUI/..." surfaces as "iCloud Drive / comfyui".
    icloud_legacy = f"{home}/Library/Mobile Documents/com~apple~CloudDocs/"
    if s.startswith(icloud_legacy):
        rest = s[len(icloud_legacy):]
        tool = _tool_hint_in("/" + rest.lower())
        first = rest.split("/", 1)[0] if rest else "(root)"
        label = f"iCloud Drive / {tool}" if tool else f"iCloud Drive / {first}"
        return {"kind": "cloud_sync", "label": label, "evidence": icloud_legacy}

    if s.startswith(f"{home}/Library/CloudStorage/"):
        rest = s[len(f"{home}/Library/CloudStorage/"):]
        provider = rest.split("/", 1)[0] if rest else "(root)"
        tool = _tool_hint_in("/" + rest.lower())
        label = f"{provider} / {tool}" if tool else provider
        return {"kind": "cloud_sync", "label": label,
                "evidence": f"~/Library/CloudStorage/{provider}/"}

    # Other Library-internal locations.
    m = re.match(rf"^{re.escape(home)}/Library/Application Support/([^/]+)/", s)
    if m:
        return {"kind": "app_support", "label": m.group(1),
                "evidence": f"~/Library/Application Support/{m.group(1)}/"}

    m = re.match(rf"^{re.escape(home)}/Library/Caches/([^/]+)/", s)
    if m:
        return {"kind": "app_cache", "label": m.group(1),
                "evidence": f"~/Library/Caches/{m.group(1)}/"}

    # HF blob layout dropped at a non-canonical location.
    m = re.search(r"models--[^/]+--[^/]+/blobs/", s)
    if m:
        idx = s.rfind("models--", 0, m.end())
        end = s.find("/blobs/", idx)
        return {"kind": "hf_format", "label": "HF-cache-format",
                "evidence": s[: end] if end != -1 else s}

    # External volumes.
    if s.startswith("/Volumes/"):
        vol = s[len("/Volumes/"):].split("/", 1)[0]
        tool = _tool_hint_in(s.lower())
        label = f"{vol} / {tool}" if tool else vol
        return {"kind": "external_volume", "label": label,
                "evidence": f"/Volumes/{vol}/"}

    # Tool-named directory anywhere else.
    tool = _tool_hint_in(s.lower())
    if tool:
        return {"kind": "tool_named", "label": tool, "evidence": _friendly(path.parent)}

    # Common user folders.
    for d in ("Downloads", "Desktop"):
        if s.startswith(f"{home}/{d}/"):
            return {"kind": "user_download", "label": d, "evidence": f"~/{d}/"}

    if s.startswith(f"{home}/Documents/"):
        rel = s[len(f"{home}/Documents/"):].split("/", 1)[0]
        return {"kind": "user_curated", "label": f"Documents/{rel}",
                "evidence": f"~/Documents/{rel}/"}

    # Code repos / working dirs that store weights in a conventional subdir.
    # Match the LAST occurrence so deeply nested paths report the closest parent.
    storage_match = None
    for kw in ("weights", "checkpoints", "model_weights", "models"):
        idx = s.rfind(f"/{kw}/")
        if idx == -1:
            continue
        if storage_match is None or idx > storage_match[0]:
            storage_match = (idx, kw)
    if storage_match is not None:
        idx, kw = storage_match
        parent = s[:idx]
        parent_name = parent.split("/")[-1] or parent
        return {"kind": "code_storage", "label": f"{parent_name}/{kw}",
                "evidence": _friendly(Path(f"{parent}/{kw}"))}

    return {"kind": "unclassified", "label": None, "evidence": _friendly(path.parent)}

_STATUS_TAGS = {
    "canonical":         "canonical",
    "mirror_candidate":  "mirror candidate",
    "dedup_confirmed":   "dedup candidate (HF cache hit)",
    "gather_candidate":  "gather candidate",
    "proprietary":       "proprietary",
    "ollama_blob":       "ollama-format",
    "owned_by_app":      "owned by app",
}

def _format_known_section(section):
    """Render one known-cache section."""
    app = section["app"]
    status = section["status"]
    paths = ", ".join(_friendly(p) for p in section["paths"])
    if status == "not present":
        return [f"[{app}]   not present"]
    if status == "installed; no model storage found":
        return [f"[{app}]   installed; no model storage found at {paths}"]
    if status.startswith("scan failed"):
        return [f"[{app}]   {paths}   {status}"]
    items = section["items"]
    total = sum(i["size"] for i in items)
    label = "repo" if app == "Hugging Face cache" else "model"
    plural = "" if len(items) == 1 else "s"
    # Per-section count of items by status, for the section header.
    n_mirror = sum(1 for i in items if i.get("status") == "mirror_candidate")
    n_dedup = sum(1 for i in items if i.get("status") == "dedup_confirmed")
    header = f"[{app}]   {paths}   {len(items)} {label}{plural} · {_human_size(total)}"
    if n_mirror:
        header += f" · {n_mirror} mirror candidate{'' if n_mirror == 1 else 's'}"
    if n_dedup:
        header += f" · {n_dedup} dedup confirmed"
    out = [header]
    if section.get("note"):
        out.append(f"  {section['note']}")
    for i in items:
        meta_parts = [i["format"], _human_size(i["size"])]
        if i.get("extra", {}).get("symlinked"):
            meta_parts.append("symlinked → HF cache")
        line = f"  - {i['id']} ({', '.join(meta_parts)})"
        tag = _STATUS_TAGS.get(i.get("status"))
        if tag and tag != "canonical":  # canonical is implicit; don't clutter
            line += f"   [{tag}]"
        out.append(line)
        enrichment_line = _format_enrichment_line(i.get("enrichment"))
        if enrichment_line:
            out.append(f"      {enrichment_line}")
    return out

def _format_enrichment_line(enrichment):
    """Render the parsed metadata as a compact one-liner, or None if empty."""
    if not enrichment:
        return None
    parts = []
    fam = enrichment.get("family")
    ver = enrichment.get("version")
    if fam:
        parts.append(fam + (f" {ver}" if ver else ""))
    sz = enrichment.get("size")
    moe = enrichment.get("moe_active")
    if sz or moe:
        size_part = sz or ""
        if moe:
            size_part = f"{size_part} {moe}".strip()
        parts.append(size_part)
    flavor_bits = []
    if enrichment.get("purpose"):
        flavor_bits.append(enrichment["purpose"])
    if enrichment.get("alignment"):
        flavor_bits.append(enrichment["alignment"])
    if enrichment.get("training"):
        flavor_bits.append(enrichment["training"])
    if flavor_bits:
        parts.append(" ".join(flavor_bits))
    caps = enrichment.get("capabilities") or []
    if caps:
        parts.append("+".join(caps))
    runner = enrichment.get("runner")
    fmt = enrichment.get("format")
    quant = enrichment.get("quantization")
    runner_fmt = []
    if runner:
        runner_fmt.append(runner)
    if fmt and fmt != runner:
        # show file format alongside runner when they differ (e.g.
        # "mflux · safetensors", "diffusers · safetensors")
        runner_fmt.append(fmt)
    if quant:
        runner_fmt.append(quant)
    if runner_fmt:
        parts.append(" ".join(runner_fmt))
    if enrichment.get("chat_template"):
        parts.append(f"ct:{enrichment['chat_template'][:8]}")
    if enrichment.get("last_modified"):
        parts.append(f"updated:{enrichment['last_modified']}")
    if enrichment.get("license"):
        parts.append(enrichment["license"])
    if not parts:
        return None
    return " · ".join(parts)

def _format_ad_hoc_section(ad_hoc, mdfind_total, already_known, mdfind_note):
    """Render the 'Outside known caches' section."""
    out = ["", "[Outside known caches]   via mdfind, classified by path pattern"]
    if mdfind_note:
        out.append(f"  {mdfind_note}")
    if not ad_hoc:
        out.append("  (none)")
        return out
    by_kind = {}
    for f in ad_hoc:
        by_kind.setdefault(f["inferred"]["kind"], []).append(f)
    # (kind, heading, action) — action is the suggested handling for this bucket.
    kind_order = [
        ("sandbox",         "Sandboxed apps (containers)",                  "owned_by_app"),
        ("app_support",     "App Support directories",                      "owned_by_app"),
        ("app_cache",       "App Caches",                                   "owned_by_app"),
        ("hf_format",       "HF-cache-format at non-canonical locations",   "gather_candidate"),
        ("tool_named",      "Tool-named directories",                       "gather_candidate"),
        ("user_curated",    "User-curated personal storage",                "gather_candidate"),
        ("user_download",   "Manual downloads (unsorted)",                  "gather_candidate"),
        ("cloud_sync",      "Cloud-synced (materialized locally)",          "gather_candidate"),
        ("external_volume", "External volumes",                             "gather_candidate"),
        ("code_storage",    "Code repos / working dirs with model weights", "gather_candidate"),
        ("unclassified",    "Truly unclassified",                           "gather_candidate"),
    ]
    for kind, heading, action in kind_order:
        files = by_kind.get(kind)
        if not files:
            continue
        out.append("")
        action_tag = _STATUS_TAGS.get(action, "")
        out.append(f"  {heading}:" + (f"   [{action_tag}]" if action_tag else ""))
        by_label = {}
        for f in files:
            label = f["inferred"]["label"] or "(unknown)"
            by_label.setdefault(label, []).append(f)
        for label in sorted(by_label):
            lfiles = by_label[label]
            total = sum(f["size"] for f in lfiles)
            n = len(lfiles)
            out.append(f"    {label}   ({n} file{'s' if n != 1 else ''}, {_human_size(total)})")
            for f in lfiles[:5]:
                out.append(f"      {_friendly(f['path'])}")
            if n > 5:
                out.append(f"      ... and {n - 5} more")
    return out

def format_discovery_report(sections, ad_hoc, mdfind_total, already_known, mdfind_note):
    """Top-level report formatter."""
    lines = []
    for s in sections:
        lines.extend(_format_known_section(s))
        lines.append("")
    lines.extend(_format_ad_hoc_section(ad_hoc, mdfind_total, already_known, mdfind_note))
    lines.append("")
    scanned = sum(1 for s in sections if s["status"] == "scanned")
    total_known_items = sum(len(s.get("items", [])) for s in sections if s["status"] == "scanned")
    lines.append("Summary:")
    lines.append(f"  {total_known_items} model entr{'y' if total_known_items == 1 else 'ies'} across {scanned} active known cache{'s' if scanned != 1 else ''}")
    lines.append(f"  Tier 2 mdfind: {mdfind_total} candidate{'s' if mdfind_total != 1 else ''} scanned, {already_known} already in known caches")
    lines.append(f"  Tier 2 unique ad-hoc finds: {len(ad_hoc)}")

    # Candidates rollup — grouped by action so the user sees what's
    # actionable today and what could be next.
    mirror_items = []  # (app, id, size)
    dedup_items = []   # (app, id, size)
    for s in sections:
        if s["status"] != "scanned":
            continue
        for i in s["items"]:
            if i.get("status") == "mirror_candidate":
                mirror_items.append((s["app"], i["id"], i["size"]))
            elif i.get("status") == "dedup_confirmed":
                dedup_items.append((s["app"], i["id"], i["size"]))

    gather_items = [a for a in ad_hoc if a.get("status") == "gather_candidate"]

    if mirror_items or dedup_items or gather_items:
        lines.append("")
        lines.append("Candidates:")
    if mirror_items:
        total = sum(sz for _, _, sz in mirror_items)
        by_app = {}
        for app, _, sz in mirror_items:
            by_app.setdefault(app, [0, 0])
            by_app[app][0] += 1
            by_app[app][1] += sz
        lines.append(f"  Mirror candidates  ({len(mirror_items)} model{'' if len(mirror_items) == 1 else 's'} · {_human_size(total)}):")
        for app in sorted(by_app):
            n, sz = by_app[app]
            lines.append(f"    {app}: {n} model{'' if n == 1 else 's'} · {_human_size(sz)}")
    if dedup_items:
        total = sum(sz for _, _, sz in dedup_items)
        by_app = {}
        for app, _, sz in dedup_items:
            by_app.setdefault(app, [0, 0])
            by_app[app][0] += 1
            by_app[app][1] += sz
        lines.append(f"  Dedup confirmed  ({len(dedup_items)} blob{'' if len(dedup_items) == 1 else 's'} · {_human_size(total)} byte-identical to HF cache):")
        for app in sorted(by_app):
            n, sz = by_app[app]
            lines.append(f"    {app}: {n} blob{'' if n == 1 else 's'} · {_human_size(sz)}")
    if gather_items:
        total = sum(a["size"] for a in gather_items)
        by_kind = {}
        for a in gather_items:
            kind = a["inferred"]["kind"]
            label = a["inferred"]["label"] or "(unknown)"
            key = (kind, label)
            by_kind.setdefault(key, [0, 0])
            by_kind[key][0] += 1
            by_kind[key][1] += a["size"]
        lines.append(f"  Gather candidates  ({len(gather_items)} file{'' if len(gather_items) == 1 else 's'} · {_human_size(total)}):")
        for (kind, label), (n, sz) in sorted(by_kind.items()):
            lines.append(f"    {kind} / {label}: {n} file{'' if n == 1 else 's'} · {_human_size(sz)}")
    return "\n".join(lines)

def _enrich_with_hub_metadata(sections, hub_dir=None):
    """Augment HF-derived items with authoritative metadata from the Hub.

    Extracts and applies, for each HF-resolved repo:
      library_name      Authoritative format (mlx, transformers, gguf).
                        Overrides the heuristic name-parsed format.
      pipeline_tag      e.g. "text-generation", "image-text-to-text".
                        Used to split text-LLM groups from image-gen
                        groups in curation when name-parsing has no
                        size signal to differentiate them.
      tags              Full tag list. Searched for alignment tokens
                        (heretic / abliterated / uncensored / decensored
                        / disinhibited / unrestricted / neutered) and
                        for an architecture identifier (qwen3_5_moe,
                        gemma3, granite, etc.).
      base_model        Upstream lineage from card_data.base_model.
      license           From card_data.license.
      last_modified     ISO date string of latest commit.

    All HF-sourced fields take precedence over the heuristic name-parsed
    fields when set. Cached per repo_id so repeated lookups of the same
    repo (e.g. HF cache + symlinked LM Studio) cost one API call.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except Exception as e:
        print(f"  (--hub-meta requires huggingface_hub: {e})", file=sys.stderr)
        return
    api = HfApi()
    cache = {}

    def _detect_alignment_from_tags(tags):
        hits = sorted(set(t for t in tags if t in _ALIGNMENT_TOKENS))
        return "+".join(hits) if hits else None

    def _detect_purpose_from_tags(tags, pipeline_tag):
        """Map HF tags + pipeline_tag onto our purpose taxonomy."""
        tag_set = {t.lower() for t in tags if isinstance(t, str)}
        if any(t in tag_set for t in ("reasoning", "thinking", "chain-of-thought")):
            return "reasoning"
        if any(t in tag_set for t in ("code", "coder", "code-generation", "code-completion")):
            return "coder"
        if pipeline_tag in ("feature-extraction", "sentence-similarity", "sentence-embeddings"):
            return "embedding"
        if pipeline_tag in ("automatic-speech-recognition", "audio-to-audio"):
            return "asr"
        if pipeline_tag in ("text-to-image", "image-to-image"):
            return "vision-output"
        if pipeline_tag in ("text-to-video", "image-to-video"):
            return "video-output"
        # Default for conversational text models: instruct.
        if "conversational" in tag_set or pipeline_tag in ("text-generation", "image-text-to-text"):
            return "instruct"
        return None

    def _detect_capabilities_from_tags(tags, pipeline_tag):
        tag_set = {t.lower() for t in tags if isinstance(t, str)}
        caps = []
        for cap, keys in _CAPABILITY_TAG_KEYS.items():
            if tag_set & keys:
                caps.append(cap)
        # Vision-input is implicit when pipeline_tag accepts image input but
        # outputs text (image-text-to-text).
        if pipeline_tag == "image-text-to-text" and "vision-input" not in caps:
            caps.append("vision-input")
        return caps

    def _detect_architecture_from_tags(tags):
        # HF tags often include the model_type (e.g. "qwen3_5_moe", "gemma3",
        # "granite"). Pick the first tag that looks like an architecture
        # identifier — alphabetic + optional digits, possibly with underscore
        # separators, no spaces or punctuation.
        for t in tags:
            if not isinstance(t, str):
                continue
            # Skip namespaced tags ("base_model:...", "license:...", "region:us")
            if ":" in t:
                continue
            if re.fullmatch(r"[a-z][a-z0-9_]*\d[a-z0-9_]*", t) or re.fullmatch(r"[a-z]+\d+[a-z_]*", t):
                # avoid plain quant tags like "6-bit" (has hyphen) or "8B"
                if t.endswith("bit") or re.fullmatch(r"\d+b", t):
                    continue
                return t
        return None

    def lookup(repo_id):
        if repo_id in cache:
            return cache[repo_id]
        result = {
            "last_modified": None, "license": None,
            "library_name": None, "pipeline_tag": None,
            "tags": [], "alignment": None,
            "base_model": None, "model_type": None,
            "hub_sha": None, "hub_lfs_shas": frozenset(),
        }
        try:
            # files_metadata=True so we can compare the Hub's LFS sha256s
            # to the local blob set and detect "model file replaced
            # entirely on the Hub" — distinct from a generic commit bump.
            info = api.model_info(repo_id, files_metadata=True)
        except (RepositoryNotFoundError, GatedRepoError):
            cache[repo_id] = result
            return result
        except Exception:
            cache[repo_id] = result
            return result
        if getattr(info, "last_modified", None):
            result["last_modified"] = info.last_modified.date().isoformat()
        result["hub_sha"] = getattr(info, "sha", None)
        result["library_name"] = getattr(info, "library_name", None)
        result["pipeline_tag"] = getattr(info, "pipeline_tag", None)
        tags = list(getattr(info, "tags", []) or [])
        result["tags"] = tags
        result["alignment"] = _detect_alignment_from_tags(tags)
        result["model_type"] = _detect_architecture_from_tags(tags)
        result["purpose"] = _detect_purpose_from_tags(tags, result["pipeline_tag"])
        result["capabilities"] = _detect_capabilities_from_tags(tags, result["pipeline_tag"])
        # Collect every LFS sha256 the current head expects. For a non-LFS
        # text file (config.json, etc.) lfs is None — those don't contribute.
        lfs_shas = set()
        for sib in (getattr(info, "siblings", None) or []):
            sha = _lfs_sha256(sib)
            if sha:
                lfs_shas.add(sha)
        result["hub_lfs_shas"] = frozenset(lfs_shas)
        cd = getattr(info, "card_data", None)
        if cd is not None:
            cd_dict = cd.to_dict() if hasattr(cd, "to_dict") else dict(cd)
            result["license"] = cd_dict.get("license")
            bm = cd_dict.get("base_model")
            if isinstance(bm, list) and bm:
                result["base_model"] = bm[0]
            elif isinstance(bm, str):
                result["base_model"] = bm
        cache[repo_id] = result
        return result

    targets = []  # (item, repo_id) pairs
    for s in sections:
        if s["status"] != "scanned":
            continue
        for item in s["items"]:
            repo_id = None
            if s["app"] in ("Hugging Face cache", "LM Studio"):
                # id is already publisher/name
                repo_id = item.get("id")
            else:
                # Ollama family: only the hf.co-pulled entries have HF metadata.
                # The id for those is "<user>/<repo>:<tag>"; strip the tag.
                extra = item.get("extra", {})
                if extra.get("registry") in ("hf.co", "huggingface.co"):
                    raw = item.get("id") or ""
                    repo_id = raw.split(":", 1)[0] if ":" in raw else raw
            if repo_id and "/" in repo_id:
                targets.append((item, repo_id))

    if not targets:
        return

    # Per-run cache for the in-memory chat-template fetches. Avoids hitting
    # the same repo twice when several entries point at the same model.
    ct_cache = {}

    print(f"  (fetching Hub metadata for {len(targets)} repo{'' if len(targets) == 1 else 's'}...)", file=sys.stderr, flush=True)
    for item, repo_id in targets:
        meta = lookup(repo_id)
        enr = item.setdefault("enrichment", {})
        # Always set even when None; downstream uses these as authoritative.
        enr["last_modified"] = meta["last_modified"]
        enr["license"] = meta["license"]
        enr["pipeline_tag"] = meta["pipeline_tag"]
        enr["base_model"] = meta["base_model"]
        enr["model_type"] = meta["model_type"]
        enr["hub_sha"] = meta["hub_sha"]
        enr["hub_lfs_shas"] = meta["hub_lfs_shas"]
        # Authoritative overrides — only when HF actually returned a value;
        # otherwise leave the heuristic-parsed field in place as fallback.
        # library_name is the runner/runtime, NOT a file format. mflux,
        # diffusers, transformers are runtimes that run safetensors files.
        if meta.get("library_name"):
            enr["runner"] = meta["library_name"]
        if meta.get("alignment"):
            enr["alignment"] = meta["alignment"]
        if meta.get("purpose"):
            parsed_purpose = enr.get("purpose")
            hf_purpose = meta["purpose"]
            # Take the more specific of the two. Coder/reasoning/embedding/asr
            # are more specific than HF's generic conversational "instruct"
            # default, so a name-parsed "coder" wins over an HF "instruct".
            parsed_rank = _PURPOSE_SPECIFICITY.get(parsed_purpose, 0)
            hf_rank = _PURPOSE_SPECIFICITY.get(hf_purpose, 0)
            if hf_rank >= parsed_rank:
                enr["purpose"] = hf_purpose
        if meta.get("capabilities"):
            # Merge capabilities from HF on top of name-parsed; dedupe.
            existing = list(enr.get("capabilities") or [])
            for c in meta["capabilities"]:
                if c not in existing:
                    existing.append(c)
            enr["capabilities"] = existing
        # Compute local LFS sha set so the static grid / picker can flag
        # entries where the Hub has bytes we don't. Cheap directory listing;
        # only relevant for HF-cache-format storage.
        if hub_dir is not None and "/" in repo_id:
            pub, name = repo_id.split("/", 1)
            enr["local_lfs_shas"] = _local_lfs_shas(hub_dir, pub, name)

        # If the Hub has a different commit sha than our local snapshot AND
        # the LFS file set is intact, the change is in non-LFS files —
        # README, tokenizer config, generation config, etc. Pull
        # tokenizer_config.json into memory to see if the chat template
        # specifically was the thing that changed (that's the
        # prompt-format-affecting case). Only fetched once per repo per
        # run, only when ↻ would otherwise be the icon — never touches
        # disk.
        local_sha = (item.get("enrichment") or {}).get("local_sha")
        hub_sha = meta.get("hub_sha")
        hub_lfs = meta.get("hub_lfs_shas") or frozenset()
        local_lfs = enr.get("local_lfs_shas") or frozenset()
        if (hub_sha and local_sha and hub_sha != local_sha
                and hub_lfs and hub_lfs.issubset(local_lfs)):
            if repo_id not in ct_cache:
                ct_cache[repo_id] = _fetch_hub_chat_template_fp(repo_id, hub_sha)
            enr["hub_chat_template"] = ct_cache[repo_id]

_DECISION_TAG = {
    "keep":      "keep",
    "supersede": "supersede",
    "redundant": "redundant",
    None:        "indeterminate",
}

# Short codes used as column headers in the curation grid. Order is the
# render order. HF is listed first because it's the canonical store and
# many tools draw from it.
_TOOL_COLUMNS = [
    ("HF",  "Hugging Face cache"),
    ("LM",  "LM Studio"),
    ("OL",  "Ollama"),
    ("MS",  "Msty family"),       # Msty / Msty Claw / MstyStudio share the column
    ("CU",  "ComfyUI"),
    ("DT",  "Draw Things"),
    ("DB",  "Diffusion Bee"),
]

# Which on-disk section's app maps to which column.
_APP_TO_TOOL = {
    "Hugging Face cache":  "HF",
    "LM Studio":           "LM",
    "Ollama":              "OL",
    "Msty":                "MS",
    "Msty Claw":           "MS",
    "MstyStudio":          "MS",
    "ComfyUI":             "CU",
    "Draw Things":         "DT",
    "Diffusion Bee":       "DB",
}

# Which tools can natively use which runners. Used to mark "compatible"
# (✓) cells in addition to "currently stored here" (●) cells.
_RUNNER_COMPAT = {
    # mlx safetensors run by LM Studio's mlx engine and by mlx-lm/mlx_vlm
    # which read directly from HF cache.
    "mlx":          {"HF", "LM"},
    # mflux is FLUX/diffusion on Apple Silicon — typically reads HF cache.
    "mflux":        {"HF"},
    # diffusers read by ComfyUI and many MFLUX-style runners.
    "diffusers":    {"HF", "CU"},
    # transformers Python — runs from HF cache; LM Studio usually wants
    # the MLX-converted version, but many transformers checkpoints work.
    "transformers": {"HF"},
    # gguf is consumed by LM Studio's llama.cpp engine, by Ollama, and by
    # all the Ollama-format apps in the Msty family.
    "gguf":         {"HF", "LM", "OL", "MS"},
    # Drawthings uses its own .ckpt repack format; only Draw Things itself
    # can load these.
    "drawthings":   {"DT"},
}

def format_curation_view(sections, ad_hoc):
    """Render entries grouped by (family, size, moe), sorted recommended-first.

    Includes scanned tool-storage entries (eligible: not proprietary, not
    ollama_blob), plus ad-hoc finds whose name parses to a known family.
    Entries with no family parsed go into an "Ungrouped" tail section.
    """
    entries = []
    for s in sections:
        if s["status"] != "scanned":
            continue
        for e in s["items"]:
            # Skip Draw Things (proprietary) and Ollama-curated (ollama_blob)
            # — neither is actionable through HF cache curation.
            if e.get("status") in ("proprietary", "ollama_blob"):
                continue
            # An LM Studio dir whose contents are all symlinks IS a view of
            # an HF cache snapshot already represented in curation; including
            # it would duplicate every keep/redundant decision. Skip.
            if s["app"] == "LM Studio" and e.get("extra", {}).get("symlinked"):
                continue
            entries.append((s["app"], e))
    for a in ad_hoc:
        # Promote ad-hoc finds with a parsed family into curation; else leave
        # them in the ad-hoc bucket above. They keep their gather_candidate
        # status; curation just adds the family-grouped ordering.
        if (a.get("enrichment") or {}).get("family"):
            entries.append(("(ad-hoc)", a))

    # Run the curation decisions on the entry objects (in-place mutation).
    curate_entries([e for _app, e in entries])

    # Group for display.
    by_group = {}
    ungrouped = []
    for app, e in entries:
        cur = e.get("curation")
        if cur is None:
            ungrouped.append((app, e))
            continue
        by_group.setdefault(cur["group"], []).append((app, e))

    out = ["", "=" * 78, "Curation view (recommended first per family group)", "=" * 78, ""]

    # Sort groups: family asc, size desc-ish (parse the leading number),
    # moe_active asc. Stable enough for readable output.
    def _group_sort_key(label):
        return label
    for group_label in sorted(by_group, key=_group_sort_key):
        members = by_group[group_label]
        # Sort within group by curation order (already deterministic).
        members.sort(key=lambda pair: _curation_sort_key(pair[1]))
        n = len(members)
        n_keep = sum(1 for _, e in members if (e.get("curation") or {}).get("decision") == "keep")
        n_super = sum(1 for _, e in members if (e.get("curation") or {}).get("decision") == "supersede")
        n_redund = sum(1 for _, e in members if (e.get("curation") or {}).get("decision") == "redundant")
        total_size = sum(e["size"] for _, e in members)
        out.append(f"[{group_label}]   {n} entr{'y' if n == 1 else 'ies'} · {_human_size(total_size)} · {n_keep} keep, {n_super} supersede, {n_redund} redundant")
        for app, e in members:
            cur = e.get("curation") or {}
            decision = cur.get("decision")
            decision_tag = _DECISION_TAG.get(decision, "")
            slot = cur.get("slot") or "?"
            id_str = e.get("id") or "(unnamed)"
            size_str = _human_size(e.get("size") or 0)
            enr = e.get("enrichment") or {}
            meta_bits = []
            if enr.get("purpose"): meta_bits.append(enr["purpose"])
            if enr.get("alignment"): meta_bits.append(enr["alignment"])
            if enr.get("capabilities"):
                meta_bits.append("+".join(enr["capabilities"]))
            runner = enr.get("runner")
            fmt = enr.get("format")
            if runner:
                meta_bits.append(runner)
            if fmt and fmt != runner:
                meta_bits.append(fmt)
            if enr.get("quantization"): meta_bits.append(enr["quantization"])
            if enr.get("chat_template"): meta_bits.append(f"ct:{enr['chat_template'][:8]}")
            if enr.get("last_modified"): meta_bits.append(f"upd:{enr['last_modified']}")
            if enr.get("base_model"): meta_bits.append(f"from {enr['base_model']}")
            meta = " · ".join(meta_bits) if meta_bits else ""
            tag = f"[{decision_tag}]"
            out.append(f"  {tag:<14} {app:<14} {id_str}")
            out.append(f"      {size_str}   {slot}   {meta}")
            if cur.get("reason") and decision in ("supersede", "redundant"):
                out.append(f"      → {cur['reason']}")
        out.append("")

    if ungrouped:
        out.append("[Ungrouped — no family parsed]")
        for app, e in ungrouped:
            id_str = e.get("id") or "(unnamed)"
            size_str = _human_size(e.get("size") or 0)
            out.append(f"  {app:<14} {id_str}   {size_str}")
        out.append("")

    # Rollup of prune candidates.
    prune_items = []
    for app, e in entries:
        cur = e.get("curation") or {}
        if cur.get("decision") in ("supersede", "redundant"):
            prune_items.append((app, e, cur))
    if prune_items:
        total = sum(e["size"] for _, e, _ in prune_items)
        out.append(f"Prune candidates: {len(prune_items)} entr{'y' if len(prune_items) == 1 else 'ies'} · {_human_size(total)} reclaimable")
        by_decision = {"supersede": [], "redundant": []}
        for app, e, cur in prune_items:
            by_decision[cur["decision"]].append((app, e))
        for d, lst in by_decision.items():
            if not lst:
                continue
            sub = sum(e["size"] for _, e in lst)
            out.append(f"  {d}: {len(lst)} · {_human_size(sub)}")
    return "\n".join(out)

_PURPOSE_DISPLAY_ORDER = (
    "instruct",
    "coder",
    "reasoning",
    "embedding",
    "asr",
    "vision-output",
    "video-output",
    "base",
    "other",
)

def _identity_string(enr):
    """Compose a compact identity string from enrichment fields.

    Form: "<family> <version> <size> <arch> <alignment-or-vanilla> <caps>".
    Architecture is appended as 'dense' or 'MoE A4B' (carrying the active-
    params suffix when MoE) so the dense vs MoE distinction is visible at
    a glance, immediately after the size class.
    """
    parts = []
    if enr.get("family"):
        parts.append(enr["family"])
    if enr.get("version"):
        parts.append(enr["version"])
    if enr.get("size"):
        parts.append(enr["size"])
    parts.append(_arch_label(enr))
    parts.append(enr.get("alignment") or "vanilla")
    if enr.get("capabilities"):
        parts.append("+".join(enr["capabilities"]))
    return " ".join(parts) if parts else "(unknown)"

# Column widths for the curation grid + interactive picker. Chosen so the
# grid fits in ~170 cols; values that would overflow are explicitly
# truncated at render time so subsequent columns stay aligned.
_GRID_W_ID, _GRID_W_RUN, _GRID_W_QUANT, _GRID_W_SIZE, _GRID_W_UPD, _GRID_W_CT, _GRID_W_HUB, _GRID_W_DEC, _GRID_W_FROM = (
    46, 12, 11, 8, 11, 8, 3, 10, 32
)

def _fetch_hub_chat_template_fp(repo_id, revision):
    """Fetch the Hub's chat_template into memory and return its fingerprint.

    Looks at tokenizer_config.json's `chat_template` field first, falling
    back to a standalone chat_template.jinja file. Returns the same shape
    of fingerprint as chat_template_fingerprint() (sha256 hex prefix).

    The fetch is in-memory only — never touches disk. Uses
    huggingface_hub.hf_hub_url + huggingface_hub.utils.build_hf_headers
    to pick up HF_TOKEN auth for gated/private repos automatically, then
    plain urllib.request to actually pull the bytes. No on-disk cache,
    no .cache/huggingface side effects from this function.

    Returns None on any failure (missing file, 404, parse error, gated
    without auth, network glitch).
    """
    try:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
    except Exception:
        return None
    import urllib.request
    import urllib.error

    candidates = [("tokenizer_config.json", "json"),
                  ("chat_template.jinja", "raw")]
    for filename, kind in candidates:
        try:
            url = hf_hub_url(repo_id, filename, revision=revision or "main")
        except Exception:
            continue
        try:
            headers = build_hf_headers()
        except Exception:
            headers = {}
        try:
            req = urllib.request.Request(url, headers=dict(headers))
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read().decode("utf-8")
        except urllib.error.HTTPError:
            continue
        except Exception:
            continue
        template = None
        if kind == "json":
            try:
                obj = json.loads(data)
                template = obj.get("chat_template")
            except Exception:
                continue
        else:
            template = data
        if template:
            return hashlib.sha256(template.encode("utf-8")).hexdigest()[:8]
    return None

def _hub_status_icon(enr):
    """Return a short icon describing this entry's local-vs-Hub state, or ' '.

    ↥ (upgrade)         Hub has at least one LFS file with a sha256 our
                        local cache doesn't have — the model file itself
                        was re-uploaded. Re-pull would replace bytes.
    ♢ (white diamond)   Commit changed AND the chat template specifically
                        is different from local. Visually evokes <html>-style
                        template brackets; signals a prompt-format update
                        that may need re-evaluation in clients.
    ↻ (refresh)         Commit changed, LFS files intact, chat template
                        unchanged or not checkable — likely a README,
                        generation_config, or other metadata update.
    ' ' (blank)         In sync OR not enough Hub data fetched to compare
                        (no --hub-meta, gated/private without auth, etc.).
    """
    hub_lfs = enr.get("hub_lfs_shas")
    local_lfs = enr.get("local_lfs_shas")
    if hub_lfs and local_lfs is not None and not hub_lfs.issubset(local_lfs):
        return "↥"
    hub_sha = enr.get("hub_sha")
    local_sha = enr.get("local_sha")
    if hub_sha and local_sha and hub_sha != local_sha:
        local_ct = enr.get("chat_template")
        hub_ct = enr.get("hub_chat_template")
        if local_ct and hub_ct and local_ct != hub_ct:
            return "♢"
        return "↻"
    return " "

def _grid_sort_key_for_pair(pair):
    """Sort hierarchy within a purpose group, most-significant first:
       1. family            A → Z
       2. version           newer → older
       3. size class        large → small (unknown last)
       4. architecture      dense → MoE
       5. exact size        large → small
       6. alignment         vanilla → alignment-tagged
       7. last-modified     newer → older
       8. quant tier        high precision → low
       9. runner            mlx/mflux → gguf → transformers/diffusers
      10. id                alphabetical
    """
    _app, e = pair
    enr = e.get("enrichment") or {}
    fam = (enr.get("family") or "~").lower()
    version = _version_float(enr) or 0.0
    sc_n = _size_class_numeric(enr.get("size") or "")
    size_class_rank = 999 if sc_n == 99 else -sc_n
    arch_rank = 1 if enr.get("moe_active") else 0
    size_n = _size_numeric(enr.get("size") or "") or 0.0
    has_alignment = 1 if enr.get("alignment") else 0
    last_mod = _last_modified_days(enr)
    quant_rank = _quant_tier_rank(_quant_tier(enr.get("quantization")))
    runner_rank = _runner_rank(enr.get("runner"))
    eid = e.get("id") or ""
    return (
        fam, -version, size_class_rank, arch_rank, -size_n,
        has_alignment, -last_mod, quant_rank, runner_rank, eid,
    )

def compute_curation_grid_data(sections, ad_hoc):
    """Build a structured row list for the curation grid.

    Returns a flat list of dicts, each either:
      {"kind": "header", "purpose": str, "n": int, "size": int,
       "keepers": int, "prunes": int}
      {"kind": "entry", "app": str, "entry": dict,
       "stored_tools": set[str], "compat_tools": set[str]}

    Both the static text renderer (format_curation_grid) and the
    interactive picker (mark_for_curation) consume this list, so a single
    code path determines which entries appear, in what order, with what
    metadata.
    """
    # id -> set of app names that hold a copy. Used to merge symlinked
    # LM Studio entries into their HF cache twin.
    id_to_apps = {}
    for s in sections:
        if s["status"] != "scanned":
            continue
        for e in s["items"]:
            eid = e.get("id") or ""
            if eid:
                id_to_apps.setdefault(eid, set()).add(s["app"])

    grid_entries = []
    for s in sections:
        if s["status"] != "scanned":
            continue
        for e in s["items"]:
            if s["app"] == "LM Studio" and e.get("extra", {}).get("symlinked"):
                continue
            grid_entries.append((s["app"], e))
    for a in ad_hoc:
        if (a.get("enrichment") or {}).get("family"):
            grid_entries.append(("(ad-hoc)", a))

    curate_entries([e for _app, e in grid_entries])

    by_purpose = {}
    for app, e in grid_entries:
        purpose = (e.get("enrichment") or {}).get("purpose") or "other"
        by_purpose.setdefault(purpose, []).append((app, e))

    purposes = list(_PURPOSE_DISPLAY_ORDER) + sorted(p for p in by_purpose if p not in _PURPOSE_DISPLAY_ORDER)

    rows = []
    for purpose in purposes:
        members = by_purpose.get(purpose) or []
        if not members:
            continue
        members.sort(key=_grid_sort_key_for_pair)
        n = len(members)
        total = sum(e["size"] for _, e in members)
        keepers = sum(1 for _, e in members if (e.get("curation") or {}).get("decision") == "keep")
        prunes = sum(1 for _, e in members
                     if (e.get("curation") or {}).get("decision") in ("supersede", "redundant"))
        rows.append({
            "kind": "header", "purpose": purpose, "n": n, "size": total,
            "keepers": keepers, "prunes": prunes,
        })
        for app, e in members:
            stored_tools = set()
            for app_name in id_to_apps.get(e.get("id") or "", set()):
                tc = _APP_TO_TOOL.get(app_name)
                if tc:
                    stored_tools.add(tc)
            tc_for_app = _APP_TO_TOOL.get(app)
            if tc_for_app:
                stored_tools.add(tc_for_app)
            enr = e.get("enrichment") or {}
            compat_tools = set(_RUNNER_COMPAT.get(enr.get("runner"), set()))
            rows.append({
                "kind": "entry", "app": app, "entry": e,
                "stored_tools": stored_tools, "compat_tools": compat_tools,
            })
    return rows

def _format_grid_row_content(row, tool_codes):
    """Render the per-row content (without leading marker/cursor) shared
    between the static grid and the interactive picker."""
    e = row["entry"]
    enr = e.get("enrichment") or {}
    cur = e.get("curation") or {}
    ident = _identity_string(enr)[:_GRID_W_ID]
    runner = (enr.get("runner") or "")[:_GRID_W_RUN]
    quant = (enr.get("quantization") or "—")[:_GRID_W_QUANT]
    size_h = _human_size(e.get("size") or 0)
    updated = (enr.get("last_modified") or "")[:_GRID_W_UPD]
    ct = enr.get("chat_template")
    ct_short = (ct[:8] if ct else "")
    hub = _hub_status_icon(enr)
    decision = (cur.get("decision") or "—")[:_GRID_W_DEC]
    bm = enr.get("base_model") or ""
    if len(bm) > _GRID_W_FROM:
        bm = bm[:_GRID_W_FROM - 1] + "…"
    cells = []
    for tc in tool_codes:
        if tc in row["stored_tools"]:
            cells.append("●")
        elif tc in row["compat_tools"]:
            cells.append("✓")
        else:
            cells.append(" ")
    tool_marks = " ".join(f"{c:^2}" for c in cells)
    return (
        f"{ident:<{_GRID_W_ID}} {runner:<{_GRID_W_RUN}} {quant:<{_GRID_W_QUANT}} "
        f"{size_h:>{_GRID_W_SIZE}} {updated:<{_GRID_W_UPD}} {ct_short:<{_GRID_W_CT}} "
        f"{tool_marks}  {hub:^{_GRID_W_HUB}} {decision:<{_GRID_W_DEC}} {bm}"
    )

def format_curation_grid(sections, ad_hoc):
    """Render a per-purpose static text grid of every model in the inventory.

    Rows are grouped by enrichment.purpose, sorted recommended-first via
    the 10-level _grid_sort_key_for_pair hierarchy. Each row shows identity
    + key metadata + a small grid of tool-compat marks:

      ●  this tool currently has the bytes locally
      ✓  this tool can natively load this runner, but the bytes aren't
         in its local store on this machine
      ' ' this tool isn't compatible with this runner
    """
    rows = compute_curation_grid_data(sections, ad_hoc)
    tool_codes = [tc for tc, _ in _TOOL_COLUMNS]
    tool_header = " ".join(f"{tc:^2}" for tc in tool_codes)
    out = ["", "=" * 132, "Curation grid — by purpose, recommended first per group", "=" * 132]
    for r in rows:
        if r["kind"] == "header":
            out.append("")
            out.append(
                f"[{r['purpose'].upper()}] · {r['n']} entries · {_human_size(r['size'])} · "
                f"{r['keepers']} keep · {r['prunes']} prune"
            )
            out.append(
                f"  {'Identity':<{_GRID_W_ID}} {'Runner':<{_GRID_W_RUN}} "
                f"{'Quant':<{_GRID_W_QUANT}} {'Size':>{_GRID_W_SIZE}} "
                f"{'Updated':<{_GRID_W_UPD}} {'CT':<{_GRID_W_CT}} {tool_header}  "
                f"{'Hub':^{_GRID_W_HUB}} {'Decision':<{_GRID_W_DEC}} From"
            )
            out.append("  " + "─" * 150)
        else:
            out.append("  " + _format_grid_row_content(r, tool_codes))
    out.append("")
    out.append("Legend:  ● stored locally in this tool  ·  ✓ compatible (could be loaded if present)")
    out.append("         Hub: ↥ Hub re-uploaded a model file (re-pull would replace bytes)  ·  ♢ chat template specifically changed  ·  ↻ Hub commit changed (likely metadata only)")
    out.append("Tools:   " + " · ".join(f"{tc}={name}" for tc, name in _TOOL_COLUMNS))
    return "\n".join(out)

def mark_for_curation(grid_rows):
    """Interactive picker: navigate the curation grid and mark each row.

    Per-row marker semantics:
      ●  default suggestion: keep    (curation said "keep")
      ○  default suggestion: delete  (curation said "supersede" / "redundant")
      ·  default suggestion: neutral (no curation decision)
      ✓  user explicitly confirms keep
      ✕  user explicitly confirms delete  (the ONLY mark that triggers action)

    Defaults are non-destructive — only ✕ rows are returned for deletion.
    The user has to opt in to every prune.

    Keys:
      ↑ / ↓     navigate (skips section headers)
      SPACE     cycle through default → ✓ → ✕ → default → ...
      Y         set ✓
      N         set ✕
      DEL       reset to default ●/○/·
      ENTER     confirm and return marked-✕ rows
      Ctrl-C    cancel (returns nothing)

    Returns the list of grid_rows (entry rows) the user marked with ✕.
    """
    # Translate curation decisions into the default state per row.
    def _default_state(entry):
        cur = entry.get("curation") or {}
        d = cur.get("decision")
        if d == "keep":
            return "keep"
        if d in ("supersede", "redundant"):
            return "delete"
        return "neutral"

    # Initial state per entry row: None = no user override, show default.
    states = [None] * len(grid_rows)
    defaults = [
        _default_state(r["entry"]) if r["kind"] == "entry" else None
        for r in grid_rows
    ]

    selectable = [i for i, r in enumerate(grid_rows) if r["kind"] == "entry"]
    if not selectable:
        return []
    idx = selectable[0]

    tool_codes = [tc for tc, _ in _TOOL_COLUMNS]
    tool_header = " ".join(f"{tc:^2}" for tc in tool_codes)

    def _marker(i):
        if grid_rows[i]["kind"] != "entry":
            return None
        u = states[i]
        if u == "keep":
            return "✓"
        if u == "delete":
            return "✕"
        d = defaults[i]
        if d == "keep":
            return "●"
        if d == "delete":
            return "○"
        return "·"

    while True:
        print("\033[H\033[J", end="")
        print("Curation — mark for keep ✓ / delete ✕  (only ✕ deletes; defaults ●/○/· are no-ops)")
        print("  ↑/↓ navigate · SPACE cycle · Y keep · N delete · DEL unmark · ENTER confirm · Q or Ctrl+C cancel")
        print("  Hub:  ↥ model file re-uploaded (re-pull would replace bytes)  ·  ♢ chat template specifically changed  ·  ↻ commit changed (likely metadata only)")

        # Live counts at the top so user sees pending impact.
        n_delete = sum(1 for s in states if s == "delete")
        n_keep = sum(1 for s in states if s == "keep")
        n_size_to_delete = sum(
            grid_rows[i]["entry"]["size"] or 0
            for i, s in enumerate(states)
            if s == "delete"
        )
        print(
            f"  marked: ✕ delete = {n_delete} ({_human_size(n_size_to_delete)}), "
            f"✓ keep = {n_keep}"
        )

        window_size = max(8, os.get_terminal_size().lines - 6)
        total = len(grid_rows)
        start = max(0, min(idx - window_size + 4, total - window_size))
        end = min(start + window_size, total)
        for i in range(start, end):
            r = grid_rows[i]
            cursor = ">" if i == idx else " "
            if r["kind"] == "header":
                # Section banner: blank line then bracketed title with stats.
                print()
                print(
                    f"{cursor} [{r['purpose'].upper()}] · {r['n']} entries · "
                    f"{_human_size(r['size'])} · {r['keepers']} keep · {r['prunes']} prune"
                )
                # Inline column header for the section, for orientation.
                print(
                    f"     {'Identity':<{_GRID_W_ID}} {'Runner':<{_GRID_W_RUN}} "
                    f"{'Quant':<{_GRID_W_QUANT}} {'Size':>{_GRID_W_SIZE}} "
                    f"{'Updated':<{_GRID_W_UPD}} {'CT':<{_GRID_W_CT}} {tool_header}  "
                    f"{'Hub':^{_GRID_W_HUB}} {'Decision':<{_GRID_W_DEC}} From"
                )
            else:
                m = _marker(i) or " "
                content = _format_grid_row_content(r, tool_codes)
                print(f"{cursor} {m} {content}")

        key = get_key()
        if key in ("\x1b[A",):  # Up arrow
            new_idx = idx - 1
            while new_idx >= 0 and grid_rows[new_idx]["kind"] != "entry":
                new_idx -= 1
            if new_idx >= 0:
                idx = new_idx
        elif key in ("\x1b[B",):  # Down arrow
            new_idx = idx + 1
            while new_idx < total and grid_rows[new_idx]["kind"] != "entry":
                new_idx += 1
            if new_idx < total:
                idx = new_idx
        elif key in ("\x1b[5~",):  # PageUp
            for _ in range(window_size - 2):
                ni = idx - 1
                while ni >= 0 and grid_rows[ni]["kind"] != "entry":
                    ni -= 1
                if ni < 0:
                    break
                idx = ni
        elif key in ("\x1b[6~",):  # PageDown
            for _ in range(window_size - 2):
                ni = idx + 1
                while ni < total and grid_rows[ni]["kind"] != "entry":
                    ni += 1
                if ni >= total:
                    break
                idx = ni
        elif key == " ":
            # Cycle: None → keep → delete → None
            cur = states[idx]
            states[idx] = {None: "keep", "keep": "delete", "delete": None}[cur]
        elif key.lower() == "y":
            states[idx] = "keep"
        elif key.lower() == "n":
            states[idx] = "delete"
        elif key in ("\x7f", "\x1b[3~"):  # Backspace OR Delete
            states[idx] = None
        elif key == "\r":
            break
        elif key == "\x03" or key.lower() == "q":
            print("\nCancelled. No deletions.")
            return []

    return [
        grid_rows[i] for i, s in enumerate(states)
        if s == "delete" and grid_rows[i]["kind"] == "entry"
    ]

def discover(hub_meta=False, curate=False, interactive=False):
    """Inventory HF-compatible models cached by other apps on this machine."""
    cache_dir = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    hub_dir = cache_dir / "hub"

    # Pre-compute the set of blob sha256 hashes already in HF cache so each
    # Ollama-format scanner can flag dedup-confirmed duplicates.
    hf_blob_hashes = _collect_hf_blob_hashes(hub_dir)

    sections = []
    known_roots = []

    def add(app, paths, scan_fn, note=None, app_install_paths=None):
        present = [p for p in paths if p.exists()]
        installed = any(Path(p).exists() for p in (app_install_paths or []))
        if not present:
            status = "installed; no model storage found" if installed else "not present"
            sections.append({"app": app, "status": status, "paths": paths, "items": [], "note": note})
            return
        items = []
        try:
            for p in present:
                known_roots.append(p)
                items.extend(scan_fn(p))
        except Exception as e:
            sections.append({"app": app, "status": f"scan failed: {e}", "paths": present, "items": [], "note": note})
            return
        sections.append({"app": app, "status": "scanned", "paths": present, "items": items, "note": note})

    add(
        "Hugging Face cache",
        [hub_dir],
        scan_hf_cache,
        note="shared by: mlx-lm, mlx_vlm, mflux, vLLM, transformers, huggingface_hub",
    )

    def _lms_record(pub, name, model_dir, fmt):
        symlinked = is_already_symlinked(model_dir)
        rec = {
            "app": "LM Studio",
            "id": f"{pub}/{name}",
            "path": model_dir,
            "format": fmt,
            "size": _dir_size(model_dir),
            "extra": {"symlinked": symlinked},
            "status": "canonical" if symlinked else "mirror_candidate",
        }
        # When LM Studio entries are symlinked into HF cache, the actual
        # tokenizer config and chat_template live in the snapshot they point
        # at; pass the snapshot path so chat_template_fingerprint succeeds.
        snap = resolve_hf_snapshot(hub_dir, pub, name)
        enrich_entry(rec, snapshot_path=snap or model_dir)
        return rec

    add(
        "LM Studio",
        [resolve_lm_studio_models_dir()],
        lambda p: [_lms_record(*t) for t in scan_lmstudio_models(p)],
        app_install_paths=["/Applications/LM Studio.app"],
    )

    ollama_dir = Path(os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models")))
    add(
        "Ollama",
        [ollama_dir],
        lambda p: scan_ollama_models(p, app_name="Ollama", hf_blob_hashes=hf_blob_hashes),
        app_install_paths=[
            "/Applications/Ollama.app",
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
        ],
    )

    add(
        "GPT4All",
        [
            Path(os.path.expanduser("~/Library/Application Support/nomic.ai/GPT4All")),
            Path(os.path.expanduser("~/.local/share/nomic.ai/GPT4All")),
        ],
        lambda p: scan_flat_dir(p, "GPT4All"),
        app_install_paths=["/Applications/GPT4All.app", "/Applications/gpt4all.app"],
    )

    add(
        "Jan",
        [Path(os.path.expanduser("~/jan/models"))],
        lambda p: scan_flat_dir(p, "Jan"),
        app_install_paths=["/Applications/Jan.app"],
    )

    # Msty family — Msty, Msty Claw, and MstyStudio all use Ollama-format
    # storage (manifests/registry.ollama.ai/ + blobs/sha256-*), so they share
    # scan_ollama_models. Each is a separate app with its own data dir.
    add(
        "Msty",
        [
            Path(os.path.expanduser("~/Library/Application Support/Msty/models")),
            Path(os.path.expanduser("~/.msty/models")),
        ],
        lambda p: scan_ollama_models(p, app_name="Msty", hf_blob_hashes=hf_blob_hashes),
        app_install_paths=["/Applications/Msty.app"],
    )

    add(
        "Msty Claw",
        [Path(os.path.expanduser("~/.mstyclaw/local-ai/models"))],
        lambda p: scan_ollama_models(p, app_name="Msty Claw", hf_blob_hashes=hf_blob_hashes),
        app_install_paths=["/Applications/Msty Claw.app"],
    )

    add(
        "MstyStudio",
        [Path(os.path.expanduser("~/Library/Application Support/MstyStudio/models"))],
        lambda p: scan_ollama_models(p, app_name="MstyStudio", hf_blob_hashes=hf_blob_hashes),
        app_install_paths=["/Applications/MstyStudio.app"],
    )

    add(
        "AnythingLLM",
        [
            Path(os.path.expanduser("~/Library/Application Support/AnythingLLM/storage/models")),
            Path(os.path.expanduser("~/.config/AnythingLLM/storage/models")),
        ],
        lambda p: scan_flat_dir(p, "AnythingLLM"),
        app_install_paths=["/Applications/AnythingLLM.app"],
    )

    comfyui_paths = [
        Path(os.path.expanduser("~/ComfyUI/models")),
        Path(os.path.expanduser("~/Documents/ComfyUI/models")),
    ]
    for install in find_comfyui_installs():
        models = install / "models"
        if models not in comfyui_paths:
            comfyui_paths.append(models)
    add(
        "ComfyUI",
        comfyui_paths,
        lambda p: scan_flat_dir(p, "ComfyUI"),
    )

    draw_things_paths = []
    for candidate in (
        Path(os.path.expanduser("~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models")),
    ):
        draw_things_paths.append(candidate)
    container = find_macos_container_for_bundle("com.liuliu.draw-things")
    if container is not None:
        resolved = container / "Data" / "Documents" / "Models"
        if resolved not in draw_things_paths:
            draw_things_paths.append(resolved)
    add(
        "Draw Things",
        draw_things_paths,
        scan_drawthings_models,
        app_install_paths=["/Applications/Draw Things.app"],
    )

    db_root = Path(os.path.expanduser("~/.diffusionbee"))
    add(
        "Diffusion Bee",
        [db_root],
        lambda p: (
            scan_flat_dir(p / "imported_models", "Diffusion Bee")
            + scan_flat_dir(p / "custom_models", "Diffusion Bee")
            + scan_flat_dir(p / "downloads", "Diffusion Bee")
        ),
        app_install_paths=[
            "/Applications/DiffusionBee.app",
            "/Applications/Diffusion Bee.app",
        ],
    )

    mdfind_results, mdfind_note = scan_mdfind()

    # The status assigned per inferred kind, mirroring kind_order in the
    # report formatter. Sandboxed / app-support / app-cache hits are owned
    # by another app and shouldn't be moved; the rest are gather candidates.
    _AD_HOC_ACTION = {
        "sandbox":         "owned_by_app",
        "app_support":     "owned_by_app",
        "app_cache":       "owned_by_app",
    }
    ad_hoc = []
    already_known = 0
    for path, size in mdfind_results:
        is_known = False
        for root in known_roots:
            try:
                path.relative_to(root)
                is_known = True
                break
            except ValueError:
                continue
        if is_known:
            already_known += 1
            continue
        inferred = infer_source_from_path(path)
        ad_hoc.append({
            "path": path,
            "size": size,
            "format": path.suffix.lower().lstrip("."),
            "inferred": inferred,
            "status": _AD_HOC_ACTION.get(inferred["kind"], "gather_candidate"),
        })

    if hub_meta:
        _enrich_with_hub_metadata(sections, hub_dir=hub_dir)

    print(format_discovery_report(sections, ad_hoc, len(mdfind_results), already_known, mdfind_note))

    if curate:
        if interactive:
            grid_rows = compute_curation_grid_data(sections, ad_hoc)
            marked = mark_for_curation(grid_rows)
            # Move below the picker's last frame.
            print("\033[H\033[J", end="")
            if not marked:
                print("No entries marked for deletion. (Defaults ●/○/· are non-destructive.)")
            else:
                total = sum(m["entry"]["size"] or 0 for m in marked)
                print(f"Marked for deletion: {len(marked)} entr{'y' if len(marked) == 1 else 'ies'} · {_human_size(total)}")
                print()
                for m in marked:
                    e = m["entry"]
                    enr = e.get("enrichment") or {}
                    print(
                        f"  ✕ {_identity_string(enr)[:60]} "
                        f"· {_human_size(e.get('size') or 0)} "
                        f"· {e.get('id') or ''}"
                    )
                print()
                print("(deletion not yet implemented; this is the list that would be removed.)")
        else:
            print(format_curation_grid(sections, ad_hoc))


def main():
    """Entry point: dispatch the import (default), mirror, or discover subcommand."""
    parser = argparse.ArgumentParser(
        prog="lmstudio-hf",
        description="Manage MLX/GGUF models between the Hugging Face cache and LM Studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run lmstudio_hf.py                       # default: import flow\n"
            "  uv run lmstudio_hf.py import                # explicit: import flow\n"
            "  uv run lmstudio_hf.py mirror                # mirror MLX models, smart-reuse on\n"
            "  uv run lmstudio_hf.py mirror --type gguf    # mirror GGUF models only\n"
            "  uv run lmstudio_hf.py mirror --type both    # mirror MLX and GGUF\n"
            "  uv run lmstudio_hf.py mirror --no-reuse     # force full re-download\n"
            "  uv run lmstudio_hf.py discover              # inventory HF-compatible models across local apps\n"
            "\nRun `lmstudio_hf.py <command> --help` for full command options."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", title="commands", metavar="<command>")
    sub.add_parser(
        "import",
        help="Import MLX models from the HF cache into LM Studio (default if no command given).",
        description=(
            "Scan the Hugging Face cache for MLX models and create per-file symlinks "
            "in ~/.cache/lm-studio/models/ so LM Studio can use them."
        ),
    )
    m = sub.add_parser(
        "mirror",
        help="Mirror LM Studio models into the HF cache and replace with symlinks. Flags: --type mlx|gguf|both, --no-reuse.",
        description=(
            "Scan LM Studio's models directory (honoring ~/.lmstudio-home-pointer), "
            "ensure each model is in the Hugging Face cache (downloading from the Hub "
            "when missing, but reusing matching local files via sha256-verified blob "
            "ingestion when possible), then replace each LM Studio model directory with "
            "per-file symlinks pointing into the HF snapshot."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    m.add_argument(
        "--type",
        metavar="<type>",
        choices=["mlx", "gguf", "both"],
        default="mlx",
        help="One of: mlx, gguf, both. Default: mlx (matching the import flow).",
    )
    m.add_argument(
        "--no-reuse",
        action="store_true",
        help="Disable sha256-verified ingestion of LM Studio's existing files into the HF blob store. By default, matching local files are reused so only missing pieces are downloaded.",
    )
    d = sub.add_parser(
        "discover",
        help="List HF-compatible models cached by other apps on this machine.",
        description=(
            "Scan known model-cache locations (Hugging Face, LM Studio, Ollama, "
            "GPT4All, Jan, Msty family, AnythingLLM, ComfyUI, Draw Things, "
            "Diffusion Bee) plus a Spotlight (mdfind) sweep on macOS for "
            "*.safetensors and *.gguf files outside those caches. Read-only: "
            "prints a report, takes no actions. Network calls are opt-in via "
            "--hub-meta."
        ),
    )
    d.add_argument(
        "--hub-meta",
        action="store_true",
        help="Augment HF-derived entries with last_modified date and license from the Hugging Face Hub. One API call per unique repo_id.",
    )
    d.add_argument(
        "--curate",
        action="store_true",
        help="Append a family-grouped curation view sorted most-recommended-first per group, with keep / supersede / redundant decisions. Combine with --hub-meta for date-aware decisions.",
    )
    d.add_argument(
        "--interactive",
        action="store_true",
        help="With --curate, show an interactive picker (cursor + ●/○/✓/✕ markers) "
             "instead of the static grid. SPACE cycles a row through keep/delete/unmark; "
             "Y/N/DEL set state directly; ENTER confirms; Q or Ctrl-C cancels. Only ✕ "
             "marks delete; default ●/○ suggestions are non-destructive.",
    )
    args = parser.parse_args()
    if args.cmd == "mirror":
        types = {"mlx", "gguf"} if args.type == "both" else {args.type}
        mirror_to_huggingface(types, reuse_local=not args.no_reuse)
    elif args.cmd == "discover":
        # --curate implies --hub-meta: curation needs HF's authoritative
        # library_name / tags / pipeline_tag / base_model to make accurate
        # decisions instead of reverse-engineering them from repo names.
        discover(
            hub_meta=args.hub_meta or args.curate,
            curate=args.curate,
            interactive=args.interactive,
        )
    else:
        manage_models()

if __name__ == "__main__":
    main()
