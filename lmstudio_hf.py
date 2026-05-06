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
    """Sum file sizes under path, following symlinks. Tolerates errors."""
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except Exception:
        pass
    return total

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
        out.append({
            "app": "Hugging Face cache",
            "id": f"{publisher}/{repo}",
            "path": snapshot,
            "format": fmt,
            "size": _dir_size(entry),
        })
    return out

def scan_ollama_models(ollama_dir):
    """Walk Ollama manifests and yield (family:tag, gguf_blob, size) records."""
    out = []
    manifests_root = ollama_dir / "manifests"
    if not manifests_root.exists():
        return out
    # Layout: manifests/<registry>/<namespace>/<family>/<tag>
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
        blob = ollama_dir / "blobs" / digest.replace(":", "-")
        try:
            size = blob.stat().st_size
        except OSError:
            continue
        family = tag_path.parent.name
        tag = tag_path.name
        out.append({
            "app": "Ollama",
            "id": f"{family}:{tag}",
            "path": blob,
            "format": "gguf",
            "size": size,
        })
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
        out.append({
            "app": app_name,
            "id": path.name,
            "path": path,
            "format": path.suffix.lower().lstrip("."),
            "size": size,
        })
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

    return {"kind": "unclassified", "label": None, "evidence": _friendly(path.parent)}

def _format_known_section(section):
    """Render one known-cache section."""
    app = section["app"]
    status = section["status"]
    paths = ", ".join(_friendly(p) for p in section["paths"])
    if status == "not present":
        return [f"[{app}]   not present"]
    if status.startswith("scan failed"):
        return [f"[{app}]   {paths}   {status}"]
    items = section["items"]
    total = sum(i["size"] for i in items)
    label = "repo" if app == "Hugging Face cache" else "model"
    plural = "" if len(items) == 1 else "s"
    out = [f"[{app}]   {paths}   {len(items)} {label}{plural} · {_human_size(total)}"]
    if section.get("note"):
        out.append(f"  {section['note']}")
    for i in items:
        extra = ""
        if i.get("extra", {}).get("symlinked"):
            extra = ", symlinked → HF cache"
        out.append(f"  - {i['id']} ({i['format']}, {_human_size(i['size'])}{extra})")
    return out

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
    kind_order = [
        ("sandbox",         "Sandboxed apps (containers)"),
        ("app_support",     "App Support directories"),
        ("app_cache",       "App Caches"),
        ("hf_format",       "HF-cache-format at non-canonical locations"),
        ("tool_named",      "Tool-named directories"),
        ("user_curated",    "User-curated personal storage"),
        ("user_download",   "Manual downloads (unsorted)"),
        ("cloud_sync",      "Cloud-synced (materialized locally)"),
        ("external_volume", "External volumes"),
        ("unclassified",    "Truly unclassified"),
    ]
    for kind, heading in kind_order:
        files = by_kind.get(kind)
        if not files:
            continue
        out.append("")
        out.append(f"  {heading}:")
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
    return "\n".join(lines)

def discover():
    """Inventory HF-compatible models cached by other apps on this machine."""
    cache_dir = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    hub_dir = cache_dir / "hub"

    sections = []
    known_roots = []

    def add(app, paths, scan_fn, note=None):
        present = [p for p in paths if p.exists()]
        if not present:
            sections.append({"app": app, "status": "not present", "paths": paths, "items": [], "note": note})
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

    add(
        "LM Studio",
        [resolve_lm_studio_models_dir()],
        lambda p: [
            {
                "app": "LM Studio",
                "id": f"{pub}/{name}",
                "path": model_dir,
                "format": fmt,
                "size": _dir_size(model_dir),
                "extra": {"symlinked": is_already_symlinked(model_dir)},
            }
            for pub, name, model_dir, fmt in scan_lmstudio_models(p)
        ],
    )

    ollama_dir = Path(os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models")))
    add("Ollama", [ollama_dir], scan_ollama_models)

    add(
        "GPT4All",
        [
            Path(os.path.expanduser("~/Library/Application Support/nomic.ai/GPT4All")),
            Path(os.path.expanduser("~/.local/share/nomic.ai/GPT4All")),
        ],
        lambda p: scan_flat_dir(p, "GPT4All"),
    )

    add("Jan", [Path(os.path.expanduser("~/jan/models"))], lambda p: scan_flat_dir(p, "Jan"))

    add("Msty", [Path(os.path.expanduser("~/.msty/models"))], lambda p: scan_flat_dir(p, "Msty"))

    add(
        "AnythingLLM",
        [
            Path(os.path.expanduser("~/Library/Application Support/AnythingLLM/storage/models")),
            Path(os.path.expanduser("~/.config/AnythingLLM/storage/models")),
        ],
        lambda p: scan_flat_dir(p, "AnythingLLM"),
    )

    add(
        "ComfyUI",
        [
            Path(os.path.expanduser("~/ComfyUI/models")),
            Path(os.path.expanduser("~/Documents/ComfyUI/models")),
        ],
        lambda p: scan_flat_dir(p, "ComfyUI"),
    )

    add(
        "Draw Things",
        [Path(os.path.expanduser(
            "~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models"))],
        lambda p: scan_flat_dir(p, "Draw Things"),
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
    )

    mdfind_results, mdfind_note = scan_mdfind()

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
        ad_hoc.append({
            "path": path,
            "size": size,
            "format": path.suffix.lower().lstrip("."),
            "inferred": infer_source_from_path(path),
        })

    print(format_discovery_report(sections, ad_hoc, len(mdfind_results), already_known, mdfind_note))


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
    sub.add_parser(
        "discover",
        help="List HF-compatible models cached by other apps on this machine.",
        description=(
            "Scan known model-cache locations (Hugging Face, LM Studio, Ollama, "
            "GPT4All, Jan, Msty, AnythingLLM, ComfyUI, Draw Things, Diffusion Bee) "
            "plus a Spotlight (mdfind) sweep on macOS for *.safetensors and *.gguf "
            "files outside those caches. Read-only: prints a report, takes no "
            "actions, makes no network calls."
        ),
    )
    args = parser.parse_args()
    if args.cmd == "mirror":
        types = {"mlx", "gguf"} if args.type == "both" else {args.type}
        mirror_to_huggingface(types, reuse_local=not args.no_reuse)
    elif args.cmd == "discover":
        discover()
    else:
        manage_models()

if __name__ == "__main__":
    main()
