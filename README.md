# LM Studio - Hugging Face Model Manager

A command-line utility to manage MLX models between your Hugging Face cache and LM Studio. This tool makes it easy to import and manage models you've downloaded from Hugging Face into LM Studio.

## Features

- Interactive model selection interface with keyboard navigation
- Three flows: `import` (HF cache → LM Studio), `mirror` (LM Studio → HF cache, then symlink back), `discover` (read-only inventory across local apps)
- Automatic detection of MLX (and optionally GGUF) models
- Smart handling of model imports via per-file symbolic links
- Support for model replacement and removal
- Terminal-based UI with scrolling for large model lists

## Prerequisites

- Python 3.x
- LM Studio installed
- [`uv`](https://docs.astral.sh/uv/) (for dependency resolution via the script's PEP 723 header)
- Hugging Face models downloaded locally (only needed for the `import` flow)

## Installation

You can skip this if you are running the script remotely via `uv run` (see below).

1. Clone this repository:
```bash
git clone https://github.com/ivanfioravanti/lmstudio_hf.git
cd lmstudio_hf
```

## Usage

Run the script with `uv run` so dependencies declared in the script's PEP 723 header
(currently `huggingface_hub`) are resolved automatically:

```bash
uv run lmstudio_hf.py                   # default: import flow
uv run lmstudio_hf.py import            # explicit: import flow
uv run lmstudio_hf.py mirror            # mirror flow, MLX only, smart-reuse on
uv run lmstudio_hf.py mirror --type gguf
uv run lmstudio_hf.py mirror --type both
uv run lmstudio_hf.py mirror --no-reuse # always re-download, skip local hash check
uv run lmstudio_hf.py discover          # inventory HF-compatible models across local apps
```

Remote execution against the raw script URL also works:

```bash
uv run https://raw.githubusercontent.com/ivanfioravanti/lmstudio_hf/main/lmstudio_hf.py mirror
```

### Subcommands

- **`import`** (default) — scans your Hugging Face cache for MLX models and creates
  symlinks in `~/.cache/lm-studio/models/` so LM Studio can use them.
- **`mirror`** — scans `~/.cache/lm-studio/models/` for models, ensures each is in
  the Hugging Face cache (downloading from the Hub if missing), then replaces each
  LM Studio model directory with per-file symlinks pointing into the HF snapshot.
  Defaults to MLX only; pass `--type gguf` or `--type both` to widen scope. Models
  already symlinked are skipped, and models not present on the Hub are reported and
  left untouched.
- **`discover`** — read-only inventory of HF-compatible models cached by other apps
  on this machine. Stats known cache locations (Hugging Face, LM Studio, Ollama,
  GPT4All, Jan, Msty, AnythingLLM, ComfyUI, Draw Things, Diffusion Bee), plus a
  Spotlight (`mdfind`) sweep on macOS for `*.safetensors`/`*.gguf` files outside
  those caches. Classifies leftover hits by parent-directory pattern (sandboxed
  apps, app support dirs, iCloud Drive, external volumes, etc.). No network calls,
  no file modifications.

### Navigation Controls

- ↑/↓ arrows: Navigate through the model list (the first row is a "Select all" toggle)
- SPACE: Select/deselect the current row; on the "Select all" row, toggles every model at once
- ENTER: Confirm selection and proceed
- Ctrl+C: Cancel operation

## How It Works

### `import` flow

1. Scans your Hugging Face cache directory (`~/.cache/huggingface` by default)
2. Identifies MLX-compatible models (`mlx-community` namespace)
3. Creates symbolic links in the LM Studio models directory (`~/.cache/lm-studio/models`)
4. Allows for easy management of existing imports

### `mirror` flow

1. Scans `~/.cache/lm-studio/models/<publisher>/<name>/` for MLX (and optionally GGUF) models.
2. For each model, classifies as: already symlinked, in HF cache, downloadable from the Hub, or not on the Hub.
3. Presents the actionable subset in the picker; you choose which to mirror.
4. **Smart reuse (default).** For each chosen model, queries the Hub for file metadata, then for every local LFS file (e.g. `*.safetensors`) whose size matches the Hub's record, computes its sha256 and compares against the Hub's expected `lfs.sha256`. Files that match are renamed into the HF cache blob layout (`hub/models--<publisher>--<name>/blobs/<sha256>`) with a relative symlink in `snapshots/<sha>/`. No re-download for matched bytes. Files that don't match (or aren't LFS) fall through to step 5.
5. Calls `huggingface_hub.snapshot_download`, which skips files already in the blob store and fetches only what's missing — typically just small metadata files (`config.json`, `tokenizer.json`, etc.) when ingestion succeeded.
6. Atomically replaces each chosen LM Studio model directory with per-file symlinks pointing into the HF snapshot. The original is held under a `<name>.old` sibling until the new symlink tree lands, then removed. Models are processed one at a time, so peak temporary disk usage is roughly one model's worth — not the full batch.

Pass `--no-reuse` to skip the hash-and-ingest step and force a full re-download. Useful if you suspect local files are corrupt.

### `discover` flow

1. Stats a registry of known cache locations:
   - Hugging Face cache (`$HF_HOME` or `~/.cache/huggingface/hub/`) — also covers `mlx-lm`, `mlx_vlm`, `mflux`, `vLLM`, and any other `huggingface_hub`-based tool, since they all share this cache.
   - LM Studio (resolved via `~/.lmstudio-home-pointer`).
   - Ollama (`$OLLAMA_MODELS` or `~/.ollama/models/`).
   - GPT4All, Jan, Msty, AnythingLLM, ComfyUI, Draw Things, Diffusion Bee — each at its conventional path on macOS / Linux / Windows.
2. Runs each cache's scanner (HF cache walks `models--<pub>--<name>/`; Ollama walks manifests to find GGUF blobs; the rest do a recursive file scan with size threshold) and reports the contents.
3. On macOS, runs `mdfind` once for `*.safetensors`/`*.gguf` files larger than 10 MB. Anything that lives under one of the known cache roots is dropped (already accounted for); the rest is classified by path pattern:
   - sandboxed apps (`~/Library/Containers/<bundle-id>/...`)
   - iCloud Drive (legacy `~/Library/Mobile Documents/...` and modern `~/Library/CloudStorage/...`)
   - app support / cache dirs
   - HF-cache-format blob layouts dropped at non-canonical locations
   - tool-named directories (substring match: `comfyui`, `ollama`, `mlx`, `llama.cpp`, etc.)
   - external volumes (`/Volumes/...`)
   - common user folders (`Downloads`, `Desktop`, `Documents/...`)
   - truly unclassified
4. Prints a sectioned report. No network calls, no file modifications.

`discover` is read-only by design — it's an inventory tool, not an action tool. Future versions may pipe its output into `mirror` for cross-app deduplication.

## Environment Variables

- `HF_HOME`: Optional. Set this to customize your Hugging Face cache location.

## Notes

- Models are imported using symbolic links to save disk space
- Already imported models are marked in the selection interface
- Selecting an already imported model will remove it from LM Studio

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

[MIT License](LICENSE)
