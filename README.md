# LM Studio - Hugging Face Model Manager

A command-line utility to manage MLX models between your Hugging Face cache and LM Studio. This tool makes it easy to import and manage models you've downloaded from Hugging Face into LM Studio.

## Features

- Interactive model selection interface with keyboard navigation
- Two flows: `import` (HF cache → LM Studio) and `mirror` (LM Studio → HF cache, then symlink back)
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
uv run lmstudio_hf.py mirror            # mirror flow, MLX only (default)
uv run lmstudio_hf.py mirror --type gguf
uv run lmstudio_hf.py mirror --type both
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

1. Scans `~/.cache/lm-studio/models/<publisher>/<name>/` for MLX (and optionally GGUF) models
2. For each model, classifies as: already symlinked, in HF cache, downloadable from the Hub, or not on the Hub
3. Presents the actionable subset in the picker; you choose which to mirror
4. Downloads any missing snapshots into the HF cache via `huggingface_hub.snapshot_download`
5. Atomically replaces each chosen LM Studio model directory with per-file symlinks pointing into the HF snapshot (the original is held under a `<name>.old` sibling until the new tree lands, then removed)

Peak temporary disk usage during a mirror is roughly the size of the model being processed, since the original files stay in place until the new symlink tree is built.

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
