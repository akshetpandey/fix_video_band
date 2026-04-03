# CLAUDE.md

## Project overview

`fix-video-band` is a CLI tool that detects and repairs stuck-pixel vertical bands in video files using OpenCV inpainting.

## Development

- Python 3.13+, managed with **uv**
- Run the tool: `uv run fix-video-band <input> <output>`
- Run tests/linting: `uv run ruff check .` and `uv run mypy .`
- Format: `uv run ruff format .`

## Project structure

- `src/fix_video_band/main.py` — CLI entry point, 3-phase pipeline (scene detect, band estimate, frame repair)
- `src/fix_video_band/lib.py` — Core detection/repair algorithms (band detection, inpainting, scene cuts)
- `src/fix_video_band/cli.py` — Terminal display (iTerm2/Kitty inline image protocols, frame annotation)
- `doc/` — Example input video (`input.m4v`, gitignored) and screenshots for README

## Key dependencies

- `opencv-python` — frame I/O, inpainting, image annotation
- `numpy` — statistical band detection (rolling median, MAD-based Z-scores)
- `rich` — terminal UI (progress bars, panels, tables)
- `ffmpeg` — must be on PATH; used for scene detection and audio muxing

## Notes

- `*.m4v` files are gitignored; `doc/input.m4v` is a sample video not tracked in git
- The tool requires iTerm2, Ghostty, WezTerm, or Konsole for inline image previews; other terminals work but skip visual output
