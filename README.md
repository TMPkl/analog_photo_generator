# Analog Photo Generator

Computer-vision toolkit for generating analog film aesthetics from digital photos. The project ships practical, composable tools (3D LUT application, halation/halo glow, multi-scale film grain, and two GAN-based transforms: AB and LAB) plus ready-to-use pipelines that chain them. A minimal web app exists only as a convenience UI; this README focuses on the CV tools. For the web app, see `Project_Web/README.md`.

## Presentation

This project was presented for the course Computer Vision at Sapienza University of Rome. The presentation slides can be found [here](https://www.canva.com/design/DAG5KI0-7gY/4ABQWTeXh79KLcTR8c9ZbA/edit?utm_content=DAG5KI0-7gY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

## How It Works

- Modular image operators written in Python (OpenCV + NumPy + Pillow) operating mostly in RGB/BGR or HLS/HSV color spaces.
- File-in, file-out model: each tool reads one image and writes a processed PNG to a dedicated `output_*` folder.
- Every CLI tool prints a tiny JSON payload to stdout (e.g., `{ "filename": "...png" }`) to make orchestration simple.
- Two pipelines compose operators in a fixed order for fast experimentation.

![Pipeline](media/readme/nb.png "Pipeline overview")

## Core Tools

- 3D LUT (`Project_Web/PythonScripts/lut.py`)
	- Parses `.cube` LUTs and applies them with trilinear interpolation on linearized RGB.
	- Input: RGB image. Options: `--lut {1|2}` selecting a file from `LUT_files`.
	- Output: `wwwroot/output_lut/<name>_<lut>.png`.

- Halation (`Project_Web/PythonScripts/haliation.py`)
	- Workflow: detect bright areas in HSV (`V > threshold`) → Gaussian blur per channel in BGR → blend back.
	- Key parameters: `--bright_threshold`, `--kernel_size`, `--sigma_x`, `--delta_mode`, `--intensity`.
	- Output: `wwwroot/output_haliation/<name>_<params>.png`.

- Film Grain (`Project_Web/PythonScripts/grain.py`)
	- Adds multi-scale Gaussian noise in HLS with channel-wise scaling, then blends with weight `--intensity`.
	- Key parameters: `--scale` (e.g. `1 0.2 0.4`), `--intensity`, `--grain_amplitude`.
	- Output: `wwwroot/output_grain/<name>_<scales>_<intensity>_<ampl>.png`.

- GAN Transforms AB/LAB (`Project_Web/PythonScripts/gan_ab.py`, `gan_lab.py`)
	- Loads pre-trained checkpoints from `PythonScripts/GAN_models/` and produces a stylized output.
	- Intended as a learned “film stock look” step that can precede halation/grain.
	- Outputs to `wwwroot/output_GAN_AB` or `wwwroot/output_GAN_LAB`.
	- Developed originally in: https://github.com/Szymon-Stasiak/cycle-gan-for-generating-analog-photo-color. This repository only carries the minimum files and dependencies needed to run inference.

## Pipelines

- LUT Pipeline (`Project_Web/PythonScripts/pipeline_lut.py`)
	- Order: LUT → Halation → Grain.
	- Emits a single final PNG in `wwwroot/output_pipeline_lut/` and JSON `{ "filename": "..." }`.

- GAN Pipeline (`Project_Web/PythonScripts/pipeline_gan.py`)
	- Order: GAN (AB or LAB) → Halation → Grain. Select model via `--model AB|LAB`.
	- Emits a single final PNG in `wwwroot/output_pipeline_gan/` and JSON `{ "filename": "..." }`.

## Utilities

- Color-space helpers (`lib/myFilm/colorSpace.py`): simple XYZ↔BGR converters used in experiments.

## Repository Layout (relevant parts)

- `Project_Web/PythonScripts/` – all production tools, models and LUTs
	- `GAN_models/` – pre-trained checkpoints (`*.pth`)
	- `LUT_files/` – LUTs (`LUT1.cube`, `LUT2.cube`)
	- `*.py` – tools and pipelines described above
- `Project_Web/wwwroot/` – outputs (`output_*`) and uploads
- `lib/`, `src/` – experimental code and early prototypes

For the optional web UI and DB seeding, see `Project_Web/README.md`.

## Quick Run (CLI)

Minimal steps to try the tools. Use Python 3.10; commands assume Linux bash.

```bash
# Set up a venv that the web/UI also uses (optional, but convenient)
cd Project_Web
python3 ./PythonScripts/activate_venv.py

# Apply LUT 1
.venv/bin/python PythonScripts/lut.py /abs/path/to/image.jpg --lut 1

# Halation
.venv/bin/python PythonScripts/haliation.py /abs/path/to/image.jpg \
	--bright_threshold 243 --kernel_size 55 --sigma_x 40 --delta_mode 0 --intensity 0.1

# Film grain
.venv/bin/python PythonScripts/grain.py /abs/path/to/image.jpg \
	--scale 1 0.2 0.4 --intensity 0.4 --grain_amplitude 0.18

# LUT pipeline
.venv/bin/python PythonScripts/pipeline_lut.py /abs/path/to/image.jpg --lut 1

# GAN pipeline (AB)
.venv/bin/python PythonScripts/pipeline_gan.py /abs/path/to/image.jpg --model AB
```

Outputs are written under `Project_Web/wwwroot/output_*` and the tool prints a JSON with the file name.

## Notes

- Ensure `PythonScripts/GAN_models/` and `PythonScripts/LUT_files/` contain the referenced assets.
- The root `requirements.txt` captures broader research deps; the production venv is provisioned via `activate_venv.py`.
- For the web UI, database, and seeding details, read `Project_Web/README.md`.

—

Photo (`media/tests/p2.jpg`) by Ehsan Ahmadi on Unsplash.

