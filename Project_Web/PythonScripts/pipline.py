import argparse
import json
import os
from pathlib import Path
import sys
import cv2 as cv

import importlib.util


def load_module_from_path(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, str(path))
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def parse_scales_arg(scale_arg):
	# Accept either a single string or a list
	if isinstance(scale_arg, list):
		# list of strings
		return scale_arg
	if isinstance(scale_arg, str):
		parts = [p for p in scale_arg.replace(',', ' ').split() if p != '']
		return parts
	return [str(scale_arg)]


def run_pipeline_for_file(input_file: Path, model_choice: str, model_mod, model_obj, device, args, working_directory: Path):
	results = {}

	# 1) Run model (AB or LAB)
	if model_choice == 'AB':
		model_out_dir = working_directory.joinpath('wwwroot', 'output_GAN_AB')
		model_out_dir.mkdir(parents=True, exist_ok=True)
		out_name = model_mod.process_file(input_file, model_obj, device, model_out_dir)
		model_output_path = model_out_dir / out_name
	else:
		model_out_dir = working_directory.joinpath('wwwroot', 'output_GAN_LAB')
		model_out_dir.mkdir(parents=True, exist_ok=True)
		out_name = model_mod.process_file(input_file, model_obj, device, model_out_dir)
		model_output_path = model_out_dir / out_name

	results['model_output'] = model_output_path.name

	# 2) Haliation step — use existing haliation module functions
	hal_mod = load_module_from_path('haliation', working_directory.joinpath('PythonScripts', 'haliation.py'))

	img = cv.imread(str(model_output_path))
	if img is None:
		raise FileNotFoundError(f"Model output not found: {model_output_path}")

	img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
	ls = hal_mod.light_source_detection_hsv(img_HSV, bright_threshold=args.bright_threshold)
	hm = hal_mod.haliation_map_generator(ls, kernel_size=args.kernel_size, sigmaX=args.sigma_x, delta_mode=args.delta_mode)
	haliated = hal_mod.add_heliation_effect(img, hm, intensity=args.haliation_intensity)

	# save haliation output similar to haliation.py naming
	haliation_out_dir = working_directory.joinpath('wwwroot', 'output_haliation')
	haliation_out_dir.mkdir(parents=True, exist_ok=True)
	base = input_file.stem
	haliation_name = base + "_" + str(args.bright_threshold) + "_" + str(args.kernel_size) + "_" + str(args.sigma_x) + "_" + str(args.delta_mode) + "_" + str(args.haliation_intensity) + ".png"
	haliation_path = haliation_out_dir / haliation_name
	cv.imwrite(str(haliation_path), haliated)
	results['haliation_output'] = haliation_path.name

	# 3) Grain step — use grain module
	grain_mod = load_module_from_path('grain', working_directory.joinpath('PythonScripts', 'grain.py'))

	# convert to HLS and call add_multiscale_grain
	haliated_hls = cv.cvtColor(haliated, cv.COLOR_BGR2HLS_FULL)
	scales_parts = parse_scales_arg(args.scale)
	# grain_mod expects a Scales object or iterable; try to construct Scales
	try:
		if hasattr(grain_mod, 'Scales'):
			if len(scales_parts) == 1:
				scales_obj = grain_mod.Scales.from_string(scales_parts[0])
			else:
				scales_obj = grain_mod.Scales(scales_parts)
		else:
			scales_obj = tuple(float(s) for s in scales_parts)
	except Exception:
		scales_obj = tuple(float(s) for s in scales_parts)

	grain_img = grain_mod.add_multiscale_grain(haliated_hls, scales=scales_obj, intensity=args.grain_intensity, grain_amplitude=args.grain_amplitude)
	grain_out_dir = working_directory.joinpath('wwwroot', 'output_grain')
	grain_out_dir.mkdir(parents=True, exist_ok=True)
	grain_name = base + "_" + "-".join(scales_parts) + "_" + str(args.grain_intensity) + "_" + str(args.grain_amplitude) + ".png"
	grain_path = grain_out_dir / grain_name
	saved = cv.cvtColor(grain_img, cv.COLOR_HLS2BGR_FULL)
	cv.imwrite(str(grain_path), saved)
	results['grain_output'] = grain_path.name

	return results


def main():
	working_directory = Path(os.getcwd())
	parser = argparse.ArgumentParser(description='Pipeline: AB/LAB model -> haliation -> grain')
	parser.add_argument('input_path', help='Input image file or folder')
	parser.add_argument('--model', choices=['AB', 'LAB'], default='AB', help='Choose model by phrasing: AB or LAB')
	parser.add_argument('--checkpoint', '-c', default=None, help='Checkpoint path for the chosen model')
	parser.add_argument('--device', '-d', choices=['cpu', 'cuda'], default='cpu', help='Device to run model on')

	# haliation params (defaults match haliation.py)
	parser.add_argument('--bright_threshold', type=int, default=243)
	parser.add_argument('--kernel_size', type=int, default=55)
	parser.add_argument('--sigma_x', type=int, default=40)
	parser.add_argument('--delta_mode', type=int, default=0)
	parser.add_argument('--haliation_intensity', type=float, default=0.1)

	# grain params (defaults match grain.py)
	parser.add_argument('--scale', nargs='+', default=["1", "0.2", "0.4"], help="Scales for multiscale noise, e.g. --scale 1 0.2 0.4")
	parser.add_argument('--grain_intensity', type=float, default=0.4)
	parser.add_argument('--grain_amplitude', type=float, default=0.18)

	args = parser.parse_args()

	# prepare model module and default checkpoint
	py_dir = working_directory.joinpath('PythonScripts')
	if args.model == 'AB':
		mod_path = py_dir.joinpath('AB_model_usage.py')
		default_ckpt = working_directory.joinpath('Project_Web', 'PythonScripts', 'GAN_models', 'best_checkpoint_AB_e4.pth') if False else working_directory.joinpath('Project_Web', 'PythonScripts', 'GAN_models', 'best_checkpoint_AB_e4.pth')
		# try local GAN_models in PythonScripts
		alt = py_dir.joinpath('GAN_models', 'best_checkpoint_AB_e4.pth')
		if alt.exists():
			default_ckpt = alt
		model_mod = load_module_from_path('AB_model_usage', mod_path)
	else:
		mod_path = py_dir.joinpath('LAB_model_usage.py')
		alt = py_dir.joinpath('GAN_models', 'best_checkpoint_LAB_e4.pth')
		default_ckpt = alt if alt.exists() else working_directory.joinpath('Project_Web', 'PythonScripts', 'GAN_models', 'best_checkpoint_LAB_e4.pth')
		model_mod = load_module_from_path('LAB_model_usage', mod_path)

	checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_ckpt

	device = 'cpu'
	model_obj = model_mod.load_model_from_checkpoint(checkpoint_path, device)

	input_path = Path(args.input_path)
	processed = []

	if input_path.is_dir():
		exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
		files = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in exts]
		for f in files:
			res = run_pipeline_for_file(f, args.model, model_mod, model_obj, device, args, py_dir)
			processed.append(res)
	else:
		res = run_pipeline_for_file(input_path, args.model, model_mod, model_obj, device, args, py_dir)
		processed.append(res)

	# Output JSON: if single input -> return single filenames dict, else list
	if len(processed) == 1:
		print(json.dumps(processed[0]))
	else:
		print(json.dumps({'results': processed}))


if __name__ == '__main__':
	main()
