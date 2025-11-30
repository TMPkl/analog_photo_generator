import argparse
import json
import os
from pathlib import Path
import sys
import cv2 as cv
import numpy as np
from PIL import Image

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


def run_pipeline_for_file(input_file: Path, args, working_directory: Path):
	results = {}

	lut_mod = load_module_from_path('lut', working_directory.joinpath('PythonScripts', 'lut.py'))
	
	lut_files = {
		1: os.path.join(working_directory, "PythonScripts", "LUT_files", "LUT1.cube"),
		2: os.path.join(working_directory, "PythonScripts", "LUT_files", "LUT2.cube")
	}

	lut_choice = int(args.lut)
	if lut_choice not in lut_files.keys():
		raise ValueError('LUT index must be 1 or 2')
		return

	lut_path = lut_files[lut_choice]

	lut_out_dir = working_directory.joinpath('wwwroot', 'output_lut')
	lut_out_dir.mkdir(parents=True, exist_ok=True)

	output_path = str(lut_out_dir) + '/' + str(input_file.stem) + "_" + str(lut_choice) + ".png"
    
	lut_path = Path(lut_path)

	lut = lut_mod.parse_cube(lut_path)

	lut = lut[..., ::-1]

	img = np.array(Image.open(input_file).convert('RGB'))
	out = lut_mod.apply_3d_lut(img, lut)
	Image.fromarray(out).save(output_path)

	results['lut_output'] = Path(output_path).name

	# 2) Haliation step — use existing haliation module functions
	hal_mod = load_module_from_path('haliation', working_directory.joinpath('PythonScripts', 'haliation.py'))

	img = cv.imread(str(output_path))
	if img is None:
		raise FileNotFoundError(f"Model output not found: {output_path}")

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
	grain_out_dir = working_directory.joinpath('wwwroot', 'output_pipeline_lut')
	grain_out_dir.mkdir(parents=True, exist_ok=True)
	grain_name = base + "_" + "-".join(scales_parts) + "_" + str(args.grain_intensity) + "_" + str(args.grain_amplitude) + ".png"
	grain_path = grain_out_dir / grain_name
	saved = cv.cvtColor(grain_img, cv.COLOR_HLS2BGR_FULL)
	cv.imwrite(str(grain_path), saved)
	results['pipeline_output'] = grain_path.name
	results['path'] = grain_path

	return results


def main():
	working_directory = Path(os.getcwd())

	parser = argparse.ArgumentParser(description='Pipeline: AB/LAB model -> haliation -> grain')
	parser.add_argument('input_path', help='Input image file or folder')
	parser.add_argument('--lut', default='1', help='Index of LUT file to run')

	# haliation params (defaults match haliation.py)
	parser.add_argument('--bright_threshold', type=int, default=243)
	parser.add_argument('--kernel_size', type=int, default=55)
	parser.add_argument('--sigma_x', type=int, default=40)
	parser.add_argument('--delta_mode', type=int, default=0)
	parser.add_argument('--haliation_intensity', type=float, default=0.1)

	# grain params (defaults match grain.py)
	parser.add_argument('--scale', nargs='+', default="1 0.2 0.4", help="Scales for multiscale noise, e.g. --scale 1 0.2 0.4")
	parser.add_argument('--grain_intensity', type=float, default=0.4)
	parser.add_argument('--grain_amplitude', type=float, default=0.18)

	args = parser.parse_args()

	
	input_path = Path(args.input_path)
	processed = []

	if input_path.is_dir():
		exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
		files = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in exts]
		for f in files:
			res = run_pipeline_for_file(f, args, working_directory)
			processed.append(res)
	else:
		res = run_pipeline_for_file(input_path, args, working_directory)
		processed.append(res)

	# Output JSON: if single input -> return single filenames dict, else list
	if len(processed) == 1:	
		print(json.dumps({'filename': processed[0]['pipeline_output']}))
	else:
		print(json.dumps({'results': processed}))


if __name__ == '__main__':
	main()
