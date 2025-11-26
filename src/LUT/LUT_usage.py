"""Apply a 3D LUT (.cube) to images.

This script supports reading common `.cube` 3D LUT files and applying the LUT
to one image or all images in a folder using trilinear interpolation.

Requirements (already in `requirements.txt`): `numpy`, `Pillow`, `opencv-python`

Examples:
  # Single image
  python3 -m src.LUT.LUT_usage --in img.jpg --lut my_lut.cube --out out.jpg

  # Whole folder
  python3 -m src.LUT.LUT_usage --in images/ --lut my_lut.cube --out out_folder/

"""

from pathlib import Path
import argparse
import sys
import numpy as np
from PIL import Image
import math
import os


def parse_cube(path: Path):
	"""Parse a .cube 3D LUT file into a numpy array of shape (size,size,size,3).

	Returns LUT with float32 values in range [0,1].
	"""
	size = None
	title = None
	entries = []
	with open(path, 'r', encoding='utf-8', errors='ignore') as f:
		for raw in f:
			line = raw.strip()
			if not line or line.startswith('#'):
				continue
			parts = line.split()
			if parts[0].upper() == 'TITLE':
				title = ' '.join(parts[1:]).strip('"')
				continue
			if parts[0].upper() == 'LUT_3D_SIZE':
				size = int(parts[1])
				continue
			if parts[0].upper() in ('DOMAIN_MIN', 'DOMAIN_MAX'):
				# ignoring domain scaling for now (assume 0..1)
				continue
			# otherwise assume three floats for an entry
			if len(parts) >= 3:
				try:
					rgb = [float(parts[0]), float(parts[1]), float(parts[2])]
					entries.append(rgb)
				except ValueError:
					continue

	if size is None:
		# try to deduce size from entries
		n = len(entries)
		cube_size = round(n ** (1 / 3))
		if cube_size ** 3 == n:
			size = cube_size
		else:
			raise ValueError('Could not determine LUT size. Provide a valid .cube file with LUT_3D_SIZE.')

	if len(entries) != size ** 3:
		raise ValueError(f'LUT file has {len(entries)} entries but expected {size**3} for size {size}')

	arr = np.array(entries, dtype=np.float32)
	# .cube typically lists red fastest, then green, then blue (or opposite depending on exporter)
	# Most .cube files follow R fastest, G next, B slowest. We'll reshape accordingly: (B, G, R, 3)
	arr = arr.reshape((size, size, size, 3))
	# Convert to [0,1] if values > 1
	if arr.max() > 1.01:
		arr = np.clip(arr / 255.0, 0.0, 1.0)
	return arr


def apply_3d_lut(image: np.ndarray, lut: np.ndarray):
	"""Apply 3D LUT to image (H,W,3) with values in 0..255 or 0..1.

	Performs trilinear interpolation on the LUT.
	"""
	if image.dtype != np.float32:
		img = image.astype(np.float32) / 255.0
	else:
		img = image.copy()

	h, w, c = img.shape
	if c < 3:
		raise ValueError('Input image must have 3 channels (RGB)')

	size = lut.shape[0]

	# scale image channels to [0, size-1]
	coords = img * (size - 1)

	# compute indices and weights
	x = coords[..., 0]
	y = coords[..., 1]
	z = coords[..., 2]

	x0 = np.floor(x).astype(np.int64)
	y0 = np.floor(y).astype(np.int64)
	z0 = np.floor(z).astype(np.int64)
	x1 = np.clip(x0 + 1, 0, size - 1)
	y1 = np.clip(y0 + 1, 0, size - 1)
	z1 = np.clip(z0 + 1, 0, size - 1)

	xd = (x - x0)[..., None]
	yd = (y - y0)[..., None]
	zd = (z - z0)[..., None]

	# fetch 8 corner values from LUT: note lut is (size,size,size,3) with ordering [b,g,r] or [r,g,b]
	# Our parse used the file order; if colors look swapped, user can swap channels after.

	c000 = lut[z0, y0, x0]
	c001 = lut[z0, y0, x1]
	c010 = lut[z0, y1, x0]
	c011 = lut[z0, y1, x1]
	c100 = lut[z1, y0, x0]
	c101 = lut[z1, y0, x1]
	c110 = lut[z1, y1, x0]
	c111 = lut[z1, y1, x1]

	c00 = c000 * (1 - xd) + c001 * xd
	c01 = c010 * (1 - xd) + c011 * xd
	c10 = c100 * (1 - xd) + c101 * xd
	c11 = c110 * (1 - xd) + c111 * xd

	c0 = c00 * (1 - yd) + c01 * yd
	c1 = c10 * (1 - yd) + c11 * yd

	out = c0 * (1 - zd) + c1 * zd

	out = np.clip(out, 0.0, 1.0)
	out_img = (out * 255.0).astype(np.uint8)
	return out_img


def process_path(in_path: Path, out_path: Path, lut: np.ndarray):
	if in_path.is_dir():
		out_path.mkdir(parents=True, exist_ok=True)
		imgs = [p for p in sorted(in_path.iterdir()) if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')]
		if not imgs:
			print('No supported images found in', in_path)
			return
		for p in imgs:
			dst = out_path / p.name
			try:
				img = np.array(Image.open(p).convert('RGB'))
				out = apply_3d_lut(img, lut)
				Image.fromarray(out).save(dst)
				print('Wrote', dst)
			except Exception as e:
				print('Failed', p, e)
	else:
		out_path.parent.mkdir(parents=True, exist_ok=True)
		img = np.array(Image.open(in_path).convert('RGB'))
		out = apply_3d_lut(img, lut)
		Image.fromarray(out).save(out_path)
		print('Wrote', out_path)


def main():
	parser = argparse.ArgumentParser(description='Apply 3D LUT (.cube) to image or folder')
	parser.add_argument('--in', dest='in_path', required=True, help='Input image or folder')
	parser.add_argument('--lut', required=True, help='Path to .cube LUT file')
	parser.add_argument('--out', required=True, help='Output image or folder')
	parser.add_argument('--swap-channels', action='store_true', help='Swap LUT channels from RGB->BGR if colors look off')
	args = parser.parse_args()

	in_path = Path(args.in_path)
	lut_path = Path(args.lut)
	out_path = Path(args.out)

	if not lut_path.exists():
		print('LUT file not found:', lut_path)
		sys.exit(1)

	lut = parse_cube(lut_path)
	# If user requested channel swap, flip LUT channels
	if args.swap_channels:
		lut = lut[..., ::-1]

	process_path(in_path, out_path, lut)


if __name__ == '__main__':
	main()
