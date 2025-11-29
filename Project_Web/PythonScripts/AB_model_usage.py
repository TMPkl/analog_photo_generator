import argparse
import json
import os
from pathlib import Path
import torch
from model.color_cyclegan_model_AB import ColorCycleGANModel
from utils.color_utils import rgb_to_lab_tensor, lab_tensor_to_rgb
from PIL import Image
import torchvision.transforms.functional as TF


def pad_to_multiple(img, multiple=8):
    w, h = img.size
    new_w = (w + multiple - 1) // multiple * multiple
    new_h = (h + multiple - 1) // multiple * multiple
    pad_w = new_w - w
    pad_h = new_h - h
    return TF.pad(img, (0, 0, pad_w, pad_h))


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    saved_opt_dict = checkpoint['opt']
    opt = argparse.Namespace(**saved_opt_dict)

    model = ColorCycleGANModel(opt)

    # load generators / discriminators if present
    if 'netG_A' in checkpoint:
        model.netG_A.load_state_dict(checkpoint['netG_A'])
    if 'netG_B' in checkpoint:
        model.netG_B.load_state_dict(checkpoint['netG_B'])
    if 'netD_A' in checkpoint:
        model.netD_A.load_state_dict(checkpoint['netD_A'])
    if 'netD_B' in checkpoint:
        model.netD_B.load_state_dict(checkpoint['netD_B'])

    model.to(device)
    if hasattr(model, 'netG_A'):
        model.netG_A.eval()
    if hasattr(model, 'netG_B'):
        model.netG_B.eval()

    return model


def process_file(image_path: Path, model, device: torch.device, output_dir: Path):
    img = Image.open(str(image_path)).convert('RGB')
    img = pad_to_multiple(img, 8)

    L_tensor, AB_tensor = rgb_to_lab_tensor(img)
    AB_tensor_batch = AB_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        fake_AB_batch = model.transform_to_analog(AB_tensor_batch)

    fake_AB = fake_AB_batch[0].cpu()
    fake_AB = torch.clamp(fake_AB, -1.0, 1.0)

    H, W = L_tensor.shape[1:]
    fake_AB = fake_AB[:, :H, :W]

    rgb_fake = lab_tensor_to_rgb(L_tensor, fake_AB)

    out_name = image_path.stem + '.png'
    out_path = output_dir / out_name
    # ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # `rgb_fake` is expected to be a PIL Image; save accordingly
    if isinstance(rgb_fake, Image.Image):
        rgb_fake.save(str(out_path))
    else:
        # fallback: try to convert numpy array or tensor to PIL
        try:
            Image.fromarray(rgb_fake).save(str(out_path))
        except Exception:
            # last resort: convert tensor
            try:
                arr = rgb_fake.detach().cpu().numpy()
                Image.fromarray(arr).save(str(out_path))
            except Exception:
                raise RuntimeError('Could not save output image for ' + str(image_path))

    return out_name


def main():

    working_directory = Path(os.getcwd())
    parser = argparse.ArgumentParser(description='Run AB analog transform on image(s).')
    parser.add_argument('input_path', help='Input image file or folder')
    parser.add_argument('--checkpoint', '-c', default=working_directory.joinpath('GAN_models','best_checkpoint_AB_e4.pth'), help='Path to checkpoint (default: best_checkpoint_AB_e4.pth)')
    args = parser.parse_args()

    # default checkpoint relative to script working directory
    checkpoint_path = Path(args.checkpoint)

    device = torch.device('cpu')

    # load model
    model = load_model_from_checkpoint(checkpoint_path, device)

    output_dir = working_directory.joinpath('wwwroot', 'output_GAN_AB')
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_path)

    processed = []
    if input_path.is_dir():
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF')
        files = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in exts]
        for f in files:
            name = process_file(f, model, device, output_dir)
            processed.append(name)
    else:
        name = process_file(input_path, model, device, output_dir)
        processed.append(name)

    # Print JSON result similar to lut.py
    if len(processed) == 1:
        result = {"filename": processed[0]}
    else:
        result = {"filenames": processed}

    print(json.dumps(result))


if __name__ == '__main__':
    main()
