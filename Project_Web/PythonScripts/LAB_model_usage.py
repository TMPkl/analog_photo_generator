import argparse
import json
import os
from pathlib import Path
import torch
from model.color_cyclegan_model_LAB import ColorCycleGANModel
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

    L_tensor, AB_tensor = rgb_to_lab_tensor(img)        # L: [1,H,W], AB: [2,H,W]
    lab_tensor = torch.cat([L_tensor, AB_tensor], dim=0)  # LAB: [3,H,W]

    lab_batch = lab_tensor.unsqueeze(0).to(device)  # → [1,3,H,W]

    with torch.no_grad():
        fake_lab_batch = model.inference(lab_batch)

    fake_lab = fake_lab_batch[0].cpu()
    fake_lab = torch.clamp(fake_lab, -1.0, 1.0)

    L = fake_lab[0:1]   # [1,H,W]
    AB = fake_lab[1:3]  # [2,H,W]

    rgb_fake = lab_tensor_to_rgb(L, AB)

    out_name = image_path.stem + '.png'
    out_path = output_dir / out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(rgb_fake, Image.Image):
        rgb_fake.save(str(out_path))
    else:
        try:
            Image.fromarray(rgb_fake).save(str(out_path))
        except Exception:
            try:
                arr = rgb_fake.detach().cpu().numpy()
                Image.fromarray(arr).save(str(out_path))
            except Exception:
                raise RuntimeError('Could not save output image for ' + str(image_path))

    return out_name


def main():
    working_directory = Path(os.getcwd())
    parser = argparse.ArgumentParser(description='Run LAB analog transform on image(s).')
    parser.add_argument('input_path', help='Input image file or folder')
    parser.add_argument('--checkpoint', '-c', default=working_directory.joinpath('PythonScripts', 'GAN_models','best_checkpoint_LAB_e4.pth'), help='Path to checkpoint (default: best_checkpoint_LAB_e4.pth)')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    device = torch.device('cpu')

    model = load_model_from_checkpoint(checkpoint_path, device)

    
    output_dir = working_directory.joinpath('wwwroot', 'output_gan_lab')
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_path)

    processed = []
    # Better not be part 2
    if input_path.is_dir():
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        files = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in exts]
        for f in files:
            name = process_file(f, model, device, output_dir)
            processed.append(name)
    else:
        name = process_file(input_path, model, device, output_dir)
        processed.append(name)

    if len(processed) == 1:
        result = {"filename": processed[0]}
    else:
        result = {"filenames": processed}

    print(json.dumps(result))


if __name__ == '__main__':
    main()
