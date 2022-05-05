from genericpath import exists
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
import typer
from guided_diffusion.script_util import sr_create_model_and_diffusion, sr_model_and_diffusion_defaults
import yaml
import torch

from tqdm import tqdm
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from guided_diffusion.image_datasets import load_data


def get_image_paths(directory: Union[Path, str], formats: List[str]=["jpg","jpeg","png"]) -> List[Path]:
    """Searches a directory for images and returns their paths.
    Note: automatically sorts the paths by name"""

    directory = Path(directory)
    image_paths = []
    for _format in formats:
        for case in (str.lower, str.upper):
            image_paths.extend(directory.glob(f"*.{case(_format)}"))

    image_paths = sorted(image_paths)
    return image_paths



class ImagePathDataset(Dataset):
    """Simple dataset for loading samples from image paths"""
    def __init__(self, im_paths, transform=None, return_path=False) -> None:
        super().__init__()
        self.image_files = im_paths
        self.transform = transform
        self.return_path = return_path

    def __getitem__(self, index):
        img = Image.open(self.image_files[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.return_path:
            im_path = str(self.image_files[index])
            return im_path, img
        else:
            return img

    def __len__(self):
        return len(self.image_files)

def load_model(checkpoint, config_path, timesteps):
    with open(config_path, "rt") as f:
        cfg = yaml.safe_load(f)

    args = sr_model_and_diffusion_defaults()
    args.update(cfg["model"])
    
    args["timestep_respacing"] = timesteps

    model, diffusion = sr_create_model_and_diffusion(**args)
    model.load_state_dict(torch.load(checkpoint))
    if args["use_fp16"]:
        model.convert_to_fp16()
    model.eval()
    return model, diffusion


def load_data(input_dir, output_size, input_size=None, batch_size=1):

    tforms = [
        transforms.Resize([output_size, output_size]),
        transforms.ToTensor(),
    ]
    if input_size:
        tforms.insert(0, transforms.Resize([input_size, input_size]))

    tforms = transforms.Compose(tforms)
    ds = ImagePathDataset(get_image_paths(input_dir), transform=tforms, return_path=True)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=1)
    return loader

def write_outputs(out, filenames, output_dir):
    output_dir = Path(output_dir)
    out = out.clamp(0,1)*255
    out = out.cpu().to(torch.uint8)
    for im, filename in zip(out, filenames):
        filename= Path(filename)
        out_name = output_dir/filename.with_suffix(".png").name
        Image.fromarray(im.permute(1,2,0).numpy()).save(out_name)



@torch.no_grad()
def run(model, diffusion, data, device, output_dir, use_ddim):
    model.to(device)
    if use_ddim:
        sample_fn = diffusion.ddim_sample_loop
    else:
        sample_fn = diffusion.p_sample_loop

    for batch in tqdm(data):
        filenames, low_res = batch
        model_kwargs = {"low_res": low_res.to(device)}
        out = sample_fn(model, low_res.shape, model_kwargs=model_kwargs, device=device, progress=True)

        write_outputs(out, filenames, output_dir)

def main(
    checkpoint:Path,
    config:Path,
    input_dir:Path,
    output_dir:Path,
    output_size:int=512,
    input_size:Optional[int]=None,
    timesteps:str="ddim50",
    device:str="cuda:0",
    batch_size:int=1,
):
    if not output_dir.exists():
        output_dir.mkdir()
    
    use_ddim = timesteps.startswith("ddim")
    model, diffusion = load_model(checkpoint, config, timesteps)
    data = load_data(input_dir, output_size, input_size, batch_size)
    run(model, diffusion, data, device, output_dir, use_ddim)

if __name__ == "__main__":
    typer.run(main)