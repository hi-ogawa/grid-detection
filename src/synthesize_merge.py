import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random
import os
import pathlib
import shutil


def main(in_dirs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    images = []
    labels = []
    count = 0
    with tqdm() as pbar:
        for in_dir in in_dirs:
            labels.extend(torch.load(f"{in_dir}/labels.pt"))
            for file in sorted(list(pathlib.Path(in_dir).glob("*.png"))):
                shutil.copyfile(file, f"{out_dir}/{count:05d}.png")
                count += 1
                pbar.update(1)
    torch.save(labels, f"{out_dir}/labels.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dirs", type=str, required=True, nargs='+')
    parser.add_argument("--out-dir", type=str, required=True)
    main(**parser.parse_args().__dict__)
