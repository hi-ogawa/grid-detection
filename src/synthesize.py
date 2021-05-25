import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random


def loop_sample(data):
    while True:
        yield data[random.randrange(0, len(data))]


def to_tensor(item):
    image, label = item
    return VF.to_tensor(image)


def synthesize(generator, nrows=8, ncols=8, p=0.3):
    x = []
    x0 = next(generator)
    c, h, w = x0.shape
    for _ in range(nrows * ncols):
        if random.uniform(0, 1) < p:
            x.append(x0)
        else:
            x.append(next(generator))
    x = torch.stack(x) # (nrows * ncols, c, h, w)
    x = x.reshape((nrows, ncols, c, h, w)).permute((2, 0, 3, 1, 4)).reshape((c, nrows * h, ncols * w))
    return x


def random_perspective_with_background(t, t_back, size):
    t = VF.resize(t, size)
    t_back = VF.resize(t_back, size)
    startpoints, endpoints = VT.RandomPerspective.get_params(*size, 0.5)
    dummy = torch.rand(3)
    s = VF.perspective(t, startpoints, endpoints, fill=list(dummy))
    mask = (s[0, :, :] == dummy[0]).broadcast_to(t.size())
    s[mask] = t_back[mask]
    return s, endpoints


def main(count, out_dir, width=300, height=300, grid_min=8, grid_max=8):
    print(":: Loading dataset...")
    dataset_cifar100 = torchvision.datasets.CIFAR100('data/cifar100')
    dataset_stl10 = torchvision.datasets.STL10('data/stl10')
    print(":: Starting synthesize...")
    gen1 = map(to_tensor, loop_sample(dataset_cifar100))
    gen2 = map(to_tensor, loop_sample(dataset_stl10))
    labels = []
    for i in tqdm(range(count)):
        nrows = random.randrange(grid_min, grid_max + 1)
        ncols = random.randrange(grid_min, grid_max + 1)
        x = synthesize(gen1, nrows=nrows, ncols=ncols)
        y = next(gen2)
        z, label = random_perspective_with_background(x, y, [width, height])
        image = VF.to_pil_image(z)
        image.save(f"{out_dir}/{i:05d}.png")
        labels.append(label)
    torch.save(labels, f"{out_dir}/labels.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--grid-min", type=int, default=8)
    parser.add_argument("--grid-max", type=int, default=8)
    main(**parser.parse_args().__dict__)
