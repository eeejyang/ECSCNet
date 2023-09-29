import os
import sys
import io
import pickle
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def PSNR(x: torch.Tensor, y: torch.Tensor):
    mse = torch.pow((x - y), 2)
    mse = mse.sum((-1, -2, -3)) / x.shape[-1] / x.shape[-2] / x.shape[-3]
    ret = -10 * torch.log10(mse)
    return ret


class DenoisingDataset(Dataset):

    def __init__(self, root_dirs, transform=None, std=50):
        super(DenoisingDataset, self).__init__()
        if type(root_dirs) is str:
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.transforms = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.image_paths = []
        self.std = std / 255.0
        for cur_path in root_dirs:
            self.image_paths += [
                os.path.join(cur_path, file).replace('\\', '/') for file in os.listdir(cur_path) if file.endswith(('png', 'jpg', 'jpeg', 'bmp'))
            ]
        self.image_paths.sort()
        self.imgs = []

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with open(path, 'rb') as f:
            groundtruth = Image.open(io.BytesIO(f.read()))
        groundtruth = self.transforms(groundtruth)
        noisy_image = torch.randn(groundtruth.shape) * self.std + groundtruth

        return idx, noisy_image, groundtruth

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    dataset = DenoisingDataset(["/home/yanglj/Desktop/data/datasets/CBSD432"])
    # dataset = DenoisingDataset(["/liws_fix/yanglj/datasets/CBSD432"])
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, shuffle=False, num_workers=4, batch_size=1)
    mean = 0
    num = 0
    import time
    print(len(dataset))
    start = time.time()
    for idx, (id, _, img) in enumerate(train_loader):
        pass
    # mean = mean/num
    end = time.time()
    print(end - start)
