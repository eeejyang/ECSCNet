import os
import torch
import time
from utils import *
from config import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from core.ECSCNet import *

device = "cuda:0"

net = ECSCNet(1, 175, 11, 8, 12).to(device)
net.load_state_dict(torch.load("logs/gray-std-50/checkpoint/best.pth")['net'])

test_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

test_data = DenoisingDataset(test_data_path, transform=test_transforms, std=sigma)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

start = time.time()
net.eval()

test_loss = 0.0  # cost function error
test_psnr = 0.0
loss_function = nn.MSELoss()

unfold = nn.Unfold(128, 1, 0, 64)
with torch.no_grad():
    for _, (ID, noisy, GT) in enumerate(test_loader):
        ID = ID.tolist()
        noisy, GT = noisy.to(device), GT.to(device)
        outputs = net(noisy).clamp(min=0.0, max=1.0)
        loss = loss_function(outputs, GT)
        cur_psnr = PSNR(outputs, GT)

        test_loss += loss.sum().item()
        test_psnr += cur_psnr.sum().item()
        print(f"{ID}: {cur_psnr}")

finish = time.time()
avg_loss = test_loss / len(test_data)
avg_psnr = test_psnr / len(test_data)

used_time = (finish - start) / 60
print('test->avg_loss: {:.4f}, avg_psnr: {:.4f}, Time consumed:{:.2f}min'.format(avg_loss, avg_psnr, used_time))
print('-----------------------------------------------------------')
print('')
