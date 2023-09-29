import os
import torch
import time
from utils import *
from config import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from core.ECSCNet import *

set_seed(seed)

net = ECSCNet(1, 175, 11, 8, 12).to(device)

os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(tensorboard_path, exist_ok=True)
os.system(f"cp -r `ls |grep -v logs`  {log_path}")


def train(epoch):
    start = time.time()
    net.train()
    loop = enumerate(train_loader)
    avg_loss = 0
    avg_psnr = 0
    for batch_index, (ID, noisy, GT) in loop:
        ID = ID.tolist()
        batch_num = batch_index + 1
        GT = GT.to(device)
        noisy = noisy.to(device)
        LR = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        outputs = net(noisy)

        loss = loss_function(outputs, GT)
        loss.backward()
        optimizer.step()
        n_iter = (epoch - 1) * iter_per_epoch + batch_num

        cur_psnr = PSNR(outputs, GT).mean().item()
        avg_psnr += cur_psnr
        avg_loss += loss.item()

        if batch_index % print_freq == 0:
            print(
                f"Iter {n_iter/1000:.3f}K-> avg_loss: {avg_loss/batch_num:.4f}, avg_psnr: {avg_psnr/batch_num:.4f}, cur_loss: {loss.item():.4f}, cur_psnr: {cur_psnr:.4f}, LR= {LR:.5e}, index: {ID},"
            )

        writer.add_scalar('Train/loss', loss.item(), n_iter)

    finish = time.time()
    used_time = (finish - start) / 60

    print('training epoch {:-4d} consumed time : {:.2f}min'.format(epoch, used_time))


@torch.no_grad()
def evaluate(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    test_psnr = 0.0

    for _, (ID, noisy, GT) in enumerate(test_loader):
        ID = ID.tolist()
        noisy, GT = noisy.to(device), GT.to(device)
        outputs = net(noisy).clamp(min=0.0, max=1.0)
        loss = loss_function(outputs, GT)
        test_loss += loss.sum().item()
        cur_psnr = PSNR(outputs, GT)
        test_psnr = test_psnr + cur_psnr.sum().item()

    finish = time.time()
    avg_loss = test_loss / len(test_data)
    avg_psnr = test_psnr / len(test_data)

    used_time = (finish - start) / 60
    print('test-> epoch: {:-4d}, avg_loss: {:.4f}, avg_psnr: {:.4f}, Time consumed:{:.2f}min'.format(epoch, avg_loss, avg_psnr, used_time))
    print('-----------------------------------------------------------')
    print('')
    # add informations to tensorboard
    if tb:
        writer.add_scalar('test/avg_loss', test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar('test/avg_psnr', avg_psnr, epoch)
    return avg_psnr


train_data = DenoisingDataset(train_data_path, transform=train_transforms, std=sigma)
test_data = DenoisingDataset(test_data_path, transform=test_transforms, std=sigma)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=5)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

iter_per_epoch = len(train_loader)

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=init_LR, betas=(0.9, 0.99), eps=1e-7)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay)

writer = SummaryWriter(log_dir=tensorboard_path)

best_psnr = 0.0
for epoch in range(1, max_epoch + 1):
    train(epoch)
    avg_test_psnr = evaluate(epoch)
    scheduler.step()
    state_dict = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if best_psnr < avg_test_psnr:
        weights_path = os.path.join(checkpoint_path, "best.pth")
        torch.save(state_dict, weights_path)
        best_psnr = avg_test_psnr

    if epoch % save_freq == 0:
        weights_path = os.path.join(checkpoint_path, f"recent.pth")
        torch.save(state_dict, weights_path)

writer.close()
