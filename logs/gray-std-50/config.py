from torchvision import transforms
import os
import time

device = "cuda:1"
seed = 777

sigma = 50

max_epoch = 300
save_freq = 20
print_freq = 200
batch_size = 1

init_LR = 1.4e-4

lr_decay = 0.3
milestones = [100 * i for i in range(1, 100)]

# dataset settings
train_data_path = ["/home/yanglj/Desktop/data/datasets/CBSD432", "/home/yanglj/Desktop/data/datasets/waterloo"]
test_data_path = ["/home/yanglj/Desktop/data/datasets/CBSD68"]

train_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

log_path = os.path.join("./logs", time.strftime("%Y-%m-%d-%H-%M", time.localtime()))
checkpoint_path = f'{log_path}/checkpoint'
tensorboard_path = f'{log_path}/tensorboard'