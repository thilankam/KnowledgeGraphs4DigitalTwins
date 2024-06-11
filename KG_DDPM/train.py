from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy import io

from network import DDPM, ContextUnet


def train_mnist():

    # hardcoding these here
    n_epoch = 5001
    batch_size = 5
    n_T = 400
    device = "cuda:0"
    n_classes = 2
    n_feat = 128
    lrate = 1e-4
    save_model = True
    save_dir = './data/diffusion_output/'
    ws_test = [0.0, 0.5, 2.0]

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    tf = transforms.Compose([transforms.ToTensor()])

    new_dataset_ = torch.from_numpy((np.load('norm_GPM_3IMERGDL.npz')['arr_0'])[:,:,:28,:28])
    print(new_dataset_.size())
    new_dataset = new_dataset_.to(torch.float32)
    new_data = torch.utils.data.TensorDataset(new_dataset)
    dataloader = DataLoader(new_data, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    LOSS = []

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x=x[0].to(device)
            c = (torch.zeros(batch_size)).long()
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            LOSS.append(loss_ema)
            optim.step()
        
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"old_model_{ep}.pth")
            print('saved model at ' + save_dir + f"old_model_{ep}.pth")
            io.savemat('loss.mat',{'loss':LOSS})


if __name__ == "__main__":
    train_mnist()


