import pandas as pd 
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchvision.transforms.v2 as v2

from .hc_dataset import HCDataset

def setup(rank, world_size): 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1739"
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup(): 
    dist.destroy_process_group()

def train(rank, world_size): 
    setup(rank, world_size)

    batch_size = 2
    EPOCHS = 50
    
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 
        'unet',
        in_channels=3, 
        out_channels=1, 
        init_features=32, 
        pretrained=False
    ).to(rank)
    model = DDP(model, device_ids=[rank])

    train_set = HCDataset(path = "data/hc18/training_set") 
    valid_set = HCDataset(path = "data/hc18/valid_set")
    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_set))
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(valid_set))

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    min_loss = 9999
    
    train_loss = []
    val_loss = []

    for epoch in range(EPOCHS): 
        print("-" * 50)
        print(f"Epoch [{epoch+1}/{EPOCHS}]: ")

        train_dataloader.sampler.set_epoch(epoch)
        valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        loss = train_one_epoch(rank, model, train_dataloader, loss_fn, optimizer)
        avg_loss = average_loss(loss, world_size)

        model.eval()
        vloss = eval_one_epoch(rank, model, valid_dataloader, loss_fn)
        avg_vloss = average_loss(vloss, world_size)
        
        if (rank != 0): 
            continue

        train_loss.append(avg_loss)
        val_loss.append(avg_vloss)

        print(f"\tTrain loss: {avg_loss}")
        print(f"\tValidation loss: {avg_vloss}")

        if (avg_vloss < min_loss): 
            min_loss = avg_vloss
            torch.save(model.module.state_dict(), "model/hc_unet.pth")
            print("Save model to model/hc_unet.pth in process ", rank)

    if (rank != 0): 
        return

    train_res = pd.DataFrame({
        "train_loss": train_loss, 
        "val_loss": val_loss
    })

    train_res.to_csv("train_res.csv", index=False)


def train_one_epoch(rank, model, train_dataloader, loss_fn, optimizer): 
    running_loss = 0

    for i, data in enumerate(train_dataloader): 
        img, gt = data 
        mask = gt['mask']

        img = img.to(rank)
        mask = mask.to(rank)
        
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, mask)
        loss.backward()
        optimizer.step() 
        
        running_loss += loss.item()
        if (i % 50 == 49): 
            print(f"[{i+1}/{len(train_dataloader)}]")

    return running_loss/len(train_dataloader)

def eval_one_epoch(rank, model, valid_dataloader, loss_fn): 
    running_vloss = 0
    
    with torch.no_grad(): 
        for i, data in enumerate(valid_dataloader): 
            img, gt = data
            mask = gt['mask']

            img = img.to(rank)           
            mask = mask.to(rank)

            output = model(img)
            loss = loss_fn(output, mask) 
            
            running_vloss += loss.item()

            if (i % 100 == 99): 
                print(f"[{i+1}/{len(valid_dataloader)}]")

    return running_vloss/len(valid_dataloader)

def average_loss(loss, world_size): 
    '''
    Average loss in all processes

    Parameter: 
    --- 
    world_size (int): Number of processes
    '''
    loss_tensor = torch.tensor(loss, device = "cuda")
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    return loss_tensor.item()/world_size

def main(): 
    world_size = torch.cuda.device_count()
    mp.spawn(
        train, 
        args=(world_size,), 
        nprocs=world_size, 
        join=True
    )

if __name__ == "__main__": 
    main()
