from config import SAVE_STATE_ROOT_FOLDER

import torch
from tqdm import tqdm
import torch.nn.functional as F
import math
from torch.optim import Adam
from pathlib import Path


def train(model,
          train_dataloader,
          val_dataloader,
          test_dataloader,
          epochs,
          lr=0.001,
          lr_decay=0.1,
          lr_decay_step=50,
          ARR=0.001,
          save_interval=5):

    if torch.cuda.is_available():
        model.cuda()

    optimizer = Adam(model.parameters(), lr=lr)
    rmse_list = []
    loss_list = []

    for epoch in tqdm(range(1, epochs+1)):
        if epoch % lr_decay_step == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay * param_group['lr']

        loss = train_one_epoch(model, train_dataloader, optimizer, ARR)
        loss_list.append(loss)
        rmse = calculate_rmse(model, val_dataloader)
        rmse_list.append(rmse)
        print("Epoch {}: loss = {:.6f}, val_rmse: {:.6f}".format(epoch, loss, rmse))

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), Path(SAVE_STATE_ROOT_FOLDER))

    print("Final validation RMSE: {:.6f}".format(rmse_list[-1]))
    test(model, test_dataloader)

    return rmse_list, loss_list

def train_one_epoch(model, train_dataloader, optimizer, ARR):
    model.train()
    cum_loss = 0
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            batch = batch.to('cuda')
        out = model(batch)
        loss = F.mse_loss(out, batch.y.view(-1))

        # Calculate ARR and add to loss.
        # This code is directly from the official IGMC source code
        for gconv in model.convs:
            w = torch.matmul(
                gconv.att,
                gconv.basis.view(gconv.num_bases, -1)
            ).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
            reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :]) ** 2)
            loss += ARR * reg_loss

        loss.backward()
        cum_loss += loss.item() * batch.num_graphs
        optimizer.step()
        torch.cuda.empty_cache()
    return cum_loss / len(train_dataloader)


def test(model, dataloader):
    if torch.cuda.is_available():
        model.cuda()
    rmse = calculate_rmse(model, dataloader)
    print("Test RMSE: ", rmse)


def calculate_loss(model, dataloader):
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    loss = 0

    for data in tqdm(dataloader):
        if torch.cuda.is_available():
            data = data.to('cuda')
        with torch.no_grad():
            out = model(data)
        loss += F.mse_loss(out, data.y.view(-1))

    return loss/len(dataloader.dataset)


def calculate_rmse(model, dataloader):
    loss = calculate_loss(model, dataloader)
    return math.sqrt(loss)
