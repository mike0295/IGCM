from dataloaders import *
from models import *
from train_eval import *

import argparse
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description='IGMC')
parser.add_argument('--dataset_code', default='ml-100k')
parser.add_argument('--dataloader_code', default='igmc')
parser.add_argument('--train_batch_size', default=50)
parser.add_argument('--val_batch_size', default=50)
parser.add_argument('--test_batch_size', default=50)
parser.add_argument('--max_nodes', default=200)

args = parser.parse_args()
print("Model: IGMC, Dataset = MovieLens100K")
model = IGMCModel(num_features=4)

print("Loading dataloaders...")
train_loader, val_loader, test_loader = dataloader_factory(args)

# for data in tqdm(train_loader):
#     print(data)

print("Dataloaders loaded\nTraining begins...")
rmse_list, loss_list = train(model, train_loader, val_loader, test_loader, epochs=80)

with open("rmse_record.json", "w") as f:
    json.dump(rmse_list, f)

with open("loss_record.json", "w") as f:
    json.dump(loss_list, f)
