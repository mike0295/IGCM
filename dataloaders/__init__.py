from datasets import dataset_factory
from .igmc import IGMCDataloader

DATALOADERS = {
    IGMCDataloader.code(): IGMCDataloader,
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    # print("dataloader created")
    return train, val, test
