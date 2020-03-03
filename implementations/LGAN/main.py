from torch.backends import cudnn
import torch
import torchvision.transforms as transforms

from .Datasets.GFVDataset import GFVDataset
from .data_loader import Data_Loader
# from .parameter import *
from .trainer import Trainer
from .utils import make_folder

def get_data_loader(config):
    dataset = GFVDataset(config)
    print("Loaded {} GFVs".format(len(dataset)))
    return torch.utils.data.DataLoader(dataset,
                                        batch_size=config["train"]["batch_size"],
                                        num_workers=8,
                                        shuffle=True,
                                        pin_memory=True)

def get_trainer(config, options, device):
    # For fast training

    cudnn.benchmark = True


    data_loader = get_data_loader(config)

    # Create directories if not exist
    make_folder(config["output"], 'models', 'sagan-1')
    make_folder(config["output"], 'samples', 'sagan-1')
    make_folder(config["output"], 'logs', 'sagan-1')
    make_folder(config["output"], 'attn', 'sagan-1')

    return Trainer(data_loader, config, device)

    # if args.train:
    # if True:
    #     if lgan_config["model"] == 'sagan':
    #         trainer = Trainer(None, lgan_config, train_loader,model_decoder,mask_l1_loss,vis_Valida) #data_loader.loader()
    #     elif lgan_config["model"] == 'qgan':
    #         trainer = qgan_trainer(None, lgan_config) # data_loader.loader()
    #     trainer.train()
    # else:
    #     tester = Tester(data_loader.loader(), args, valid_loader)
    #     tester.test()

def run(config, options, device=torch.device("cuda")):
    trainer = get_trainer(config, options, device)
    trainer.train()



