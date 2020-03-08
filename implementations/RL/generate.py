import os
from importlib import import_module

import torch
from torch.autograd.variable import Variable
from ignite.engine import Events, Engine
from ignite.utils import convert_tensor
from ignite.contrib.handlers import ProgressBar
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset

import dataset

# Local Imports
from .models.losses import NLL, MSE, Norm
from .models import DDPG, OurDDPG, TD3
from .train import envs
from ..LGAN.sagan_models import Generator, Discriminator
from ..PG2.loss import MaskL1Loss
from ..PG2.model import Generator2, Generator1
from ..PG2.mobile.model import Generator2  as MobileGenerator2, Generator1 as MobileGenerator1

def get_data_loader(config):
    cfg = config["dataset"]["path"]["test"]
    image_dataset = dataset.PairBoneDataset(cfg["pair"], cfg["image"], cfg["bone"], cfg["mask"], cfg["annotation"])
    if "generated_limit" in config:
        image_dataset = Subset(image_dataset, range(config["generated_limit"]))
    image_loader = DataLoader(image_dataset, batch_size=config["train"]["batch_size"],
                              num_workers=8, pin_memory=True, drop_last=True)
    print(image_dataset)
    return image_loader

def make_generator(config, device=torch.device("cuda"), mobilenet=False):
    if mobilenet:
        cfg = config["model"]["generator1"]
        generator1 = MobileGenerator1(3 + 18, cfg["num_repeat"], cfg["middle_features_dim"], cfg["channels_base"],
                                cfg["image_size"])
        generator1.to(device)
        generator1.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

        cfg = config["model"]["generator2"]
        generator2 = MobileGenerator2(3 + 3, cfg["channels_base"], cfg["num_repeat"], cfg["num_skip_out_connect"])
        generator2.to(device)
        generator2.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))
    else:    
        cfg = config["model"]["generator1"]
        generator1 = Generator1(3 + 18, cfg["num_repeat"], cfg["middle_features_dim"], cfg["channels_base"],
                                cfg["image_size"])
        generator1.to(device)
        generator1.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

        cfg = config["model"]["generator2"]
        generator2 = Generator2(3 + 3, cfg["channels_base"], cfg["num_repeat"], cfg["num_skip_out_connect"])
        generator2.to(device)
        generator2.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

    # Load LGAN
    cfg = config["model"]["lgan"]
    lgan_G = Generator(cfg["imsize"], cfg["z_dim"], cfg["g_conv_dim"]).to(device)
    lgan_D = Discriminator(cfg["imsize"], cfg["d_conv_dim"]).to(device)
    lgan_G.load_state_dict(torch.load(cfg["pretrained_G"], map_location="cpu"))
    lgan_D.load_state_dict(torch.load(cfg["pretrained_D"], map_location="cpu"))
    action_dim = cfg["z_dim"]

    # Load RL Agent
    cfg = config["model"]["rl"]
    if cfg["policy_name"] == "TD3":
        policy = TD3.TD3(cfg["state_dim"], action_dim, cfg["max_action"])
    elif cfg["policy_name"] == "OurDDPG":
        policy = OurDDPG.DDPG(cfg["state_dim"], action_dim, cfg["max_action"], device)
    elif cfg["policy_name"] == "DDPG":
        policy = DDPG.DDPG(cfg["state_dim"], action_dim, cfg["max_action"], device)
    
    file_name = "%s_%s" % (cfg["policy_name"], cfg["env_name"])
    policy.load(file_name, directory=cfg["pretrained_path"])

    env = envs(config, lgan_G, lgan_D, generator1, generator2, 0, device)

    def generate(batch):
        with torch.no_grad():
            generator1.eval()
            generator2.eval()
            lgan_G.eval()
            lgan_D.eval()
            policy.actor.eval()
            policy.critic.eval()

            obs = env.agent_input(batch)
            obs = obs.cpu().numpy()

            action = policy.select_action(obs)
            action = torch.tensor(action).to(device).unsqueeze(dim=0)
            generated_img, g1_out = env.generate(batch, action)
            return generated_img, g1_out  

    return generate

def make_engine(config, device=torch.device("cuda"), mobilenet=False, verbose=False):

    generate = make_generator(config, device, mobilenet)

    def _step(engine, batch):
        batch = convert_tensor(batch, device)
        generated_images, coarse_generated_images = generate(batch)

        if verbose:
            bone = batch["target_bone"].round().sum(dim=1, keepdim=True).repeat_interleave(repeats=3, dim=1)
            bone = torch.where(bone >= 1, torch.ones_like(bone), torch.zeros_like(bone))
            return (batch["condition_path"], batch["target_path"]), \
               (batch["condition_img"], bone, batch["target_mask"], coarse_generated_images, generated_images, batch["target_img"])

        return (batch["condition_path"], batch["target_path"]), \
               (batch["condition_img"], batch["target_img"], generated_images)

    engine = Engine(_step)
    ProgressBar(ncols=0).attach(engine)

    @engine.on(Events.ITERATION_COMPLETED)
    def save(e):
        names, images = e.state.output
        for i in range(images[0].size(0)):
            image_name = os.path.join(config["output"], "{}___{}_vis.jpg".format(names[0][i], names[1][i]))
            save_image([imgs.data[i] for imgs in images], image_name,
                       nrow=len(images), normalize=True, padding=0)
    return engine

def run(config, options, device=torch.device("cuda")):
    train_data_loader = get_data_loader(config)
    engine = make_engine(config, device, options.mobilenet, options.verbose)
    engine.run(train_data_loader, max_epochs=1)
