import os
from importlib import import_module

import torch
from ignite.engine import Events, Engine
from ignite.utils import convert_tensor
from ignite.contrib.handlers import ProgressBar
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Subset

import dataset

# each line represent a generator: key is name, value is the import path for this generator.
IMPLEMENTED_GENERATOR = {
    "PG2-Generator": "implementations.PG2.generate",
}

def get_data_loader(config):
    cfg = config["dataset"]["path"]["test"]
    image_dataset = dataset.PairBoneDataset(cfg["pair"], cfg["image"], cfg["bone"], cfg["mask"], cfg["annotation"])
    if "generated_limit" in config:
        image_dataset = Subset(image_dataset, range(config["generated_limit"]))
    image_loader = DataLoader(image_dataset, batch_size=config["train"]["batch_size"],
                              num_workers=8, pin_memory=True, drop_last=True)
    print(image_dataset)
    return image_loader

def make_engine(generator_name, config, device=torch.device("cuda"), mobilenet=False, verbose=False):
    try:
        make_generator = import_module(IMPLEMENTED_GENERATOR[generator_name]).make_generator
    except KeyError:
        raise RuntimeError("not implemented generator <{}>".format(generator_name))
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
    engine = make_engine(config["engine"], config, device, options.mobilenet, options.verbose)
    engine.run(train_data_loader, max_epochs=1)
