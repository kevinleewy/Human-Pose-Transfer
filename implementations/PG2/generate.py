import torch

from .model import Generator2, Generator1
from .mobile.model import Generator2 as MobileGenerator2, Generator1 as MobileGenerator1
from helper.misc import count_parameters

def make_generator(config, device=torch.device("cuda"), mobilenet=False):
    if mobilenet:
        cfg = config["model"]["generator1"]
        generator1 = MobileGenerator1(3 + 18, cfg["num_repeat"], cfg["middle_features_dim"], cfg["channels_base"],
                                cfg["image_size"])
        generator1.to(device)
        generator1.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

        cfg = config["model"]["generator2"]
        generator2 = Generator2(3 + 3, cfg["channels_base"], cfg["num_repeat"], cfg["num_skip_out_connect"])
        #generator2 = MobileGenerator2(3 + 3, cfg["channels_base"], cfg["num_repeat"], cfg["num_skip_out_connect"])
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

    print("Generator 1 parameter count: {}".format(count_parameters(generator1)))
    print("Generator 2 parameter count: {}".format(count_parameters(generator2)))

    def generate(batch, gfv=False):
        with torch.no_grad():
            generator1.eval()
            generator2.eval()
            g1_out = generator1(batch["condition_img"], batch["target_bone"])

            if gfv:
                return generator2.getGFV(batch["condition_img"], g1_out)

            generated_img = generator2(batch["condition_img"], g1_out) + g1_out
            return generated_img, g1_out  

    return generate
