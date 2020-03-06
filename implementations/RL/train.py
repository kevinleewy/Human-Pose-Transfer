import numpy as np
import os
import sys
import time
from torch.autograd.variable import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

# Local Imports
from .models.losses import NLL, MSE, Norm
from .models import DDPG, OurDDPG, TD3
from .utils import AverageMeter, ReplayBuffer
from ..LGAN.sagan_models import Generator, Discriminator
from ..PG2.data import get_data_loader, get_val_data_pairs
from ..PG2.loss import MaskL1Loss
from ..PG2.model import Generator2, Generator1
from ..PG2.mobile.model import Generator2  as MobileGenerator2, Generator1 as MobileGenerator1

# Set RNG seed
np.random.seed(5)
#torch.manual_seed(5)

def evaluate_policy(policy, data_loader, env, config, device='cpu', eval_episodes=6, render=False):
    avg_reward = 0.
    env.reset(epoch_size=eval_episodes, figures=8) # reset the visdom and set number of figures

    with tqdm(range(eval_episodes)) as pbar:
        for i in pbar:
            input = {}
            for key in data_loader.keys():
                if key.endswith('path'):
                    input[key] = [data_loader[key][i]]
                else:
                    input[key] = data_loader[key][i].unsqueeze(0)

            obs = env.agent_input(input)
            obs = obs.cpu().numpy()
            done = False

            while not done:
                # Action By Agent and collect reward
                action = policy.select_action(obs)
                action = torch.tensor(action).to(device).unsqueeze(dim=0)
                _, _, reward, done, _ = env(input, action, render=render, printf=pbar.set_description)
                avg_reward += reward

            if i+1 >= eval_episodes:
                break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    return avg_reward

class Trainer(object):
    def __init__(self, config, options, device):
        self.config = config
        self.options = options
        self.device = device

        self.model_save_path = config["output"]
        
        self.train_data_loader = get_data_loader(config)
        self.val_data_pair = get_val_data_pairs(config)

        if options.mobilenet:
            cfg = config["model"]["generator1"]
            self.generator1 = MobileGenerator1(3 + 18, cfg["num_repeat"], cfg["middle_features_dim"],
                                    cfg["channels_base"], cfg["image_size"])
            self.generator1.to(device)
            self.generator1.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

            cfg = config["model"]["generator2"]
            self.generator2 = MobileGenerator2(3 + 3, cfg["channels_base"], cfg["num_repeat"], cfg["num_skip_out_connect"], weight_init_way=cfg["weight_init_way"])
            self.generator2.to(device)
            self.generator2.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

        else:
            cfg = config["model"]["generator1"]
            self.generator1 = Generator1(3 + 18, cfg["num_repeat"], cfg["middle_features_dim"],
                                    cfg["channels_base"], cfg["image_size"])
            self.generator1.to(device)
            self.generator1.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

            cfg = config["model"]["generator2"]
            self.generator2 = Generator2(3 + 3, cfg["channels_base"], cfg["num_repeat"], cfg["num_skip_out_connect"], weight_init_way=cfg["weight_init_way"])
            self.generator2.to(device)
            self.generator2.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

        cfg = config["model"]["lgan"]
        self.lgan_G = Generator(cfg["imsize"], cfg["z_dim"], cfg["g_conv_dim"]).to(self.device)
        self.lgan_D = Discriminator(cfg["imsize"], cfg["d_conv_dim"]).to(self.device)
        self.lgan_G.load_state_dict(torch.load(cfg["pretrained_G"], map_location="cpu"))
        self.lgan_D.load_state_dict(torch.load(cfg["pretrained_D"], map_location="cpu"))
        self.action_dim = cfg["z_dim"]

        cfg = config["model"]["rl"]
        self.env_name = cfg["env_name"]
        self.policy_name = cfg["policy_name"]
        self.state_dim = cfg["state_dim"] # 196,608 = 384 x 32 x 16
        self.max_action = cfg["max_action"]
        self.rl_pretrained_path = cfg["pretrained_path"]

        cfg = config["train"]
        self.batch_size = cfg["batch_size"]
        self.start_timesteps = cfg["start_timesteps"]
        self.eval_freq = cfg["eval_freq"]
        self.max_timesteps = cfg["max_timesteps"]
        self.save_models = cfg["save_models"]
        self.batch_size_actor = cfg["batch_size_actor"]
        self.discount = cfg["discount"]
        self.tau = cfg["tau"]
        self.expl_noise = cfg["expl_noise"]
        self.policy_noise = cfg["policy_noise"]
        self.noise_clip = cfg["noise_clip"]
        self.policy_freq = cfg["policy_freq"]
        self.max_episodes_steps = cfg["max_episodes_steps"]

        self.val_batch_size = config["log"]["verify"]["batch_size"]


    def train(self):

        self.generator1.eval()
        self.generator2.eval()
        self.lgan_G.eval()
        self.lgan_D.eval()

        epoch_size = self.val_batch_size

        file_name = "%s_%s" % (self.policy_name, self.env_name)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        
        env = envs(self.config, self.lgan_G, self.lgan_D, self.generator1, self.generator2, epoch_size, self.device)

        # Initialize policy
        if self.policy_name == "TD3":
            policy = TD3.TD3(self.state_dim, self.action_dim, self.max_action)
        elif self.policy_name == "OurDDPG":
            policy = OurDDPG.DDPG(self.state_dim, self.action_dim, self.max_action, self.device)
        elif self.policy_name == "DDPG":
            policy = DDPG.DDPG(self.state_dim, self.action_dim, self.max_action, self.device)

        if self.rl_pretrained_path:
            policy.load(file_name, directory=self.rl_pretrained_path)

        replay_buffer = ReplayBuffer()

        evaluations = [evaluate_policy(policy, self.val_data_pair, env, self.config, self.device, eval_episodes=self.val_batch_size)]

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0

        # Reset environment
        done = False
        episode_reward = 0
        episode_timesteps = 0

        env.reset(epoch_size=len(self.train_data_loader))

        while total_timesteps < self.max_timesteps:

            with tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader), position=0, leave=True) as pbar: # progress bar
                for i, input in pbar:

                    if done:

                        if total_timesteps != 0:
                            # print("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                            if self.policy_name == "TD3":
                                policy.train(replay_buffer, episode_timesteps, self.batch_size, self.discount, self.tau, self.policy_noise, self.noise_clip, self.policy_freq)
                            else:
                                policy.train(replay_buffer, episode_timesteps, self.batch_size, self.discount, self.tau)

                        # Evaluate episode
                        if timesteps_since_eval >= self.eval_freq:
                            timesteps_since_eval %= self.eval_freq

                            evaluations.append(evaluate_policy(policy, self.val_data_pair, env, self.config, self.device, eval_episodes=self.val_batch_size, render=False))

                            if self.save_models:
                                policy.save(file_name, directory=self.model_save_path)

                            env.reset(epoch_size=len(self.train_data_loader))


                        # Reset environment
                        done = False
                        episode_reward = 0
                        episode_timesteps = 0
                        episode_num += 1

                    # Select action randomly or according to policy
                    obs = env.agent_input(input)
                    obs = obs.cpu().numpy()

                    if total_timesteps < self.start_timesteps:
                        action_t = torch.FloatTensor(self.batch_size, self.action_dim).uniform_(-self.max_action, self.max_action)
                        action = action_t.detach().cpu().numpy().squeeze(0)



                    # obs, _, _, _, _ = env(input, action_t)
                    else:

                    # action_rand = torch.randn(args.batch_size, args.z_dim)
                    #
                    # obs, _, _, _, _ = env( input, action_rand)

                        action = policy.select_action(obs)
                        if self.expl_noise != 0:
                            action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip(
                                -self.max_action * np.ones(self.action_dim,), self.max_action * np.ones(self.action_dim,)
                            )
                            action = np.float32(action)
                        action_t = torch.tensor(action).to(self.device).unsqueeze(dim=0)
                    # Perform action

                    # env.render()

                    new_obs, _, reward, done, _ = env(input, action_t, printf=pbar.set_description)

                    # new_obs, reward, done, _ = env.step(action)
                    done_bool = 0 if episode_timesteps + 1 == self.max_episodes_steps else float(done)
                    episode_reward += reward

                    # Store data in replay buffer
                    replay_buffer.add((obs.reshape(-1), new_obs.reshape(-1), action, reward, done_bool))

                    obs = new_obs

                    episode_timesteps += 1
                    total_timesteps += 1
                    timesteps_since_eval += 1

class envs(nn.Module):
    def __init__(self, config, lgan_G, lgan_D, generator1, generator2, epoch_size, device='cpu'):
        super(envs,self).__init__()

        # Logging
        self.log_freq = config["log"]["check_freq"]
        self.log_path = os.path.join(config["output"], 'train.log')

        self.device = device
        self.mask_l1_loss = MaskL1Loss(config["loss"]["mask_l1"]["mask_ratio"])
        self.mask_l1_loss.to(device)
        self.nll = NLL()
        self.mse = MSE(reduction='mean')
        self.norm = Norm(dims=config["model"]["lgan"]["z_dim"])

        self.epoch = 0
        self.epoch_size = epoch_size

        self.lgan_G = lgan_G
        self.lgan_D = lgan_D
        self.generator1 = generator1
        self.generator2 = generator2
        self.j = 1
        self.figures = 3
        self.attempts = config["model"]["rl"]["attempts"]
        self.state_dim = config["model"]["rl"]["state_dim"]
        self.k1 = config["model"]["rl"]["k1"]
        self.k2 = config["model"]["rl"]["k2"]
        self.k3 = config["model"]["rl"]["k3"]
        self.k4 = config["model"]["rl"]["k4"]
        self.end = time.time()
        self.batch_time = AverageMeter()
        self.losses = AverageMeter()
        self.attempt_id =0
        self.state_prev = np.zeros([4,])
        self.iter = 0

    def reset(self, epoch_size, figures=3):
        self.j = 1
        self.i = 0
        self.epoch_size = epoch_size
        self.figures = figures
        
    def agent_input(self,input):
        with torch.no_grad():
            condition_img = Variable(input["condition_img"].to(self.device, non_blocking=True), requires_grad=True)
            target_bone = Variable(input["target_bone"].to(self.device, non_blocking=True), requires_grad=True)
            g1_out = self.generator1(condition_img, target_bone)
            gfv = self.generator2.getGFV(condition_img, g1_out)
        return gfv

    def forward(self, input, action, render=False, printf=None):

        condition_img = Variable(input["condition_img"].to(self.device, non_blocking=True), requires_grad=True)
        target_bone = Variable(input["target_bone"].to(self.device, non_blocking=True), requires_grad=True)
        target_mask = Variable(input["target_mask"].to(self.device, non_blocking=True), requires_grad=True)

        with torch.no_grad():

            # Encoder Input
            g1_out = self.generator1(condition_img, target_bone)
            gfv = self.generator2.getGFV(condition_img, g1_out)
            g2_out = g1_out + self.generator2(condition_img, g1_out)
            
            # Generator Input
            z = Variable(action, requires_grad=True).to(self.device)

            # Generator Output
            out_G, _ = self.lgan_G(z)
            # out_G = torch.squeeze(out_GD, dim=1)
            # out_G = out_G.contiguous().view(-1, 384, 32, 16)
            g2_out_G = g1_out + self.generator2(condition_img, g1_out, out_G)

            # Discriminator Output
            out_D, _ = self.lgan_D(out_G)

        # Discriminator Loss
        loss_D = self.nll(out_D)

        # Loss Between Noisy GFV and Clean GFV
        loss_GFV = self.mse(out_G, gfv)

        # Norm Loss
        loss_norm = self.norm(z)

        # Chamfer loss
        mask_l1_loss = self.mask_l1_loss(g2_out_G, g2_out, target_mask)

        # States Formulation
        state_curr = np.array([
            loss_D.cpu().data.numpy(),
            loss_GFV.cpu().data.numpy(),
            mask_l1_loss.cpu().data.numpy(),
            loss_norm.cpu().data.numpy()
        ])

        reward_D = state_curr[0]
        reward_GFV =-state_curr[1]
        reward_l1 = -state_curr[2]
        reward_norm =-state_curr[3]

        # Reward Formulation
        # Possible combinations:
        #  0.01. 10.0  100.0 0.1
        #  1.00  10.0  100.0 0.002
        #  1.00  1.00  1.00  1/30
        #  0.002 10.0  100.0 1.0
        #  0.2   100.0 100.0 1.0
        reward = self.k1 * reward_D + self.k2 * reward_GFV + self.k3 * reward_l1 + self.k4 * reward_norm

        # measured elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()


        # Display stats
        s = '[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size, reward, state_curr, self.iter)
        if printf is not None:
            printf(s)

        if (self.iter * self.epoch_size + self.i) % self.log_freq == 0:
            with open(self.log_path, 'a+') as f:
                f.write(s + '\n')

        self.i += 1
        if(self.i >= self.epoch_size):
            self.i = 0
            self.iter +=1

        done = True
        state = out_G.detach().cpu().data.numpy().squeeze()
        return state, _, reward, done, self.losses.avg



def run(config, options, device=torch.device("cuda")):
    trainer = Trainer(config, options, device)
    trainer.train()







