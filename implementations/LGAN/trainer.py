
import os
import time
import torch
import datetime
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm

from .sagan_models import Generator, Discriminator
from .utils import *
from collections import OrderedDict


class Trainer(object):
    def __init__(self, data_loader, config, device):

        lgan_config = config["model"]["lgan"]
        train_config = config["train"]

        # decoder settings
        # self.model_decoder = model_decoder
        # self.chamfer =chamfer
        # self.vis = vis_Valida
        # self.j =0

        self.device = device

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.config = lgan_config
        self.model = lgan_config["model"]
        self.adv_loss = lgan_config["adv_loss"]

        # Model hyper-parameters
        self.imsize = lgan_config["imsize"]
        self.g_num = lgan_config["g_num"]
        self.z_dim = lgan_config["z_dim"]
        self.g_conv_dim = lgan_config["g_conv_dim"]
        self.d_conv_dim = lgan_config["d_conv_dim"]
        self.parallel = False#config["parallel

        self.lambda_gp = lgan_config["lambda_gp"]
        self.total_step = train_config["total_step"]
        self.d_iters = train_config["d_iters"]
        self.batch_size = train_config["batch_size"]
        self.num_workers = train_config["num_workers"]
        self.g_lr = train_config["g_lr"]
        self.d_lr = train_config["d_lr"]
        self.lr_decay = train_config["lr_decay"]
        self.beta1 = train_config["beta1"]
        self.beta2 = train_config["beta2"]
        self.pretrained_model = None #config["pretrained_model"]

        self.dataset = config["dataset"]
        self.use_tensorboard = config["log"]["use_tensorboard"]
        self.image_path = config["dataset"]["path"]["train"]["image"]
        self.log_path = os.path.join(config["output"], 'logs', 'sagan-1')
        self.model_save_path = os.path.join(config["output"], 'models', 'sagan-1')
        self.sample_path = os.path.join(config["output"], 'samples', 'sagan-1')
        self.log_step = config["log"]["log_step"]
        # self.sample_step = config["sample_step"]
        self.model_save_step = config["log"]["model_save_step"]
        self.version = 'sagan-1'

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()



    def train(self):

        step_per_epoch = len(self.data_loader)

        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z_np = np.arange(-self.config["max_action"],self.config["max_action"],(self.config["max_action"]*2)/50)#self.batchsize replace with 10
        fixed_z_n = tensor2var(torch.FloatTensor(fixed_z_np,))
        fixed_z = fixed_z_n.unsqueeze(1)
       # fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
    #    fixed_z = tensor2var(torch.)
        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()

        step = start
        while step < self.total_step:

            # ================== Train D ================== #
            self.D.train()
            self.G.train()
        
            with tqdm(enumerate(self.data_loader), total=len(self.data_loader)) as pbar: # progress bar
                for i, real_images in pbar:

                    # Compute loss with real images
                    # dr1, dr2, df1, df2, gf1, gf2 are attention scores
                    real_images = tensor2var(real_images)
                    d_out_real, dr1 = self.D(real_images)#,dr2
                    if self.adv_loss == 'wgan-gp':
                        d_loss_real = - torch.mean(d_out_real)
                    elif self.adv_loss == 'hinge':
                        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

                    # apply Gumbel Softmax
                    z = tensor2var((torch.randn(real_images.size(0), self.z_dim)))
                    fake_images, gf1 = self.G(z) #,gf2
                    d_out_fake, df1 = self.D(fake_images) #,df2

                    if self.adv_loss == 'wgan-gp':
                        d_loss_fake = d_out_fake.mean()
                    elif self.adv_loss == 'hinge':
                        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()


                    # Backward + Optimize
                    d_loss = d_loss_real + d_loss_fake
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()


                    if self.adv_loss == 'wgan-gp':
                        # Compute gradient penalty
                        alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.device).expand_as(real_images)
                        interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                        out,_= self.D(interpolated) # TODO "_"

                        grad = torch.autograd.grad(outputs=out,
                                                inputs=interpolated,
                                                grad_outputs=torch.ones(out.size(), device=self.device),
                                                retain_graph=True,
                                                create_graph=True,
                                                only_inputs=True)[0]

                        grad = grad.view(grad.size(0), -1)
                        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                        # Backward + Optimize
                        d_loss = self.lambda_gp * d_loss_gp

                        self.reset_grad()
                        d_loss.backward()
                        self.d_optimizer.step()

                    # ================== Train G and gumbel ================== #
                    # Create random noise
                    z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
                    fake_images,_ = self.G(z) # _

                    # Compute loss with fake images
                    g_out_fake,_ = self.D(fake_images)  # batch x n  TODO "_"
                    if self.adv_loss == 'wgan-gp':
                        g_loss_fake = - g_out_fake.mean()
                    elif self.adv_loss == 'hinge':
                        g_loss_fake = - g_out_fake.mean()

                    self.reset_grad()
                    g_loss_fake.backward()
                    self.g_optimizer.step()

                    s = "G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_out_fake: {:.4f}".format(
                        step + 1, self.total_step, (step + 1), self.total_step , d_loss_real.data.item(), d_loss_fake.data.item(), g_loss_fake.data.item()
                    )
                    pbar.set_description(s)

                    # Write log info
                    if (step + 1) % self.log_step == 0:
                        elapsed = time.time() - start_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        s = "Elapsed [{}], ".format(elapsed) + s
                        with open(os.path.join(self.log_path, 'log.txt'), 'a+') as file:
                            file.write(s)

                    # Sample images
                    # if (step + 1) % self.sample_step == 0:
                    #     fake_images,_= self.G(fixed_z) #TODO "_"

                    #     encoded = fake_images.contiguous().view(50,128) # 64,[1024,128] Gan output dims

                    #     pc_1 = self.model_decoder(encoded) #   real_images.contiguous().view(64,128)
                    #     #pc_1_temp = pc_1[0, :, :]

                    #     epoch =0
                    #     for self.j in range(0,50):#self.bacth_size
                    #         pc_1_temp = pc_1[self.j, :, :]
                    #         test = fixed_z.detach().cpu().numpy()
                    #         test1 = np.asscalar(test[self.j,0])
                    #        # test1 = 0
                    #         visuals = OrderedDict(
                    #             [('Validation Predicted_pc', pc_1_temp.detach().cpu().numpy())])
                    #         self.vis[self.j].display_current_results(visuals, epoch, step,z =str(test1))


                    #     save_image(denorm(fake_images.data),
                    #                os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

                    if (step+1) % model_save_step==0:
                        torch.save(self.G.state_dict(), os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                        torch.save(self.D.state_dict(), os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

                    step += 1

    def build_model(self):

        self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).to(self.device)
        self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).to(self.device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
