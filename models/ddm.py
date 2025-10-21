import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from models.decom import CTDN
import cv2
from KAN import KANLinear
from Convolution import Convolutional_Layer


class DEFE(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.channels = channels
        

        def create_dwconv():
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, 
                          padding=1, groups=channels, bias=False),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            )
        
        self.dw_conv1 = create_dwconv()
        self.dw_conv2 = create_dwconv()
        self.dw_conv3 = create_dwconv()
        # self.dw_conv4 = create_dwconv()
        # self.dw_conv5 = create_dwconv()

        def create_kan_layer(grid_size, grid_range,scale_spline):
            return Convolutional_Layer(
                n_convs=1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),  
                dilation=(1, 1),
                grid_size=grid_size,
                spline_order=3,
                scale_spline=scale_spline,
                grid_range=grid_range,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        
        self.kan_layer1 = create_kan_layer(6, [-0.5, 0.5],0.5)
        self.kan_layer2 = create_kan_layer(8, [-0.4, 0.4],0.4)
        self.kan_layer3 = create_kan_layer(6, [-0.5, 0.5],0.4)
        # self.kan_layer4 = create_kan_layer(8, [-0.4, 0.4],0.4)
        # self.kan_layer5 = create_kan_layer(6, [-0.5, 0.5],0.4)
        

        self.norm = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        identity = x
        

        hf = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        

        out = self.kan_layer1(hf)
        out = self.dw_conv1(out)
        

        out = self.kan_layer2(out)
        out = self.dw_conv2(out)
        

        out = self.kan_layer3(out)
        out = self.dw_conv3(out)

        # out = self.kan_layer4(out)
        # out = self.dw_conv4(out)

        # out = self.kan_layer5(out)
        # out = self.dw_conv5(out)

        out = self.norm(out)
        return identity + self.alpha * out
    


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device
        self.hf_enhance = DEFE()
        self.Unet = DiffusionUNet(config)
        if self.args.mode == 'training':
            self.decom = self.load_stage1(CTDN(), 'ckpt/stage1')
        else:
            self.decom = CTDN()

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    @staticmethod
    def load_stage1(model, model_dir):
        checkpoint = utils.logging.load_checkpoint(os.path.join(model_dir, 'stage1_weight.pth.tar'), 'cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        return model

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, inputs):
        data_dict = {}

        b = self.betas.to(inputs.device)

        if self.training:
            output = self.decom(inputs, pred_fea=None)
            low_R, low_L, low_fea, high_L = output["low_R"], output["low_L"], \
                output["low_fea"], output["high_L"]

            low_condition_norm = utils.data_transform(low_fea)
            t = torch.randint(low=0, high=self.num_timesteps, size=(low_condition_norm.shape[0] // 2 + 1,)).to(
                self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:low_condition_norm.shape[0]].to(inputs.device)
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

            e = torch.randn_like(low_condition_norm)

            high_input_norm_raw = utils.data_transform(low_R * high_L)
            high_input_norm =self.hf_enhance(high_input_norm_raw)

            data_dict["high_input_norm_raw"] = high_input_norm_raw
            data_dict["high_input_norm"] = high_input_norm
            x = high_input_norm * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([low_condition_norm, x], dim=1), t.float())
            pred_fea = self.sample_training(low_condition_norm, b)
            pred_fea = utils.inverse_data_transform(pred_fea)
            reference_fea = low_R * torch.pow(low_L, 0.2)

            data_dict["noise_output"] = noise_output
            data_dict["e"] = e
            data_dict["pred_fea"] = pred_fea
            data_dict["reference_fea"] = reference_fea
            
        else:
            output = self.decom(inputs, pred_fea=None)
            low_fea = output["low_fea"]
            low_condition_norm = utils.data_transform(low_fea)

            pred_fea = self.sample_training(low_condition_norm, b)
            pred_fea = utils.inverse_data_transform(pred_fea)
            pred_x = self.decom(inputs, pred_fea=pred_fea)["pred_img"]
            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        
        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0



    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))


    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for name, param in self.model.named_parameters():
            if "decom" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                output = self.model(x)

                noise_loss, scc_loss,edge_loss = self.noise_estimation_loss(output)
                loss = noise_loss + scc_loss+ edge_loss

                data_time += time.time() - data_start

                if self.step % 10 == 0:
                    print("step:{}, noise_loss:{:.5f} scc_loss:{:.5f} edge_loss:{:.5f} time:{:.5f}".
                          format(self.step, noise_loss.item(),
                                 scc_loss.item(),edge_loss.item() ,data_time / (i + 1)))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                    utils.logging.save_checkpoint({'step': self.step,
                                                   'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir, 'model_latest'))

    def edge_preservation_loss(self, pred, target):

        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                            dtype=torch.float32, device=pred.device)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                            dtype=torch.float32, device=pred.device)

        sobel_x = sobel_x.repeat(pred.size(1), 1, 1, 1)
        sobel_y = sobel_y.repeat(pred.size(1), 1, 1, 1)

        grad_pred_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.size(1))
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.size(1))
        grad_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-8)

        grad_target_x = F.conv2d(target, sobel_x, padding=1, groups=target.size(1))
        grad_target_y = F.conv2d(target, sobel_y, padding=1, groups=target.size(1))
        grad_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-8)

        return F.l1_loss(grad_pred, grad_target)
    def noise_estimation_loss(self, output,):
        pred_fea, reference_fea = output["pred_fea"], output["reference_fea"]
        noise_output, e = output["noise_output"], output["e"]
        high_input_norm_raw = output["high_input_norm_raw"]
        high_input_norm = output["high_input_norm"]
        # ==================noise loss==================
        noise_loss = self.l2_loss(noise_output, e)
        # ==================scc loss==================
        scc_loss = 0.002 * self.l1_loss(pred_fea, reference_fea)
        # ================== edge_loss ==================
        edge_loss = 0.3 *self.edge_preservation_loss(high_input_norm, high_input_norm_raw)
        return noise_loss, scc_loss, edge_loss

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder,
                                    self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()

        with torch.no_grad():
            print('Performing validation at step: {}'.format(step))
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape

                img_h_64 = int(64 * np.ceil(img_h / 64.0))
                img_w_64 = int(64 * np.ceil(img_w / 64.0))
                x = F.pad(x, (0, img_w_64 - img_w, 0, img_h_64 - img_h), 'reflect')
                pred_x = self.model(x.to(self.device))["pred_x"][:, :, :img_h, :img_w]
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step), '{}'.format(y[0])))
                
