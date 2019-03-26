import os
from collections import OrderedDict
import pandas as pd
import scipy.misc as misc

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as thutil

from networks import create_model
from .base_solver import BaseSolver
from networks import init_weights
from utils import util

class SRSolver(BaseSolver):
    def __init__(self, opt):
        super(SRSolver, self).__init__(opt)
        self.train_opt = opt['solver']
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.SR = None

        self.records = {'train_loss': [],
                        'val_loss': [],
                        'psnr': [],
                        'ssim': [],
                        'lr': []}

        self.model = create_model(opt)
        self.print_network()

        if self.is_train:
            self.model.train()

            # set cl_loss
            if self.use_cl:
                self.cl_weights = self.opt['solver']['cl_weights']
                assert self.cl_weights, "[Error] 'cl_weights' is not be declared when 'use_cl' is true"

            # set loss
            loss_type = self.train_opt['loss_type']
            if loss_type == 'l1':
                self.criterion_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion_pix = nn.MSELoss()
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!'%loss_type)

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer = optim.Adam(self.model.parameters(),
                                            lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!' % optim_type)

            # set lr_scheduler
            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('Only MultiStepLR scheme is supported!')

        self.load()

        print('===> Solver Initialized : [%s] || Use CL : [%s] || Use GPU : [%s]'%(self.__class__.__name__,
                                                                                       self.use_cl, self.use_gpu))
        if self.is_train:
            print("optimizer: ", self.optimizer)
            print("lr_scheduler milestones: %s   gamma: %f"%(self.scheduler.milestones, self.scheduler.gamma))

    def _net_init(self, init_type='kaiming'):
        print('==> Initializing the network using [%s]'%init_type)
        init_weights(self.model, init_type)


    def feed_data(self, batch, need_HR=True):
        input = batch['LR']
        self.LR.resize_(input.size()).copy_(input)

        if need_HR:
            target = batch['HR']
            self.HR.resize_(target.size()).copy_(target)


    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        loss_batch = 0.0
        sub_batch_size = int(self.LR.size(0) / self.split_batch)
        for i in range(self.split_batch):
            loss_sbatch = 0.0
            split_LR = self.LR.narrow(0, i*sub_batch_size, sub_batch_size)
            split_HR = self.HR.narrow(0, i*sub_batch_size, sub_batch_size)
            if self.use_cl:
                outputs = self.model(split_LR)
                loss_steps = [self.criterion_pix(sr, split_HR) for sr in outputs]
                for step in range(len(loss_steps)):
                    loss_sbatch += self.cl_weights[step] * loss_steps[step]
            else:
                output = self.model(split_LR)
                loss_sbatch = self.criterion_pix(output, split_HR)

            loss_sbatch /= self.split_batch
            loss_sbatch.backward()

            loss_batch += (loss_sbatch.item())

        # for stable training
        if loss_batch < self.skip_threshold * self.last_epoch_loss:
            self.optimizer.step()
            self.last_epoch_loss = loss_batch
        else:
            print('[Warning] Skip this batch! (Loss: {})'.format(loss_batch))

        self.model.eval()
        return loss_batch


    def test(self):
        self.model.eval()
        with torch.no_grad():
            forward_func = self._overlap_crop_forward if self.use_chop else self.model.forward
            if self.self_ensemble and not self.is_train:
                SR = self._forward_x8(self.LR, forward_func)
            else:
                SR = forward_func(self.LR)

            if isinstance(SR, list):
                self.SR = SR[-1]
            else:
                self.SR = SR

        self.model.train()
        if self.is_train:
            loss_pix = self.criterion_pix(self.SR, self.HR)
            return loss_pix.item()


    def _forward_x8(self, x, forward_function):
        """
        self ensemble
        """
        def _transform(v, op):
            v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = self.Tensor(tfnp)

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = []
        for aug in lr_list:
            sr = forward_function(aug)
            if isinstance(sr, list):
                sr_list.append(sr[-1])
            else:
                sr_list.append(sr)

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


    def _overlap_crop_forward(self, x, shave=10, min_size=100000, bic=None):
        """
        chop for less memory consumption during test
        """
        n_GPUs = 2
        scale = self.scale
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if bic is not None:
            bic_h_size = h_size*scale
            bic_w_size = w_size*scale
            bic_h = h*scale
            bic_w = w*scale
            
            bic_list = [
                bic[:, :, 0:bic_h_size, 0:bic_w_size],
                bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
                bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
                bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if bic is not None:
                    bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)

                sr_batch_temp = self.model(lr_batch)

                if isinstance(sr_batch_temp, list):
                    sr_batch = sr_batch_temp[-1]
                else:
                    sr_batch = sr_batch_temp

                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self._overlap_crop_forward(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
                ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output


    def save_checkpoint(self, epoch, is_best):
        """
        save checkpoint to experimental dir
        """
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        print('===> Saving last checkpoint to [%s] ...]'%filename)
        ckp = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
            'records': self.records
        }
        torch.save(ckp, filename)
        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' % filename.replace('last_ckp','best_ckp'))
            torch.save(ckp, filename.replace('last_ckp','best_ckp'))

        if epoch % self.train_opt['save_ckp_step'] == 0:
            print('===> Saving checkpoint [%d] to [%s] ...]' % (epoch,
                                                                filename.replace('last_ckp','epoch_%d_ckp.pth'%epoch)))

            torch.save(ckp, filename.replace('last_ckp','epoch_%d_ckp.pth'%epoch))


    def load(self):
        """
        load or initialize network
        """
        if (self.is_train and self.opt['solver']['pretrain']) or not self.is_train:
            model_path = self.opt['solver']['pretrained_path']
            if model_path is None: raise ValueError("[Error] The 'pretrained_path' does not declarate in *.json")

            print('===> Loading model from [%s]...' % model_path)
            if self.is_train:
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['state_dict'])

                if self.opt['solver']['pretrain'] == 'resume':
                    self.cur_epoch = checkpoint['epoch'] + 1
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.best_pred = checkpoint['best_pred']
                    self.best_epoch = checkpoint['best_epoch']
                    self.records = checkpoint['records']

            else:
                checkpoint = torch.load(model_path)
                if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
                load_func = self.model.load_state_dict if isinstance(self.model, nn.DataParallel) \
                    else self.model.module.load_state_dict
                load_func(checkpoint)

        else:
            self._net_init()


    def get_current_visual(self, need_np=True, need_HR=True):
        """
        return LR SR (HR) images
        """
        out_dict = OrderedDict()
        out_dict['LR'] = self.LR.data[0].float().cpu()
        out_dict['SR'] = self.SR.data[0].float().cpu()
        if need_np:  out_dict['LR'], out_dict['SR'] = util.Tensor2np([out_dict['LR'], out_dict['SR']],
                                                                        self.opt['rgb_range'])
        if need_HR:
            out_dict['HR'] = self.HR.data[0].float().cpu()
            if need_np: out_dict['HR'] = util.Tensor2np([out_dict['HR']],
                                                           self.opt['rgb_range'])[0]
        return out_dict


    def save_current_visual(self, epoch, iter):
        """
        save visual results for comparison
        """
        if epoch % self.save_vis_step == 0:
            visuals_list = []
            visuals = self.get_current_visual(need_np=False)
            visuals_list.extend([util.quantize(visuals['HR'].squeeze(0), self.opt['rgb_range']),
                                 util.quantize(visuals['SR'].squeeze(0), self.opt['rgb_range'])])
            visual_images = torch.stack(visuals_list)
            visual_images = thutil.make_grid(visual_images, nrow=2, padding=5)
            visual_images = visual_images.byte().permute(1, 2, 0).numpy()
            misc.imsave(os.path.join(self.visual_dir, 'epoch_%d_img_%d.png' % (epoch, iter + 1)),
                        visual_images)


    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']


    def update_learning_rate(self, epoch):
        self.scheduler.step(epoch)


    def get_current_log(self):
        log = OrderedDict()
        log['epoch'] = self.cur_epoch
        log['best_pred'] = self.best_pred
        log['best_epoch'] = self.best_epoch
        log['records'] = self.records
        return log


    def set_current_log(self, log):
        self.cur_epoch = log['epoch']
        self.best_pred = log['best_pred']
        self.best_epoch = log['best_epoch']
        self.records = log['records']


    def save_current_log(self):
        data_frame = pd.DataFrame(
            data={'train_loss': self.records['train_loss']
                , 'val_loss': self.records['val_loss']
                , 'psnr': self.records['psnr']
                , 'ssim': self.records['ssim']
                , 'lr': self.records['lr']
                  },
            index=range(1, self.cur_epoch + 1)
        )
        data_frame.to_csv(os.path.join(self.records_dir, 'train_records.csv'),
                          index_label='epoch')


    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                                 self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'), 'w') as f:
                f.writelines(net_lines)

        print("==================================================")
