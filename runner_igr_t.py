import argparse
import json
import math
import os
import pdb
import pickle
import random
import signal
import sys
from datetime import datetime
from venv import create
from weakref import ref

import matplotlib
matplotlib.use('Agg')

import trimesh
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import scipy.io as scio
import torch
from pyhocon import ConfigFactory
from torch.utils.data import DataLoader
import GPUtil


import lib
import wandb
from lib.dataset import Dataset, RawDataset
from lib.mesh import *
#from lib.models.decoder_relu import *
from lib.models.decoder_siren import *
from lib.models.network import *
from lib.models.sample import Sampler
from lib.utils import *
from lib.workspace import *
from lib.plots import *

import warnings
warnings.filterwarnings("ignore")

class Runner:
    def __init__(self, **kwargs):
        
        # Configuration
        self.conf_path = kwargs['conf_path']
        self.case = kwargs['case']
        f = open(self.conf_path, 'r')
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', kwargs['case'])
        # conf_text = conf_text.replace('WANDB_NAME', wandb_name)
        f.close()
        
        # wandb.init(project='NeuralMD_igr', name='test', entity='')
        
        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.use_wandb = kwargs['use_wandb']
        if self.use_wandb:
            self.wandb_name = self.conf['general.wandb_name']
            wandb.init(project='NeuralMD_igr', name=self.wandb_name, entity='baai-health-team')
        
        # Training parameters
        self.epoch = 0
        self.start_epoch = self.conf['train.start_iter']
        self.end_epoch = self.conf['train.end_iter']
        self.log_freq = self.conf['train.log_freq']
        self.validate_freq = self.conf['train.validation_freq']
        self.num_workers = self.conf['train.num_workers']
        
        if self.start_epoch >= self.end_epoch:
            raise Exception("Total number of train epoch should not be smaller than start epoch!")
        
        self.scene_per_batch = self.conf['train.scene_per_batch']
        self.grad_clip = get_spec_with_default(self.conf['train'], "gradient_clip_norm", None)
        self.clamp_dist = self.conf['train.clip_distance']
        self.enforce_minmax = self.conf['train.clamp_SDF_minmax']
        self.do_code_regularization = self.conf['train.code_regularization'] 
        self.code_reg_lambda = self.conf['train.code_regularization_lambda']
        
        # Weights
        self.mode = kwargs['mode']
        self.is_continue = kwargs['is_continue']
        self.interp_time = kwargs['interpolate_time']
        
        # Dataset & DataLoader
        self.num_sequences = None
        if self.mode == 'train':
            self.dataset = RawDataset(**self.conf['dataset'])
            print(f"self.dataset.s: {self.dataset.s}")
            print(f"self.dataset.t: {self.dataset.t}")
            print("Loading data with {} threads".format(self.num_workers))
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.scene_per_batch,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=False
            )
            self.dataloader_reconstruction = DataLoader(
                self.dataset,
                batch_size=self.scene_per_batch,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False
            )

            self.num_frames_total = len(self.dataset)
            self.num_sequences = self.dataset.num_sequences
            print("There are toral of {} frames across {} sequences".format(self.num_frames_total, self.num_sequences))
        
        else:
        # if self.mode != 'train':
            self.num_sequences = self.conf.get_int('reconstruction.num_sequences')
            
        # Networks
        # if not self.num_sequences:
        #     self.num_sequences = self.conf['dataset.num_sequences']
        self.d_in = self.conf.get_int('dataset.d_in')
        self.latent_size = self.conf['model.latent_vector.latent_size']
        self.grad_lambda = self.conf.get_float('model.loss.lambda')
        self.normals_lambda = self.conf.get_float('model.loss.normals_lambda')
        self.latent_lambda = self.conf.get_float('model.loss.latent_lambda')
        self.global_sigma = self.conf.get_float('model.sampler.properties.global_sigma')
        
        self.with_normals = self.normals_lambda > 0
        
        self.network = ImplicitNet(d_in=(self.d_in+self.latent_size+1), **self.conf.get_config('model.network')).cuda()
        self.network = torch.nn.DataParallel(self.network)
        print(self.network)
        self.lat_vecs = torch.nn.Embedding(self.num_sequences, self.latent_size).cuda()
        torch.nn.init.normal_(
            self.lat_vecs.weight.data,
            0.0,
            get_spec_with_default(self.conf['model.latent_vector'], "code_init_std_dev", 1.0) / math.sqrt(self.latent_size)
        )
        self.lat_vecs.requires_grad_()
        self.loss_log = None
        
        self.lr_schedules = get_learning_rate_schedules(self.conf)
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0)
                },
                {
                    "params": self.lat_vecs.parameters(),
                    "lr": self.lr_schedules[1].get_learning_rate(0)
                }
            ]
        )
        
        # settings for loading an existing experiment
        timestamp = kwargs['timestamp']

                
        if self.is_continue and kwargs['timestamp'] == 'latest':
            if os.path.exists(self.base_exp_dir):
                timestamps = os.listdir(self.base_exp_dir)
                if len(timestamps) == 0:
                    self.is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    self.is_continue = True
            else:
                self.is_continue = False
                timestamp = None
                
        
        if self.is_continue:
            self.timestamp = timestamp
        else:
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        
        self.cur_exp_dir = os.path.join(self.base_exp_dir, self.timestamp)
        os.makedirs(self.cur_exp_dir, exist_ok=True)
        
        self.checkpoints_path = os.path.join(self.cur_exp_dir, 'checkpoints')
        os.makedirs(self.checkpoints_path, exist_ok=True)

        # TODO: if mode == 'train
        
        # TODO: if is_continue
        if kwargs['ckpt'] == 'latest':
            self.ckpt = kwargs['ckpt']
        else:
            self.ckpt = '{:0>6d}'.format(int(kwargs['ckpt']))
        
        if self.is_continue:
            self.load_checkpoints()

        # Plots validation shapes
        self.plots_validation_dir = os.path.join(self.cur_exp_dir, 'plots')
        self.plots_reconstruction_dir = os.path.join(self.cur_exp_dir, 'reconstruction_plots')
        os.makedirs(self.plots_validation_dir, exist_ok=True)
        os.makedirs(self.plots_reconstruction_dir, exist_ok=True)
        
        
    def train(self):
        print("Training starting from epoch {}".format(self.start_epoch))
        
        if not self.loss_log:
            self.loss_log = {}
            self.loss_log['epoch'] = []
            self.loss_log['mnfld_loss'] = []
            self.loss_log['grad_loss'] = []
            self.loss_log['normals_loss'] = []
            self.loss_log['latent_loss'] = []
            self.loss_log['loss'] = []
        
        # if self.use_wandb:
        #     wandb.watch(self.network ,log='all')
        
        for epoch in range(self.start_epoch, self.end_epoch+1):
            self.epoch = epoch
            self.network.train()
            self.adjust_learning_rate(self.epoch)
            
            if self.use_wandb:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    lr = param_group["lr"]
                    wandb_lr = 'lr_' + str(i+1)
                    wandb.log({'epoch': self.epoch, wandb_lr: lr})
        
            for data, local_sigmas, t, _, sequence_id in self.dataloader:
                data = data.squeeze(0).cuda()
                indices = torch.tensor(np.random.choice(data.shape[0], data.shape[0], False))
                # return indices
                
                cur_data = data[indices]
                mnfld_pnts = cur_data[:, :self.d_in]
                normals = cur_data[:,self.d_in:]
                mnfld_sigma = local_sigmas.squeeze(0)[indices].cuda()
                
                 
                sampler = Sampler.get_sampler(self.conf.get_string('model.sampler.sampler_type'))(self.global_sigma,
                                                                                                    mnfld_sigma)
                nonmnfld_pnts = sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze(0)
                
                # add t
                mnfld_pnts_t = torch.cat((mnfld_pnts, t.unsqueeze(-1).repeat(1, mnfld_pnts.shape[0]).view(-1,1).cuda()), dim=1)
                nonmnfld_pnts_t = torch.cat((nonmnfld_pnts, t.unsqueeze(-1).repeat(1, nonmnfld_pnts.shape[0]).view(-1,1).cuda()), dim=1)
                
                # add latent vector
                mnfld_pnts_t = self.add_latent(mnfld_pnts_t, sequence_id)
                nonmnfld_pnts_t = self.add_latent(nonmnfld_pnts_t, sequence_id)
                    
                # forward pass
                mnfld_pnts.requires_grad_()
                nonmnfld_pnts.requires_grad_()
                
                mnfld_pred = self.network(mnfld_pnts_t)
                nonmnfld_pred = self.network(nonmnfld_pnts_t)
                
                mnfld_grad = gradient(mnfld_pnts_t, mnfld_pred)
                nonmnfld_grad = gradient(nonmnfld_pnts_t, nonmnfld_pred)
                
                # manifold loss
                mnfld_loss = (mnfld_pred.abs()).mean()
                
                # eikonal loss
                grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
                loss = mnfld_loss + self.grad_lambda * grad_loss
                
                # normals loss
                if self.with_normals:
                    normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
                    loss = loss + self.normals_lambda * normals_loss
                else:
                    normals_loss = torch.zeros(1)
                # latent loss
                latent_loss = self.latent_size_reg(sequence_id.cuda())
                loss = loss + self.latent_lambda * latent_loss
                
                
                # return mnfld_loss, grad_loss, normals_loss, latent_loss
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.use_wandb:
                    wandb.log({'epoch': self.epoch, 'mnfld_loss': mnfld_loss})
                    wandb.log({'epoch': self.epoch, 'grad_loss': grad_loss})
                    wandb.log({'epoch': self.epoch, 'normals_loss': normals_loss})
                    wandb.log({'epoch': self.epoch, 'latent_loss': latent_loss})
                    wandb.log({'epoch': self.epoch, 'loss':loss})
                
                self.loss_log['epoch'].append(epoch)
                self.loss_log['mnfld_loss'].append(mnfld_loss.item())
                self.loss_log['grad_loss'].append(grad_loss.item())
                self.loss_log['normals_loss'].append(normals_loss.item())
                self.loss_log['latent_loss'].append(latent_loss.item())
                self.loss_log['loss'].append(loss.item())
            # print status
            if epoch % self.conf.get_int('train.status_frequency') == 0 or epoch == self.start_epoch:
                print('Train Epoch: {} \tTrain Loss: {:.6f}\tManifold loss: {:.6f}'
                          '\tGrad loss: {:.6f}\tLatent loss: {:.6f}\tNormals Loss: {:.6f}'.format(
                        epoch, loss.item(), mnfld_loss.item(), grad_loss.item(), latent_loss.item(), normals_loss.item()))
            
            if epoch % self.conf.get_int('train.log_freq') == 0 or epoch == self.start_epoch:
                self.learned_vectors = np.zeros([self.lat_vecs.num_embeddings, self.lat_vecs.embedding_dim], dtype='float32')
                # self.num_frames_pre_seq = self.num_frames_total // self.num_sequences
                # self.dataset.frame_num_per_seq
                i = 0
                j = 0
                for _, _, time, name, sequence_id in self.dataloader_reconstruction:
                    latent = self.lat_vecs(sequence_id.cuda()).squeeze(0)
                    if i == (self.dataset.frame_num_per_seq[j] - 1):
                        self.learned_vectors[j,:] = latent.detach().cpu().numpy().astype(np.float32)
                        j += 1
                        i = 0
                        if j == len(self.dataset.frame_num_per_seq):
                            break
                    i += 1
                self.save_checkpoints()
                self.plot_loss_curve()
                self.plot_validation_shapes()
                pickle.dump(self.loss_log, open(os.path.join(self.cur_exp_dir, 'loss_log.pkl'), 'wb'))
    
    def plot_validation_shapes(self, with_cuts=False, with_data=True):
        print('Ploting validation shapes...')
        with torch.no_grad():
            self.network.eval()
            data, _,t, _, sequence_id = next(iter(self.dataloader))
            data = data.squeeze(0)
            indices = torch.tensor(np.random.choice(data.shape[0], data.shape[0], False))
            pnts = data[indices, :3]
            pnts_t = torch.cat([pnts,t.unsqueeze(-1).repeat(1, pnts.shape[0]).view(-1,1)], dim=1).cuda()
            pnts_t_lat = self.add_latent(pnts_t, sequence_id)
            latent = self.lat_vecs(sequence_id.cuda())
            shapename = 't_{:.6f}_seq_{:02d}'.format(t.item(), sequence_id.item())
            
            plot_surface(with_points=True,
                        points=pnts_t_lat,
                        t=t,
                        decoder=self.network,
                        latent=latent,
                        path=self.plots_validation_dir,
                        epoch=self.epoch,
                        shapename=shapename,
                        extension='obj',
                        **self.conf.get_config('plot'))
            
            if with_cuts:
                plot_cuts(points=pnts_t_lat,
                          decoder=self.network,
                          latent=latent,
                          path=self.plots_validation_dir,
                          epoch=self.epoch,
                          t=t,
                          near_zero=False)
                plot_cuts_axis(points=pnts_t_lat,
                               decoder=self.network,
                               latent=latent,
                               t=t,
                               path=self.plots_validation_dir,
                               epoch=self.epoch,
                               near_zero=False,
                               axis=0)
            
            
    def extract_sdf(self):
        with torch.no_grad():
            self.network.eval()
            data, _,t, _, sequence_id = next(iter(self.dataloader))
            data = data.squeeze(0)
            indices = torch.tensor(np.random.choice(data.shape[0], data.shape[0], False))
            pnts = data[indices, :3]
            pnts_t = torch.cat([pnts,t.unsqueeze(-1).repeat(1, pnts.shape[0]).view(-1,1)], dim=1).cuda()
            pnts_t_lat = self.add_latent(pnts_t, sequence_id)
            latent = self.lat_vecs(sequence_id.cuda())
            shapename = 't_{:.6f}_seq_{:02d}'.format(t.item(), sequence_id.item())

            sdf = get_sdf(with_points=True,
                        points=pnts_t_lat,
                        t=t,
                        decoder=self.network,
                        latent=latent,
                        path=self.plots_validation_dir,
                        epoch=self.epoch,
                        shapename=shapename,
                        **self.conf.get_config('plot'))
        return sdf
    
    
    
    def add_latent(self, points, sequence_ids):
        num_of_points, dim = points.shape
        batch_vecs = self.lat_vecs(sequence_ids.unsqueeze(-1).repeat(1, num_of_points).view(-1).cuda())

        points = torch.cat([batch_vecs, points], 1)
        return points
                                                                                             
                                                                                            
    def latent_size_reg(self, sequence_ids):
        latents = self.lat_vecs(sequence_ids)
        # latents = torch.index_select(self.lat_vecs, 0, sequence_ids)
        latent_loss = latents.norm(dim=1).mean()
        return latent_loss
    
    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)
    
    def save_checkpoints(self):
        print('Saving checkpoint...')
        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "latent_codes": self.lat_vecs.state_dict(),
            "learned_vectors": self.learned_vectors,
            "epoch": self.epoch
        }
        torch.save(checkpoint, os.path.join(self.checkpoints_path, 'ckpt_{:0>6d}.pth'.format(self.epoch)))
        torch.save(checkpoint, os.path.join(self.checkpoints_path, 'ckpt_latest.pth'))
        
    def load_checkpoints(self):
        checkpoint = torch.load(os.path.join(self.checkpoints_path, 'ckpt_' + self.ckpt + '.pth'))
        self.epoch = checkpoint['epoch']
        self.start_epoch = self.epoch + 1
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lat_vecs.load_state_dict(checkpoint['latent_codes'])
        self.learned_vectors = checkpoint['learned_vectors']
        self.loss_log = pickle.load(open(os.path.join(self.cur_exp_dir, 'loss_log.pkl'), 'rb'))
        
        
        
    def plot_level_set_shape(self,resolution, t, sequence_id, path, shape_name=None):
        # print("Ploting multiple level-set shapes...")
        os.makedirs(path, exist_ok=True)
        with torch.no_grad():
            self.network.eval()

            sequence_id = torch.tensor([sequence_id])
            t = torch.tensor([t])
            latent = self.lat_vecs(sequence_id.cuda())
            if not shape_name:
                shapename_inside = 'level_set_inside_t_{:.6f}_seq_{:02d}'.format(t.item(), sequence_id.item())
                shapename_outside = 'level_set_outside_t_{:.6f}_seq_{:02d}'.format(t.item(), sequence_id.item())
            else:
                shapename_inside = 'level_set_inside_' + shape_name
                shapename_outside = 'level_set_outside_' + shape_name
                
            t = t.cuda()
        
            plot_level_outside(path=path,
                               epoch=self.epoch,
                               decoder=self.network,
                               latent=latent,
                               t=t,
                               shapename=shapename_outside,
                               resolution=resolution)
            plot_level_inside(path=path,
                              epoch=self.epoch,
                              decoder=self.network,
                              latent=latent,
                              t=t,
                              shapename=shapename_inside,
                              resolution=resolution)
    
    def plot_interpolate_shape(self,resolution, t, sequence_id, path, shape_name=None):
        # print("Ploting interpolation shapes...")
        os.makedirs(path, exist_ok=True)
        with torch.no_grad():
            self.network.eval()
            sequence_id = torch.tensor([sequence_id])
            t = torch.tensor([t])
            latent = self.lat_vecs(sequence_id.cuda())
            if not shape_name:
                shape_name = 'interpolation_t_{:.6f}_seq_{:02d}'.format(t.item(), sequence_id.item())
            t = t.cuda()
            
            get_mesh(path=path,
                     epoch=self.epoch,
                     decoder=self.network,
                     latent=latent,
                     t=t,
                     shapename=shape_name,
                     resolution=resolution, 
                     verbose=False,
                     connected=False,
                     return_sdf=self.conf.get_bool('sdf.return_sdf'),
                     mc_value=self.conf.get_float('plot.mc_value'))
            
    def plot_loss_curve(self):
        print('Ploting  loss curves...')

        epoch = max(self.loss_log['epoch'])
        fig = plt.figure(figsize=(28, 15), dpi=150)
        loss_list = ['loss', 'mnfld_loss', 'grad_loss', 'normals_loss', 'latent_loss', 'epoch']
        for i in range(1, 7):
            plt.subplot(2, 3, i)
            plt.plot(self.loss_log['epoch'], self.loss_log[loss_list[i-1]])
            plt.xlim(0, epoch)
            plt.xlabel('epoch', fontsize=16)
            plt.title(loss_list[i-1], fontsize=16)
        figname = os.path.join(self.cur_exp_dir, 'loss_curve.png')
        fig.savefig(figname)      
       
        fig = plt.figure(figsize=(28, 15), dpi=150)

        steps = int(len(self.loss_log['epoch']) / max(self.loss_log['epoch']))
        epoch_np = np.array(range(1, max(self.loss_log['epoch'])+1))
        for i in range(1, 7):
            y = np.average(np.array(self.loss_log[loss_list[i-1]]).reshape(-1, steps), axis=1)
            plt.subplot(2, 3, i)
            plt.plot(epoch_np, y, linewidth=2.5)
            plt.xlim(0, epoch)
            plt.xlabel('epoch', fontsize=16)
            plt.title(loss_list[i-1], fontsize=16)
        figname = os.path.join(self.cur_exp_dir, 'loss_curve_average.png')
        fig.savefig(figname)

       
if __name__ == '__main__':
    print('Neural Molecular Dynamics Experiments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='../confs/trajs/default.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--timestamp', type=str, default='latest')
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU auto]')
    parser.add_argument('--case', type=str, default='trajectories_mesh')
    parser.add_argument('--use_wandb', default=False, action="store_true")
    parser.add_argument('--interpolate_time', type=float, default=None)
    parser.add_argument('--plot_level_set', default=False, action="store_true")
    parser.add_argument('--return_sdf', default=False, action="store_true")

    args = parser.parse_args()

    if args.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    
    runner = Runner(conf_path=args.conf,
                    mode=args.mode,
                    case=args.case,
                    is_continue=args.is_continue,
                    timestamp=args.timestamp,
                    ckpt=args.ckpt,
                    interpolate_time=args.interpolate_time,
                    use_wandb=
                    args.use_wandb)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'interpolation':
        print("Ploting interpolation shapes...")
        runner.plot_interpolate_shape(**runner.conf.get_config('interpolation'))
            
        if args.plot_level_set:
            print("Ploting multiple level-set shapes...")
            runner.plot_level_set_shape(**runner.conf.get_config('interpolation'))
    elif args.mode == 'reconstruction':
        # configs
        datafolders = get_instance_folders(runner.conf.get_string('dataset.data_dir'))
        frame_per_sequence_train = runner.conf.get_int('reconstruction.frame_per_sequence_train')
        train_frame_interval = runner.conf.get_int('reconstruction.train_frame_interval')
        resolution = runner.conf.get_int('reconstruction.resolution')
        reconstruct_frame_per_seq = runner.conf.get_list('reconstruction.reconstruct_frame_per_seq')
        # reconstruct_path = runner.conf.get_string('reconstruction.path')
        reconstruct_path = runner.plots_reconstruction_dir
        scale = runner.conf.get_float('dataset.s')
        translate = runner.conf.get_float('dataset.t')
        
        # times
        t_min = range(frame_per_sequence_train)[0] * train_frame_interval
        t_max = range(frame_per_sequence_train)[-1] * train_frame_interval
        reconstruction_times = []
        for frame in reconstruct_frame_per_seq:
            t = np.array(range(frame), dtype='float32')
            reconstruction_times.append(((t - t_min) / (t_max - t_min)) * 2. - 1.)
        
        # Reconstructing
        print(f'Reconstructing total sequences: {runner.num_sequences}')
        for sequence_id in range(runner.num_sequences):
            print(f'sequence {sequence_id+1}')
            folder = datafolders[sequence_id]
            for idx, t in tqdm(enumerate(reconstruction_times[sequence_id])):
                # if idx > 0:
                #     break
                
                path = os.path.join(reconstruct_path, f'seq_{sequence_id}_{folder}')
                os.makedirs(path, exist_ok=True)
                shape_name = 'frame_{:05d}_t_{:.6f}_seq_{:02d}'.format(idx, t, sequence_id)

                runner.plot_interpolate_shape(resolution=resolution,
                                            t=t,
                                            sequence_id=sequence_id,
                                            path=path,
                                            shape_name=shape_name)
                filename = '{0}/igr_epoch_{1}_{2}'.format(path, runner.epoch, shape_name)
                mesh = trimesh.load(f'{filename}.obj')
                os.system(f'rm {filename}.obj')
                mesh.vertices = ((mesh.vertices + 1.0) * scale) / 2.0 + translate
                mesh.export(filename + '.obj', 'obj')
                    
    print('Done!')       