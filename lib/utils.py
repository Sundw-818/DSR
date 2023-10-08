#!/usr/bin/env python3

import os
import pdb
import logging
import torch
import trimesh
import glob
# import lib.workspace as ws
import numpy as np
import imageio
import pickle
from scipy.spatial import cKDTree

import scipy.io as sio # matlab loading

from tqdm import tqdm # progress bar

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj

def compute_normal_consistency(gt_normal, pred_normal):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """
    gt_normal_np = gt_normal.float().detach().cpu().numpy()[0]
    gt_mask_np = (gt_normal[...,0]>0).float().detach().cpu().numpy()[0]

    pred_normal_np = pred_normal.float().detach().cpu().numpy()[0]
    pred_mask_np = (pred_normal[...,0]>0).float().detach().cpu().numpy()[0]

    # take valid intersection
    inner_mask = (gt_mask_np * pred_mask_np).astype(bool)

    gt_vecs = 2*gt_normal_np[inner_mask]-1
    pred_vecs = 2*pred_normal_np[inner_mask]-1
    metric = np.mean(np.sum(gt_vecs*pred_vecs, 1))

    return metric

class Renderer(torch.nn.Module):
    def __init__(self, silhouette_renderer, depth_renderer, max_depth = 5, image_size=256):
        super().__init__()
        self.silhouette_renderer = silhouette_renderer
        self.depth_renderer = depth_renderer

        self.max_depth = max_depth

        # sobel filters
        with torch.no_grad():
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")

        # Pixel coordinates
        self.X, self.Y = torch.meshgrid(torch.arange(0, image_size), torch.arange(0, image_size))
        self.X = (2*(0.5 + self.X.unsqueeze(0).unsqueeze(-1))/image_size - 1).float().cuda()
        self.Y = (2*(0.5 + self.Y.unsqueeze(0).unsqueeze(-1))/image_size - 1).float().cuda()

    def depth_2_normal(self, depth, depth_unvalid, cameras):

        B, H, W, C = depth.shape

        grad_out = torch.zeros(B, H, W, 3).cuda()
        # Pixel coordinates
        xy_depth = torch.cat([self.X, self.Y, depth], 3).cuda().reshape(B,-1, 3)
        xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)

        # compute tangent vectors
        XYZ_camera = xyz_unproj.reshape(B, H, W, 3)
        vx = XYZ_camera[:,1:-1,2:,:]-XYZ_camera[:,1:-1,1:-1,:]
        vy = XYZ_camera[:,2:,1:-1,:]-XYZ_camera[:,1:-1,1:-1,:]

        # finally compute cross product
        normal = torch.cross(vx.reshape(-1, 3),vy.reshape(-1, 3))
        normal_norm = normal.norm(p=2, dim=1, keepdim=True)

        normal_normalized = normal.div(normal_norm)
        # reshape to image
        normal_out = normal_normalized.reshape(B, H-2, W-2, 3)
        grad_out[:,1:-1,1:-1,:] = (0.5 - 0.5*normal_out)

        # zero out +Inf
        grad_out[depth_unvalid] = 0.0

        return grad_out

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        # take care of soft silhouette
        silhouette_ref = self.silhouette_renderer(meshes_world=meshes_world, **kwargs)
        silhouette_out = silhouette_ref[..., 3]

        # now get depth out
        depth_ref = self.depth_renderer(meshes_world=meshes_world, **kwargs)
        depth_ref = depth_ref.zbuf[...,0].unsqueeze(-1)
        depth_unvalid = depth_ref<0
        depth_ref[depth_unvalid] = self.max_depth
        depth_out = depth_ref[..., 0]

        # post process depth to get normals, contours
        normals_out = self.depth_2_normal(depth_ref, depth_unvalid.squeeze(-1), kwargs['cameras'])

        return normals_out, silhouette_out

def process_image(images_out, alpha_out):
    image_out_export = 255*images_out.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    alpha_out_export = 255*alpha_out.detach().cpu().numpy()[0]
    image_out_export = np.concatenate( (image_out_export, alpha_out_export[:,:,np.newaxis]), -1 )
    return image_out_export.astype(np.uint8)

def store_image(image_filename, images_out, alpha_out):
    image_out_export = process_image(images_out, alpha_out)
    imageio.imwrite(image_filename, image_out_export)

def interpolate_on_faces(field, faces):
    #TODO: no batch support for now
    nv = field.shape[0]
    nf = faces.shape[0]
    field = field.reshape((nv, 1))
    # pytorch only supports long and byte tensors for indexing
    face_coordinates = field[faces.long()].squeeze(0)
    centroids = 1.0/3 * torch.sum(face_coordinates, 1)
    return centroids

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length



def get_learning_rate_schedules(conf):
    learning_rate_schedules = conf['train.learning_rate_schedules']
    schedules = []

    for key in learning_rate_schedules:
        if learning_rate_schedules[key]['type'] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    learning_rate_schedules[key]['initial'],
                    learning_rate_schedules[key]['interval'],
                    learning_rate_schedules[key]['factor']
                )
            )
        elif learning_rate_schedules[key]['type'] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    learning_rate_schedules[key]['initial'],
                    learning_rate_schedules[key]['interval'],
                    learning_rate_schedules[key]['factor']
                )
            )
        elif learning_rate_schedules[key]['type'] == "Constant":
            schedules.append(ConstantLearningRateSchedule(learning_rate_schedules[key]['value']))
        
        else:
            raise Exception('no known learning rate schedule of type "{}"'.format(
                learning_rate_schedules[key]['type']
            ))
    
    return schedules




# def load_optimizer(experiment_directory, filename, optimizer):

#     full_filename = os.path.join(
#         ws.get_optimizer_params_dir(experiment_directory), filename
#     )

#     if not os.path.isfile(full_filename):
#         raise Exception(
#             'optimizer state dict "{}" does not exist'.format(full_filename)
#         )

#     data = torch.load(full_filename)

#     optimizer.load_state_dict(data["optimizer_state_dict"])

#     return data["epoch"]



# def load_latent_vectors(experiment_directory, filename, lat_vecs):

#     full_filename = os.path.join(
#         ws.get_latent_codes_dir(experiment_directory), filename
#     )

#     if not os.path.isfile(full_filename):
#         raise Exception('latent state file "{}" does not exist'.format(full_filename))

#     data = torch.load(full_filename)

#     if isinstance(data["latent_codes"], torch.Tensor):

#         # for backwards compatibility
#         if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
#             raise Exception(
#                 "num latent codes mismatched: {} vs {}".format(
#                     lat_vecs.num_embeddings, data["latent_codes"].size()[0]
#                 )
#             )

#         if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
#             raise Exception("latent code dimensionality mismatch")

#         for i, lat_vec in enumerate(data["latent_codes"]):
#             lat_vecs.weight.data[i, :] = lat_vec

#     else:
#         lat_vecs.load_state_dict(data["latent_codes"])

#     return data["epoch"]


# def save_logs(
#     experiment_directory,
#     loss_log,
#     epoch,
# ):

#     torch.save(
#         {
#             "epoch": epoch,
#             "loss": loss_log,
#         },
#         os.path.join(experiment_directory, ws.logs_filename),
#     )


# def load_logs(experiment_directory):

#     full_filename = os.path.join(experiment_directory, ws.logs_filename)

#     if not os.path.isfile(full_filename):
#         raise Exception('log file "{}" does not exist'.format(full_filename))

#     data = torch.load(full_filename)

#     return (
#         data["loss"],
#         data["epoch"],
#     )


def clip_logs(loss_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)
    loss_log = loss_log[: (iters_per_epoch * epoch)]

    return loss_log


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def fourier_transform(x, L=5):
    cosines = torch.cat([torch.cos(2**l*3.1415*x) for l in range(L)], -1)
    sines = torch.cat([torch.sin(2**l*3.1415*x) for l in range(L)], -1)
    transformed_x = torch.cat((cosines,sines),-1)
    return transformed_x


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]
    latent_repeat = latent_vector.expand(num_samples, -1)
    sdf = decoder(latent_repeat, queries)
    return sdf


def get_instance_filenames(data_source, extension='.pkl'):
    # find all matlab files
    file_list = [file for file in os.listdir(data_source) if file.endswith(extension)]
    file_list.sort()
    return file_list

def get_instance_folders(data_source):
    # find all folders in data_source dir
    file_list = [file for file in os.listdir(data_source) if os.path.isdir(os.path.join(data_source, file))]
    file_list.sort()
    return file_list
    
    
def unpack_sdf_samples_fraction(filename, fraction, radius, sequence_id=0):
    
    print('Processing ', filename)
    # load pkl file
    sdf_protein = pickle.load(open(filename, 'rb'))
    sdf_protein['points'].shape
    points = sdf_protein['points']
    sdf_t = sdf_protein['sdf']
    
    frames = []
    times = [int(t) for t in sdf_t.keys()]
    times = sorted(times)
    # for t, sdf in sdf_t.items():
    for t in tqdm(times, desc="sequence {}".format(sequence_id), leave=False):
        sdf = sdf_t[str(t)]
        samples = torch.zeros(points.shape[0], 4)

        samples[:,0:3] = torch.from_numpy(points / radius).float()
        samples[:,3] = torch.from_numpy(sdf / radius).float()
        
        sample_size = int(samples.shape[0] * fraction)
        # separate inner and outer points
        index_ip = np.where(samples[:,3] <= 0.1)
        ip_tensor = samples[index_ip,:]
        index_op = np.where(samples[:,3] > 0.1)
        op_tensor = samples[index_op,:] 
        # 70% of samples are inner points
        no_ip = int(sample_size * 0.7)
        if len(index_ip[0]) < no_ip:
            no_ip = len(index_ip[0])
            samples_ip = ip_tensor

        else:
            ip_rnd_idx = torch.randint(0, ip_tensor.shape[1], (no_ip,))
            #random_bp = (torch.rand(no_bp).cpu() * no_bp).long()
            samples_ip = torch.index_select(ip_tensor, 1, ip_rnd_idx)
        no_op = sample_size - no_ip
            
        op_rnd_idx = torch.randint(0, op_tensor.shape[1], (no_op,))
        samples_op = torch.index_select(op_tensor, 1, op_rnd_idx)
        
        samples = torch.cat([samples_ip, samples_op], 1).squeeze().float()
        frames.append((samples, t, filename, sequence_id)) 

    # normalize time coordinate
    t = np.array(times, dtype='float32')
    t = (t - np.min(t)) / (np.max(t) - np.min(t)) * 2. - 1. 
    frames = [(scene[0], t[i], *scene[2:]) for i, scene in enumerate(frames)] 
    
    return frames

def load_point_cloud_by_file_extension(file_name, with_normal=True):

    ext = file_name.split('.')[-1]
    mesh = trimesh.load(file_name, ext)
    point_set = torch.tensor(mesh.vertices).float()

    if with_normal:
        vertices_normal = torch.tensor(mesh.vertex_normals).float()
        point_set = (point_set, vertices_normal)
    return point_set

def unpack_point_samples(folder_name, extension='obj', with_normal=True, translate=-20.0, scale=50.0, sequence_id=0):
    
    print('Processing, ', folder_name)
    objfiles = get_instance_filenames(folder_name, extension=extension)
    objfiles.sort(key=lambda i: int(i.split('_')[-1][0:-(len(extension)+1)]))
    
    # times = [int(file.split('_')[-1][0:-(len(extension)+1)]) for file in objfiles]
    frames = []
    times = []
    
    for file in tqdm(objfiles, desc=f"sequence {sequence_id}"):
        t = int(file.split('_')[-1][0:-(len(extension)+1)])
        times.append(t)
        if with_normal:
            raw_data, vertices_normal = load_point_cloud_by_file_extension(os.path.join(folder_name, file), with_normal=True)
        else:
            raw_data = load_point_cloud_by_file_extension(os.path.join(folder_name, file), with_normal=False)
            
        scaled_data = ((raw_data - translate) / scale) * 2. - 1.    # scale the data to [-1, 1]
        
        # local sigmas 
        sigma_set = []
        ptree = cKDTree(scaled_data)

        for p in np.array_split(scaled_data, 20, axis=0):
            d = ptree.query(p, 50 + 1)
            sigma_set.append(d[0][:, -1])

        sigmas = np.concatenate(sigma_set, axis=0) 
        local_sigmas = torch.from_numpy(sigmas).float()
        
        if with_normal:
            scaled_data = torch.cat((scaled_data, vertices_normal), axis=1)
            
        frames.append((scaled_data, local_sigmas, t, os.path.join(folder_name, file), sequence_id))
    
    # normalize time coordinate
    t = np.array(times, dtype='float32')
    t = (t - np.min(t)) / (np.max(t) - np.min(t)) * 2. - 1.
    
    frames = [(*scene[0:2], t[i], *scene[3:]) for i, scene in enumerate(frames)]
        
    return frames

def unpack_sdf_samples(filename, subsample=32768):

    # load matlab file
    sdf = np.asarray([sio.loadmat(filename)['sdf_norm']], dtype=np.float32)     
    sdf = np.squeeze(sdf)
    
    # create coordinate grid
    pixel_coords = np.stack(np.mgrid[:sdf.shape[0], :sdf.shape[1], :sdf.shape[2]], axis=-1)[None, ...].astype(np.float32)
    pixel_coords[..., 0] = pixel_coords[..., 0] / max(sdf.shape[0] - 1, 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / (sdf.shape[1] - 1)
    pixel_coords[..., 2] = pixel_coords[..., 2] / (sdf.shape[2] - 1)
    pixel_coords = np.squeeze(pixel_coords)
    
    # normalize from [0, 1] to [-1, 1]
    pixel_coords -= 0.5
    pixel_coords *= 2.0
    
    # flatten all but the last dimension
    pixel_coords = pixel_coords.reshape(-1, pixel_coords.shape[-1])

    # combine grid & corresponding SDF values
    samples = torch.zeros(sdf.shape[0]*sdf.shape[1]*sdf.shape[2], 4)

    sdf = sdf.flatten()
    samples[:, 0:3] = torch.from_numpy(pixel_coords).float()
    samples[:, 3] = torch.from_numpy(sdf).float()
    
    # balance the data, i.e.,
    # randomly select the same amount of positive and negative SDF values
    half = int(subsample / 2)
    
    index = np.where(sdf > 0.)
    pos_tensor = samples[index,:]
    index = np.where(sdf <= 0.)
    neg_tensor = samples[index,:]   
    #print('negative values:', neg_tensor.shape[1], 'positive values:', pos_tensor.shape[1])

    random_pos = (torch.rand(half).cpu() * pos_tensor.shape[1]).long()
    random_neg = (torch.rand(half).cpu() * neg_tensor.shape[1]).long()

    sample_pos = torch.index_select(pos_tensor, 1, random_pos)
    sample_neg = torch.index_select(neg_tensor, 1, random_neg)
    
    samples = torch.cat([sample_pos, sample_neg], 1).squeeze().float()

    return samples


def read_params(lines):
    params = []
    for line in lines:
        line = line.strip()[1:-2]
        param = np.fromstring(line, dtype=float, sep=',')
        params.append(param)
    return params


def get_rotate_matrix(rotation_angle1):
    cosval = np.cos(rotation_angle1)
    sinval = np.sin(rotation_angle1)

    rotation_matrix_x = np.array([[1, 0, 0, 0],
                                  [0, cosval, -sinval, 0],
                                  [0, sinval, cosval, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_y = np.array([[cosval, 0, sinval, 0],
                                  [0, 1, 0, 0],
                                  [-sinval, 0, cosval, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_z = np.array([[cosval, -sinval, 0, 0],
                                  [sinval, cosval, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    scale_y_neg = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    neg = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    return np.linalg.multi_dot([neg, rotation_matrix_z, rotation_matrix_z, scale_y_neg, rotation_matrix_x])

rot90y = np.array([[0, 0, -1],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=np.float32)

def getBlenderProj(az, el, distance_ratio, roll = 0, focal_length=35, img_w=137, img_h=137):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    F_MM = focal_length  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.
    CAM_MAX_DIST = 1.75
    CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                      [1.0, -4.371138828673793e-08, -0.0],
                      [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    # Calculate intrinsic matrix.
    scale = RESOLUTION_PCT / 100
    # print('scale', scale)
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    # print('f_u', f_u, 'f_v', f_v)
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    # print('distance', distance_ratio * CAM_MAX_DIST)
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))
    # finally, consider roll
    cr = np.cos(np.radians(roll))
    sr = np.sin(np.radians(roll))
    R_z = np.matrix(((cr, -sr, 0),
                  (sr, cr, 0),
                  (0, 0, 1)))
    return K, R_z@RT


def get_camera_matrices(metadata_filename, id):
    # Adaptation of Code/Utils from DISN
    with open(metadata_filename, 'r') as f:
        lines = f.read().splitlines()
        param_lst = read_params(lines)
        rot_mat = get_rotate_matrix(-np.pi / 2)
        az, el, distance_ratio = param_lst[id][0], param_lst[id][1], param_lst[id][3]
        intrinsic, RT = getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224)
        extrinsic = np.linalg.multi_dot([RT, rot_mat])
    intrinsic = torch.tensor(intrinsic).float()
    extrinsic = torch.tensor(extrinsic).float()

    return intrinsic, extrinsic

def get_projection(az, el, distance, focal_length=35, img_w=256, img_h=256, sensor_size_mm = 32.):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""

    # Calculate intrinsic matrix.
    f_u = focal_length * img_w  / sensor_size_mm
    f_v = focal_length * img_h  / sensor_size_mm
    u_0 = img_w / 2
    v_0 = img_h / 2
    K = np.matrix(((f_u, 0, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    sa = np.sin(np.radians(az))
    ca = np.cos(np.radians(az))
    R_azimuth = np.transpose(np.matrix(((ca, 0, sa),
                                          (0, 1, 0),
                                          (-sa, 0, ca))))
    se = np.sin(np.radians(el))
    ce = np.cos(np.radians(el))
    R_elevation = np.transpose(np.matrix(((1, 0, 0),
                                          (0, ce, -se),
                                          (0, se, ce))))
    # fix up camera
    se = np.sin(np.radians(180))
    ce = np.cos(np.radians(180))
    R_cam = np.transpose(np.matrix(((ce, -se, 0),
                                          (se, ce, 0),
                                          (0, 0, 1))))
    T_world2cam = np.transpose(np.matrix((0,
                                           0,
                                           distance)))
    RT = np.hstack((R_cam@R_elevation@R_azimuth, T_world2cam))

    return K, RT

def unpack_images(filename):

    image = imageio.imread(filename).astype(float)/255.0
    return torch.tensor(image).float().permute(2,0,1)
