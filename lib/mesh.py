#!/usr/bin/env python3

# import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch

from lib.utils import *

def convert_sdf_to_mesh(sdf_array, voxel_origin, N=256, offset=None, scale=None):
    voxel_size = 2.0 / (N - 1)
    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        sdf_array, level=0.0, spacing=[voxel_size] * 3
    )
    
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset 
    return mesh_points, faces


def write_verts_faces_to_file(verts, faces, filename):
    if filename.split('/')[:-1]:
        filedir = os.path.join(*filename.split('/')[:-1])
        os.makedirs(filedir, exist_ok=True)
    
    if filename[-3:] == 'obj':
        with open(filename, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in faces+1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    elif filename[-3:] == 'ply':
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]

        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(verts[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        ply_data.write(filename)



def create_SDF(
    decoder, latent_vec, time, N=np.array([128,128,128]), max_batch=64**3, offset=None, scale=None):

    decoder.eval()
    
    # pre-allocate array
    #samples = torch.zeros(N[0]*N[1]*N[2], 4)
    samples = torch.zeros(N[0]*N[1]*N[2], 5)  # (x, y, z, val) -> (x, y, z, t, val)
    
    # prepare coordinate grid
    pixel_coords = np.stack(np.mgrid[:N[0], :N[1], :N[2]], axis=-1)[None, ...].astype(np.float32)
    pixel_coords[..., 0] = pixel_coords[..., 0] / max(N[0] - 1, 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / (N[1] - 1)
    pixel_coords[..., 2] = pixel_coords[..., 2] / (N[2] - 1)
    pixel_coords = np.squeeze(pixel_coords)
    
    # normalize from [0, 1] to [-1, 1]
    pixel_coords -= 0.5
    pixel_coords *= 2.  
    
    # flatten all but the last dimension
    pixel_coords = pixel_coords.reshape(-1, pixel_coords.shape[-1])
    
    # paste grid into the samples array
    samples[:, 0:3] = torch.from_numpy(pixel_coords).float()
    samples[:, 3] = time.expand(samples.size(0))
    
    samples.requires_grad = False

    num_samples = N[0]*N[1]*N[2] 

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:4].cuda()
        samples[head : min(head + max_batch, num_samples), 4] = (
            decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch
    
    #print('lat. vec. preview:', latent_vec[:5].detach().cpu().numpy().astype(np.float), 
    #    "...")
    
    # reshape SDF
    sdf_values = samples[:, 4]
    print('min:', min(sdf_values.detach().cpu().numpy().astype(np.float)), 'max:', max(sdf_values.detach().cpu().numpy().astype(np.float)),'\n')
    sdf_values = sdf_values.reshape(N[0], N[1], N[2])

    # save SDF to matlab file
    #sio.savemat(filename + '.mat', {'sdf_norm':
    #            sdf_values.detach().cpu().numpy().astype(np.float)})
                
    # save SDF to numpy file            
    #np.save(filename + '.npy', sdf_values)

    # return
    return sdf_values.detach().cpu().numpy().astype(np.float)


def create_mesh_optim_fast(
    samples, indices, decoder, latent_vec, N=256, max_batch=32 ** 3, offset=None, scale=None, fourier = False, taylor = False
):

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    num_samples = indices.shape[0]

    with torch.no_grad():

        head = 0
        while head < num_samples:
            sample_subset = samples[indices[head : min(head + max_batch, num_samples)], 0:3].reshape(-1, 3).cuda()
            samples[indices[head : min(head + max_batch, num_samples)], 3] = (
                decode_sdf(decoder, latent_vec, sample_subset)
                .squeeze(1)
                .detach()
                .cpu()
            )
            head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)

    verts, faces = convert_sdf_samples_to_mesh(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )

    # fetch bins that are activated
    k = ((verts[:, 2] -  voxel_origin[2])/voxel_size).astype(int)
    j = ((verts[:, 1] -  voxel_origin[1])/voxel_size).astype(int)
    i = ((verts[:, 0] -  voxel_origin[0])/voxel_size).astype(int)
    # find points around
    next_samples = i*N*N + j*N + k
    next_samples_i_plus = np.minimum(i+1,N-1)*N*N + j*N + k
    next_samples_j_plus = i*N*N + np.minimum(j+1,N-1)*N + k
    next_samples_k_plus = i*N*N + j*N + np.minimum(k+1,N-1)
    next_samples_i_minus = np.maximum(i-1,0)*N*N + j*N + k
    next_samples_j_minus = i*N*N + np.maximum(j-1,0)*N + k
    next_samples_k_minus = i*N*N + j*N + np.maximum(k-1,0)
    next_indices = np.concatenate((next_samples,next_samples_i_plus, next_samples_j_plus,next_samples_k_plus,next_samples_i_minus,next_samples_j_minus, next_samples_k_minus))

    return verts, faces, samples, next_indices
