import trimesh
import numpy as np
import os
import mdtraj as md
from pymol import cmd

traj = md.load('datasets/abeta.pdb', top='datasets/abeta_0.pdb')

gt_path = 'datasets/abeta_30_gt'
train_path = 'datasets/abeta_30/obj'
temp_path = 'datasets/temp'
os.makedirs(gt_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

for i in range(traj.n_frames):
    pdb_name = os.path.join(temp_path, 'temp.pdb')
    obj_name = f'traj_{i}.obj'
    traj[i].save_pdb(pdb_name)
    cmd.load(pdb_name)
    cmd.show_as('surface')
    cmd.save(os.path.join(gt_path, obj_name))
    if i % 5 == 0:
        cmd.save(os.path.join(train_path, obj_name))
    cmd.delete('all')
os.system(f'rm -r {temp_path}')