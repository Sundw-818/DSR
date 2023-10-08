import json
import os
from tqdm import tqdm
from lib.metrics import *
import argparse

class MeshEvaluator():
    def __init__(self, path_gt, path_pr, scale, translate, n_points=100000):
        self.path_gt = path_gt
        self.path_pr = path_pr
        self.scale = scale
        self.translate = translate
        self.n_points = n_points
        
        self.mesh_gt = trimesh.load(path_gt)
        self.mesh_pr = trimesh.load(path_pr)
        self.mesh_gt.vertices = ((self.mesh_gt.vertices - self.translate) / self.scale) * 2.0 - 1.0
        self.mesh_pr.vertices = ((self.mesh_pr.vertices - self.translate) / self.scale) * 2.0 - 1.0
    
    def eval(self):
        iou = compute_iou(self.mesh_gt, self.mesh_pr)
        chamfer_dist = compute_trimesh_chamfer(self.mesh_gt, self.mesh_pr)
        normal_consistency = compute_normal_consistency(self.mesh_gt, self.mesh_pr)
        
        out_dict = {
            "iou": iou,
            "chamfer_dist": chamfer_dist,
            "normal_consistency": normal_consistency
        }
        return out_dict


if __name__ == '__main__':
    print('Evaluating...')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', '-p', type=str, default=None)
    parser.add_argument('--gt', '-g', type=str, default=None)
    parser.add_argument('--out', '-o', type=str, default=None)
    parser.add_argument('--scale', '-s', type=float, default=2.0)
    parser.add_argument('--translate', '-t', type=float, default=-1.0)
    
    args = parser.parse_args()


    pred_dir = args.pred 
    gt_dir = args.gt
    s = args.scale
    t = args.translate
    out = args.out


    pred_list = sorted(os.listdir(pred_dir))
    print(f'Total {len(pred_list)} objs')
    print('Begin evaluation...')

    # metric_file = open('../exp/GPCR_stepmore/GPCR_stepmore/2022_11_29_09_14_26/metrics_result_10792_trj_81.txt', 'a')
    metric_file = open(out, 'a')
    metric_file.writelines('file\tiou\tchamifer_dist\tnormal_consistency\n')
    for i in tqdm(range(len(pred_list))):
    # for i in tqdm(range(5)):
        path_gt = os.path.join(gt_dir, f'traj_{i}.obj')
        # path_gt = os.path.join(gt_dir, '%05d.obj' % i)
        path_pr = os.path.join(pred_dir, pred_list[i])
        mesh_evaluator = MeshEvaluator(path_gt, path_pr, s, t)
        metrics = mesh_evaluator.eval()
        line = '{}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(pred_list[i], metrics['iou'], metrics['chamfer_dist'], metrics['normal_consistency'])
        metric_file.writelines(line)
        
    metric_file.close()  