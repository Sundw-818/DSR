import argparse
import os
import json

import trimesh
from pyhocon import ConfigFactory

from lib.metrics import *


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
    
    # os.environ["WANDB_MODE"] = "offline"

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/debug.conf')
    parser.add_argument('--path_gt', type=str, required=True, default=None)
    parser.add_argument('--path_pr', type=str, required=True, default=None)
    parser.add_argument('--path_out', type=str, default=None)
    parser.add_argument('--scale', type=float, default=None)
    parser.add_argument('--translate', type=float, default=None)

    args = parser.parse_args()
    
    if not args.scale and not args.translate:
    
        f = open(args.conf, 'r')
        conf_text = f.read()
        conf = ConfigFactory.parse_string(conf_text)
        
        scale = conf.get_float('dataset.s')
        translate = conf.get_float('dataset.t')
    else:
        scale = args.scale
        translate = args.translate
    
    if args.path_out:
        path_out = args.path_out
    else:
        path_out = './metrics.json'
    
    mesh_evaluator = MeshEvaluator(path_gt=args.path_gt,
                                 path_pr=args.path_pr,
                                 scale=scale,
                                 translate=translate)
    metrics = mesh_evaluator.eval()
    
    with open(path_out, 'w') as f:
        f.write(json.dumps(metrics, indent=4))
    
    