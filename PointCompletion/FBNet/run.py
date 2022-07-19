import os
import sys
import argparse
import numpy as np
import torch
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{0}".format(os.path.dirname(BASE_DIR)))
from vis_utils import *
from FBNet import Model
# import munch
# import yaml

def test():
    save_completion_path = "{0}/res/fine".format(BASE_DIR)
    save_input_path = "{0}/res/input".format(BASE_DIR)
    save_gt_path = "{0}/res/gt".format(BASE_DIR)
    os.makedirs(save_completion_path, exist_ok=True)
    os.makedirs(save_input_path, exist_ok=True)
    os.makedirs(save_gt_path, exist_ok=True)

    model_file = "{0}/model/mvp2k_network.pth".format(BASE_DIR)
    net = Model()
    net.eval()
    net.cuda()

    net.load_state_dict(torch.load(model_file)['net_state_dict'])

        
    for _, _, files in os.walk("{0}/data/partial_input".format(BASE_DIR)):
        for file in files:
            prefix = file.replace(".npy", "")
            input_file = "{0}/data/partial_input/{1}".format(BASE_DIR, file)
            gt_file = "{0}/data/gt/{1}".format(BASE_DIR, file)
            inputs = torch.from_numpy(np.load(input_file)).unsqueeze(0).transpose(2,1).contiguous()
            gt = torch.from_numpy(np.load(gt_file)).unsqueeze(0).transpose(2,1).contiguous()
            inputs = inputs.float().cuda()
            with torch.no_grad():
                fine = net(inputs)
                # print(fine.shape)
                pic = "{0}.png".format(prefix)
                plot_single_pcd(fine[0].cpu().numpy(), os.path.join(save_completion_path, pic))
                plot_single_pcd(gt.transpose(2,1)[0], os.path.join(save_gt_path, pic))
                plot_single_pcd(inputs.transpose(2,1)[0].cpu().numpy(), os.path.join(save_input_path, pic))
           

if __name__ == "__main__":
    test()




