from __future__ import print_function
import os
import cv2
import sys
import tqdm
from typing import Sequence
import h5py
import time
import datetime
from h5py._hl.selections import select
import numpy as np
from tabulate import tabulate
import argparse
import numpy

import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from config import config
# from utils.file_process import Logger, read_json, write_json, save_checkpoint
# from networks.DSN import DSN
# from networks.RL import compute_reward
# from utils import vsum_tool

torch.manual_seed(config.SEED)
os.environ["CUDA_VISIBLE_DEVCIES"] = config.GPU
use_gpu = torch.cuda.is_available()
if config.USE_CPU: use_gpu = False

parser = argparse.ArgumentParser("Training")
parser.add_argument('-d', '--dataset', type=str, default='', help="path to dataset file h5")
args = parser.parse_args()

def compute_reward(dataset):
   for key_idx, key in enumerate(dataset):
    seq = dataset[key]['features'][...]
    print(seq)
    print(type(seq))
    print("testestfdtsadfgsayidaasgx")
    seq_l =len(seq)
    # con_actions = torch.from_numpy(actions)
    print(seq.shape)
    seq = torch.from_numpy(seq).unsqueeze(0)
    

    x1 = seq.resize_(1,seq_l,1024) # resized to 2048 to 1024
    x1 = x1.resize_(seq_l,1024)    #3 dimentional to 2 dimentional 
    print(x1.shape)
    print(x1)
    seq_len = seq.shape[0]
    print(seq_len)
    
    seq_list = []
    for i in range(0, seq_len):
        seq_list.append(i)

    print('elements in sequence are ')
    print(seq_list)
    
    print("***************************")
    
    cps = dataset[key]['change_points'][...]
    cps_int_array = []
    for x in cps:
        #   import pdb;pdb.set_trace()
          med = np.mean(x)
          int_array = med.astype(int)
          cps_int_array.append(int_array)
    print("median of change_points is")
    print(cps_int_array)
    print(cps)

    
    find_median = cps_int_array
    median_cps= []
    for i in find_median:
        div_median= i/15  # Took only one frame from the 15 frames
        real_num = int(div_median)
        store_val = real_num
        median_cps.append(store_val) 
    print("median from the group ")
    print(median_cps)    # find median of a changepoint  are stored 
    print("******************************")

def compute_diversity(con_seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
   
    _seq = con_seq.detach()
    # print(type(_seq))
    # import pdb;pdb.set_trace()
    _actions = actions.detach() 
    pick_idxs = _actions.squeeze().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    # print(num_picks)
    
    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        # import pdb;pdb.set_trace()
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
        # dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        # if ignore_far_sim:
        #     # ignore temporally distant similarity
        #     pick_mat = pick_idxs.expand(num_picks, num_picks)
        #     temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
        #     dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        #     # import pdb;pdb.set_trace()
        #     reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.)) # diversity reward [Eq.3]   
        #     print(reward_div)
        print("Reward code ends here ")
    return dissim_mat 


def diversity_score(i_determinant,dissim_mat,dis_mat_rep):
    #multiply diversity with determinant value
    det_div = torch.matmul(i_determinant,dissim_mat)
    print(det_div)
    
    #multiply representation with determinant value
    det_rep = torch.matmul(i_determinant,dis_mat_rep)
    print(det_rep)

    #adding diversity and representation
    mul_reward = torch.add(det_div,det_rep)/2
    print(mul_reward)
    
    #find top n/2 values in the tensor 
    max_index = torch.topk(mul_reward, len(mul_reward/2))
    print(max_index)


    frame = sorted(range(len(mul_reward)), key=lambda i: mul_reward[i], reverse=True)[:4]
    print(frame)
    con_frame = torch.Tensor(frame)

    print(type(con_frame))

    a = [15]
    b = torch.FloatTensor(a)

    exact_frame = torch.multiply(con_frame ,b)
    con_frame = exact_frame.long()
    print(con_frame)

    print("det & dissim index vales printed here ")


def main():
    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    print(num_videos)
    print(dataset.keys())
    compute_rewards = compute_reward(dataset)
    
    for key_idx, key in enumerate(dataset):
       seq = dataset[key]['features'][...]
       print(seq)
       print(type(seq))
       # seq_l =len(seq)
       con_seq = torch.tensor(seq)
       print(type(con_seq))

    for key_idx, key in enumerate(dataset):
     seq = dataset[key]['features'][...]
     print(seq)
     print(type(seq))
     print("//////////////////////////////////////")
     seq_l =len(seq)
    # con_actions = torch.from_numpy(actions)
    print(seq.shape)
    seq = torch.from_numpy(seq).unsqueeze(0)

    x1 = seq.resize_(1,seq_l,1024) # resized to 2048 to 1024
    x1 = x1.resize_(seq_l,1024)    #3 dimentional to 2 dimentional 
    print(x1.shape)
    print(x1)
    seq_len = seq.shape[0]
    print(seq_len)
    
    seq_list = []
    for i in range(0, seq_len):
        seq_list.append(i)

    print('elements in sequence are ')
    print(seq_list)
    
    print("***************************")
    
    cps = dataset[key]['change_points'][...]
    cps_int_array = []
    for x in cps:
        #   import pdb;pdb.set_trace()
          med = np.mean(x)
          int_array = med.astype(int)
          cps_int_array.append(int_array)
    print("median of change_points is")
    print(cps_int_array)

    
    find_median = cps_int_array
    median_cps= []
    for i in find_median:
        div_median= i/15  # Took only one frame from the 15 frames
        real_num = int(div_median)
        store_val = real_num
        median_cps.append(store_val) 
    print("median from the group ")
    print(median_cps)    # find median of a changepoint  are stored 

    print("action process started")
    
    
    con_dim = []

    # find actions from the seq_len and median_cps
    for i in seq_list:
        
        if i not in median_cps: 
          print(0)
          con_dim.append(0)
        else:
          print(1)
          con_dim.append(1)
    
    print(con_dim)
    # print(type(_actions))
    a = np.array(con_dim).reshape(1,seq_l,1)
    actions = torch.tensor(a)
    print(type(actions ))
    print("action process finished")
            
    diversity = compute_diversity(con_seq,actions,ignore_far_sim=True, temp_dist_thre=20, use_gpu=False)


    print("action process started")
    
    
    con_dim = []

    # find actions from the seq_len and median_cps
    for i in seq_list:
        
        if i not in median_cps: 
          print(0)
          con_dim.append(0)
        else:
          print(1)
          con_dim.append(1)
    
    print(con_dim)
    # print(type(_actions))
    actions = np.array(con_dim).reshape(1,seq_l,1)
    print(actions)
    _actions  = torch.tensor(actions)
    print(type(_actions))

    print("action process finished")
        

    i_deter = []
    
    for i in range(seq_len):   #iterate i and j values #median_cps
        # for j in range(seq_len):
            print(i)
            # i_seq= dataset[key]['features'][i]
            i_pass = x1[i]
            print(i_pass.shape)
            # j_pass = x1[j]
            i_pass = i_pass.unsqueeze(0)
            print(i_pass.size())
            i_det = torch.matmul(i_pass, i_pass.t())
            print(i_det)
            i_deter.append(i_det)
            # i_fdet = torch.det(i_detr)
    print(i_deter)
    # b = torch.FloatTensor(a)
    i_determinant = torch.FloatTensor(i_deter)
    # con_detr = torch.unique(torch.tensor(i_determinant))
    # print(con_detr.shape)
    print(i_determinant.shape)

    num_frames = dataset[key]['n_frames'][...]
    nfps = dataset[key]['n_frame_per_seg'][...].tolist()
    picks  = dataset[key]['picks'][...]
    print(type(picks))
    for i in range(len(picks)):
        print(i)
    print("Reward code starts from here ")
    print("******************************")
    print("******************************")

    _seq = con_seq.detach()
    # print(type(_seq))
    # import pdb;pdb.set_trace()
    _actions = _actions.detach() 
    pick_idxs = _actions.squeeze()
    num_picks = len(pick_idxs) #if pick_idxs.ndimension() > 0 else 1
    # print(num_picks)
    
    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        # import pdb;pdb.set_trace()
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]

    # compute representativeness reward
    # import pdb;pdb.set_trace()
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    
    dist_mat = dist_mat + dist_mat.t()
    dis_mat_rep = dist_mat.addmm_(1, -2, _seq, _seq.t())
    print("Reward code ends here ")
    
    find_diversity = diversity_score(i_determinant,dissim_mat,dis_mat_rep)
    
    
    
    # frame_2video = frm2video(dataset,summary,vid_writer)

if __name__ == '__main__':
    main()
