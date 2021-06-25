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
from utils.generate_dataset import Generate_Dataset
import os.path as osp
import multiprocessing
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from config import config

torch.manual_seed(config.SEED)
os.environ["CUDA_VISIBLE_DEVCIES"] = config.GPU
use_gpu = torch.cuda.is_available()
if config.USE_CPU: use_gpu = False

parser = argparse.ArgumentParser("Training")
parser.add_argument('-d', '--dataset1', type=str, default='', help="path to dataset file h5")
parser.add_argument('-f', '--frm-dir', type=str, default="../data/video1",help="path to frame directory")
parser.add_argument('--save-dir', type=str, default='log', help="directory to save")
parser.add_argument('--save-name', type=str, default='summary.mp4', help="video name to save (ends with .mp4)")
parser.add_argument('--fps', type=int, default=30, help="frames per second")
parser.add_argument('--width', type=int, default=640, help="frame width")
parser.add_argument('--height', type=int, default=480, help="frame height")

parser.add_argument('-da', '--dataset2', type=str, default='', help="path to dataset file h5")
parser.add_argument('-fa', '--frm-dira', type=str, default="../data/video2",help="path to frame directory")
parser.add_argument('--save-namea', type=str, default='summary2.mp4', help="video name to save (ends with .mp4)")
args = parser.parse_args()

def compute_reward(dataset1):
    for key_idx, key in enumerate(dataset1):
        # import pdb;pdb.set_trace()
        seq = dataset1[key]['features'][...]
        print(seq.shape)
        print(seq)
        print(type(seq))
        con_seq = torch.tensor(seq)
        print(type(con_seq))
        seq_l =len(seq)
        seq = torch.from_numpy(seq).unsqueeze(0)

        x1 = seq.resize_(1,seq_l,1024)      #... resized to 2048 to 1024
        x1 = x1.resize_(seq_l,1024)         #... 3 dimentional to 2 dimentional 
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
    
    cps = dataset1[key]['change_points'][...]
    cps_int_array = []
    for x in cps:
        med = np.mean(x)
        int_array = med.astype(int)
        cps_int_array.append(int_array)
    print("median of change_points is")
    print(cps_int_array)
    
    find_median = cps_int_array
    median_cps= []
    for i in find_median:
        div_median= i/15                  #... Took only one frame from the 15 frames
        real_num = int(div_median)
        store_val = real_num
        median_cps.append(store_val) 
    print("median from the group ")
    print(median_cps)                    #... find median of a changepoint  are stored 

    print("action process started")

                                         #... find actions from the seq_len and median_cps
    con_dim = []
    for i in seq_list:
        
        if i not in median_cps: 
            print(0)
            con_dim.append(0)
        else:
            print(1)
            con_dim.append(1)
    
    print(con_dim)
    a = np.array(con_dim).reshape(1,(seq_l),1)
    actions = torch.tensor(a)
    print(type(actions ))
    print("action process finished")


    actions = np.array(con_dim).reshape(1,seq_l,1)
    # print(actions)
    _actions  = torch.tensor(actions)
    print(type(_actions))
    print("action process finished")
    
    i_deter = []
    for i in range(seq_len):            #iterate i and j values #median_cps
        #   for j in median_cps:
            # print(i)
            # i_seq= dataset[key]['features'][i]
            i_pass = x1[i]
            # print(i_pass.shape)
            # j_pass = x1[j]
            i_pass = i_pass.unsqueeze(0)
            # print(i_pass.size())
            i_det = torch.matmul(i_pass, i_pass.t())
            # print(i_det)
            i_deter.append(i_det)
            # i_fdet = torch.det(i_detr)
    print(i_deter)
    # b = torch.FloatTensor(a)
    i_determinant = torch.FloatTensor(i_deter)
    # con_detr = torch.unique(torch.tensor(i_determinant))
    # print(con_detr.shape)
    print(i_determinant.shape)

    num_frames = dataset1[key]['n_frames'][...]
    picks  = dataset1[key]['picks'][...]
    print(type(picks))
    for i in range(len(picks)):
        print(i)
    
    print("Reward code starts from here ")
    # print(i_detr)
    # print(i_detr.shape)


    print("******************************")
    # return cond_detr
    

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
    
    print("det & dissim index vales printed here ")

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

    index = int(len(seq) / 2)
    print(index)
    print("Here sequence length divided by two and converted as interger")

    frame = sorted(range(len(mul_reward)), key=lambda i: mul_reward[i], reverse=True)[:index]
    print(frame)
    con_frame = torch.Tensor(frame)

    # print(type(con_frame))

    a = [15]
    b = torch.FloatTensor(a)

    exact_frame = torch.multiply(con_frame ,b)
    convert_frame = exact_frame.long()
    print(convert_frame)


    for i in range(seq_len):
    
        org_seq = (i*15)
        org_seq = torch.sub(org_seq,-1)

    frame_num_binary= []
    for i in range(org_seq):
        if i not in convert_frame: 
        #   print(0)
            frame_num_binary.append(0)
        else:
        #   print(1)
            frame_num_binary.append(1)
    print(frame_num_binary)

    if not osp.exists(args.save_dir):
        # import pdb;pdb.set_trace()
        os.mkdir(args.save_dir)
    vid_writer = cv2.VideoWriter(osp.join(args.save_dir, args.save_name),     
        cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(args.width, args.height))
    h5_res = h5py.File(args.dataset1, 'r')
    # key = h5_res.keys()[args.idx]
    summary = frame_num_binary
    h5_res.close()
    frm2video(args.frm_dir, summary, vid_writer)
    vid_writer.release()


def frm2video(frm_dir,summary, vid_writer):
    # import pdb;pdb.set_trace()
        # print(frm_dir)
    for idx, val in enumerate(summary):
        try:
            if val == 1:
                # here frame name starts with '000001.jpg'
                # change according to your need
                frm_name = str(idx) + '.jpg' # zfill to zfill(6), delete zfill
                frm_path = osp.join(frm_dir, frm_name)
                print(frm_path)
                frm = cv2.imread(frm_path)
                print(frm)
                print("***********")
                frm = cv2.resize(frm, (args.width, args.height))
                vid_writer.write(frm)
        except:
            # pdb.set_trace()
            print()






def compute_rewardb(dataset2):
    for key_idx, key in enumerate(dataset2):
        # import pdb;pdb.set_trace()
        seq = dataset2[key]['features'][...]
        print(seq.shape)
        print(seq.shape)
        print(seq)
        print(type(seq))
        con_seq = torch.tensor(seq)
        print(type(con_seq))
        seq_l =len(seq)
        seq = torch.from_numpy(seq).unsqueeze(0)

        x1 = seq.resize_(1,seq_l,1024)      #... resized to 2048 to 1024
        x1 = x1.resize_(seq_l,1024)         #... 3 dimentional to 2 dimentional 
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
    
    cps = dataset2[key]['change_points'][...]
    cps_int_array = []
    for x in cps:
        med = np.mean(x)
        int_array = med.astype(int)
        cps_int_array.append(int_array)
    print("median of change_points is")
    print(cps_int_array)
    
    find_median = cps_int_array
    median_cps= []
    for i in find_median:
        div_median= i/15                    #... Took only one frame from the 15 frames
        real_num = int(div_median)
        store_val = real_num
        median_cps.append(store_val) 
    print("median from the group ")
    print(median_cps)                       #... find median of a changepoint  are stored 

    print("action process started")

                                            #... find actions from the seq_len and median_cps
    con_dim = []
    for i in seq_list:
        
        if i not in median_cps: 
            print(0)
            con_dim.append(0)
        else:
            print(1)
            con_dim.append(1)
    
    print(con_dim)
    a = np.array(con_dim).reshape(1,(seq_l),1)
    actions = torch.tensor(a)
    print(type(actions ))
    print("action process finished")


    actions = np.array(con_dim).reshape(1,seq_l,1)
    # print(actions)
    _actions  = torch.tensor(actions)
    print(type(_actions))
    print("action process finished")
    
    i_deter = []
    for i in range(seq_len):   #iterate i and j values #median_cps
        #   for j in median_cps:
            # print(i)
            # i_seq= dataset[key]['features'][i]
            i_pass = x1[i]
            print(i_pass.shape)
            # j_pass = x1[j]
            i_pass = i_pass.unsqueeze(0)
            print(i_pass.size())
            i_det = torch.matmul(i_pass, i_pass.t())
            # print(i_det)
            i_deter.append(i_det)
            # i_fdet = torch.det(i_detr)
    print(i_deter)
    # b = torch.FloatTensor(a)
    i_determinant = torch.FloatTensor(i_deter)
    # con_detr = torch.unique(torch.tensor(i_determinant))
    # print(con_detr.shape)
    print(i_determinant.shape)

    num_frames = dataset2[key]['n_frames'][...]
    # nfps = dataset[key]['n_frame_per_seg'][...].tolist()
    picks  = dataset2[key]['picks'][...]
    # print(type(picks))
    # for i in range(len(picks)):
    #     print(i)
    
    print("Reward code starts from here ")
    # print(i_detr)
    # print(i_detr.shape)


    print("******************************")
    # return cond_detr
    

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
    
    print("det & dissim index vales printed here ")

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

    index = int(len(seq) / 2)
    print(index)
    print("Here sequence length divided by two and converted as interger")

    frame = sorted(range(len(mul_reward)), key=lambda i: mul_reward[i], reverse=True)[:index]
    print(frame)
    con_frame = torch.Tensor(frame)

    # print(type(con_frame))

    a = [15]
    b = torch.FloatTensor(a)

    exact_frame = torch.multiply(con_frame ,b)
    convert_frame = exact_frame.long()
    print(convert_frame)
    
    for i in range(seq_len):
        org_seq = i*15
        org_seq = torch.sub(org_seq,-1)

    frame_num_binary= []
    for i in range(org_seq-1):
        if i not in convert_frame: 
        #   print(0)
            frame_num_binary.append(0)
        else:
        #   print(1)
            frame_num_binary.append(1)
    print(frame_num_binary)

    if not osp.exists(args.save_dir):
        # import pdb;pdb.set_trace()
        os.mkdir(args.save_dir)
    vid_writer = cv2.VideoWriter(osp.join(args.save_dir, args.save_namea),     
        cv2.VideoWriter_fourcc(*'mp4v'),args.fps,
        (args.width, args.height))
    # (args.width, args.height), to (args.width, args.height)

    h5_res = h5py.File(args.dataset1, 'r')
    # key = h5_res.keys()[args.idx]
    summary = frame_num_binary
    h5_res.close()
    frm2video(args.frm_dir, summary, vid_writer)
    vid_writer.release()

def frm2video(frm_dir,summary, vid_writer):
    # import pdb;pdb.set_trace()
        # print(frm_dir)
    for idx, val in enumerate(summary):
        try:
            if val == 1:
                # here frame name starts with '000001.jpg'
                # change according to your need
                frm_name = str(idx) + '.jpg' # zfill to zfill(6), delete zfill
                frm_path = osp.join(frm_dir, frm_name)
                print(frm_path)
                frm = cv2.imread(frm_path)
                print(frm)
                print("***********")
                frm = cv2.resize(frm, (args.width, args.height))
                vid_writer.write(frm)
        except:
            # pdb.set_trace()
            print()

    print("Computer Reward Function Ended Here")



def main():
    dataset = h5py.File(args.dataset1, 'r')
    for key_idx , key in enumerate(dataset):
        print()
    num_videos = len(dataset.keys())
    print(num_videos)
    print(dataset.keys())
    compute_rewards = compute_reward(dataset)

    dataset = h5py.File(args.dataset2, 'r')
    for key_idx , key in enumerate(dataset):
        print()
    num_videos = len(dataset.keys())
    print(num_videos)
    print(dataset.keys())
    compute_rewarda = compute_rewardb(dataset)
    
    print("Computer Reward Function Ended Here")
    
    



    
if __name__ == '__main__':
    main()
