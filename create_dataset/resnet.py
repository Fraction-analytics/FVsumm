'''
Extract feature vectors from video frames.
These features come from the Pool5 layers of a ResNet deep
neural network, pre-trained on ImageNet. The algorithm captures
frames directly from video, there is not need for prior frame extraction.

Copyright (C) 2019 Alexandros I. Metsai
alexmetsai@gmail.com

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
import h5py
import os
import sys
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np

class Rescale(object):
    """
    Rescale an image to the given size.

    Args:
        output_size : Can be int or tuple. In the case a single integer
        is given, PIL will resize the smallest of the original
        dimensions to this value, and resize the largest dimention 
        such as to keep the same aspect ratio.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


class ResNetPool5(nn.Module):
    
    def __init__(self, DNN='resnet101'):
        """
        Load pretrained ResNet weights on ImageNet. Return the Pool5
        features as output when called.
        
        Args:
            DNN (string): The DNN architecture. Choose from resnet101, 
            resnet50 or resnet152. ResNet50 and ResNet152 are not yet 
            in the release version of TorchVision, you will have to 
            build from source for these nets to work, or wait for the
            newer versions.
        """
        super().__init__()
        
        if DNN == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif DNN == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif DNN == "resnet152":
            resnet = models.resnet152(pretrained=True)
        else:
            print("Error. Network " + DNN + " not supported.")
            exit(1)
        resnet.float()
        
        # Use GPU is possible
        if torch.cuda.is_available():
            resnet.cuda()
        resnet.eval()
        
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]
        
    def forward(self, x):
        res5c = self.conv5(x)
        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.size(0), -1)
        pool5 = pool5.cpu().data.numpy().flatten()
        return pool5

# Check torchvision docs about these normalization values applied on ResNet.
# Since it was applied on training data, we should do so as well.
# Note that this transform does not cast the data first to [0,1].
# Should this action be performed? Or does this normalization make
# it uneccessary?
data_normalization = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

if __name__=='__main__':
    
    model = ResNetPool5()
    video_folder = "videos/"
    
    # Extract features for all the videos in the list.
    
    for video_idx, file in enumerate(os.listdir(video_folder)):
        save_path = file +".h5"
        h5_file = h5py.File(save_path, 'w')
        h5_file.create_group('video_{}'.format(video_idx+1))
        
        # Empty list to append tensors to.
        features_list = []
        
        if file.endswith(".mp4"):
            
            print("Processing " + file)
            video_capture = cv2.VideoCapture(video_folder + file)
            success, image = video_capture.read()
            i = 1
            
            if not success :
                print("Error while reading video file.")
                sys.exit(-1)
            while success:
                # print(i)
                video_feat = None
                # The video's frames are captured with cv2. OpenCV treats
                # images as numpy arrays, but since PyTorch works with PIL
                # images, we convert them as such.
                image = Image.fromarray(image)
                
                # Transform the data to ResNet's desired characteristics.
                image = data_normalization(image)
                
                # Add the extra "batch" dimension.
                image = image.unsqueeze(0)
                
                # Move the data to GPU and do a forward pass.
                if torch.cuda.is_available():
                    pool5 = model.forward(image.cuda())
                else:
                    pool5 = model.forward(image)
                
                # Detach the tensor from the model and store it to CPU memory.
                # temp = pool5.clone()
                temp = pool5
                # temp = temp.detach()
                if torch.cuda.is_available():
                    temp.cpu()
                
                # Append the tensor to the list.
                features_list.append(temp)
                if video_feat is None:
                    video_feat = features_list
                else:
                    video_feat = np.vstack((video_feat, features_list))
                
                # Capture the next frame.
                success, image = video_capture.read()
                i+=1
                print(i)
            # Save the list of features to pickle file.
            # filename = video_folder + file[:-4] + "_features.h5"
            # torch.save(features_list, filename)
            # print("total number of extracted feature vectors for ", file, ":", i)

            # for name in video_names:
            # save_path = file +".h5"
            # h5_file = h5py.File(save_path, 'w')
            
            h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat)
            h5_file.close()


            # type(features_list)
            # adict=dict(video1=features_list)
            # for k,v in adict.items():
            #   f.create_dataset(k,data=v)

            # f.create_dataset('video1' + '/features', data=features_list)
            
            

# TODO:
# Need to add option to save the features on a single pickle
# file instead of a separate for each video.









# ### To form your own dataset.. for video summ.. https://github.com/KaiyangZhou/vsumm-reinforce/issues/1
# import h5py
# h5_file_name = 'vsumm_dataset.h5'
# f = h5py.File(h5_file_name, 'w')

# # video_names is a list of strings containing the 
# # name of a video, e.g. 'video_1', 'video_2'
# for name in video_names:
#     f.create_dataset(name + '/features', data=data_of_name)
#     f.create_dataset(name + '/gtscore', data=data_of_name)
#     f.create_dataset(name + '/user_summary', data=data_of_name)
#     f.create_dataset(name + '/change_points', data=data_of_name)
#     f.create_dataset(name + '/n_frame_per_seg', data=data_of_name)
#     f.create_dataset(name + '/n_frames', data=data_of_name)
#     f.create_dataset(name + '/picks', data=data_of_name)
#     f.create_dataset(name + '/n_steps', data=data_of_name)
#     f.create_dataset(name + '/gtsummary', data=data_of_name)
#     f.create_dataset(name + '/video_name', data=data_of_name)

# f.close()