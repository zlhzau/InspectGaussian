#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,detect_image,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        # self.depth_reliable = False
        self.detect_image = None
        self.depth_reliable = True

        # if invdepthmap is not None:
        #     self.depth_mask = torch.ones_like(self.alpha_mask)
        #     self.invdepthmap = cv2.resize(invdepthmap, resolution)
        #     self.invdepthmap[self.invdepthmap < 0] = 0
        #     self.depth_reliable = True

        #     if depth_params is not None:
        #         if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
        #             self.depth_reliable = False
        #             self.depth_mask *= 0
                
        #         if depth_params["scale"] > 0:
        #             self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

        #     if self.invdepthmap.ndim != 2:
        #         self.invdepthmap = self.invdepthmap[..., 0]
        #     self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)
        
        if invdepthmap is not None :
            dis_mask = np.zeros_like(invdepthmap)
            dis_mask[invdepthmap > 10] = 1

            self.depth_mask = torch.from_numpy(dis_mask).to(self.data_device)
            
            invdepthmapScaled = (float(1e3) / (invdepthmap+1e-12))*dis_mask
            invdepthmapScaled = cv2.resize(invdepthmapScaled, resolution)
            invdepthmapScaled = invdepthmapScaled
            # print(np.max(invdepthmapScaled),np.min(invdepthmapScaled))
            # import pdb
            # pdb.set_trace()
            if invdepthmapScaled.ndim != 2:
                invdepthmapScaled = invdepthmapScaled[..., 0]

            self.invdepthmap = torch.from_numpy(invdepthmapScaled[None]).to(self.data_device)
            
            # import pdb
            # pdb.set_trace()
            self.depth_reliable = True

        if detect_image is not None:
                    if detect_image.ndim == 3 and detect_image.shape[2] == 3:
                        self.detect_image=detect_image[:,:,0:1]
                    if detect_image.ndim == 2:
                        self.detect_image = detect_image
                    

                    # 检查像素值范围，如果是0-255，则归一化到0-1
                    if self.detect_image.max() <= 1.0:
                        # 已经是0-1范围，不需要转换
                        self.detect_image = torch.from_numpy(self.detect_image).to(self.data_device)
                    else:
                        # 假设像素值范围是0-255，归一化到0-1
                        self.detect_image = torch.from_numpy(self.detect_image / 255.0).to(self.data_device)
                # print("6666",self.detect_image)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

