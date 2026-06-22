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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix2
from torch.nn import functional as F

class Camera(nn.Module):
    def __init__(self, colmap_id, quaternionOrR, T, FoVx, FoVy, intrinsics, image, gt_alpha_mask, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", fid=None, depth=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        # self.R = R
        # self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.intrinsics = torch.from_numpy(intrinsics).float().to('cuda')
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.fid = torch.Tensor(np.array([fid])).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.depth = torch.Tensor(depth).to(self.data_device) if depth is not None else None

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        register = lambda x, y: self.register_parameter(x, torch.nn.Parameter(y))
        
        # # Extrinsics
        self.oR=quaternionOrR
        self.oT=T

        self.quaternionOrR = torch.nn.Parameter(quaternionOrR)
        self.T = torch.nn.Parameter(T)
        
        _T = self.T.reshape(3, )
        _quaternion = self.matrix_to_quaternion(self.quaternionOrR.reshape(3, 3)).reshape(4, )

        register('quaternion', _quaternion)
        register('T', _T)

        # assert quaternionOrR.dtype == T.dtype
        # quaternionOrR = quaternionOrR.reshape(-1)
        # _quaternion = quaternionOrR.reshape(4, ) if len(quaternionOrR) == 4 else self.matrix_to_quaternion(quaternionOrR.reshape(3, 3)).reshape(4, )
        # _T = T.reshape(3, )

        # register('quaternion', _quaternion)
        # register('T', _T)
      
        
        self.znear, self.zfar = 0.01, 100.0

        
    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return self.__getattribute__(name)
    
    def cam_requires_grad_(self, requires_grad=True):
        self.quaternion.requires_grad_(requires_grad)
        self.T.requires_grad_(requires_grad)
    
    @property
    def projection_matrix(self):
        return getProjectionMatrix2(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)

    @property
    def projection_matrix_inverse(self):
        return self.projection_matrix.inverse()
    
    def init_(self, cam):
        self.quaternion.data.copy_(cam.quaternion.data)
        self.T.data.copy_(cam.T.data)
        return self

    @staticmethod
    def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    @staticmethod
    def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        
        def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
            ret = torch.zeros_like(x)
            positive_mask = x > 0
            ret[positive_mask] = torch.sqrt(x[positive_mask])
            return ret

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        q_abs = _sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )

        quat_by_rijk = torch.stack(
            [
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        return quat_candidates[
            F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
        ].reshape(batch_dim + (4,))

    @property
    def R(self) -> torch.Tensor:
        return self.quaternion_to_matrix(self.quaternion)
    
    @property
    def world_view_transform(self) -> torch.Tensor:
        matrix = torch.eye(4, device=self.quaternion.device, dtype=self.quaternion.dtype, requires_grad=False)
        matrix[:3, :3] = self.R.T
        matrix[:3, 3] = self.T
        return matrix.T
    
    @property
    def view_world_transform(self) -> torch.Tensor:
        return self.world_view_transform.inverse()
    
    @property
    def full_proj_transform(self) -> torch.Tensor:
        return self.world_view_transform @ self.projection_matrix
    
    @property
    def full_proj_transform_inverse(self) -> torch.Tensor:
        return self.full_proj_transform.inverse()
    
    @property
    def camera_center(self) -> torch.Tensor:
        return -self.T.view(1, 3) @ self.R.T
    
    def __repr__(self):
        return f"[Camera {self.uid}] Quaternion: {self.quaternion.detach().squeeze().cpu().numpy().tolist()}, Translation: {self.T.detach().squeeze().cpu().numpy().tolist()}"


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
