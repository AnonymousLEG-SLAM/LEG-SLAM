import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from plyfile import PlyData
from simple_knn._C import distCUDA2


class GaussianModel:
    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._language_features = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def features_dc(self) -> torch.Tensor:
        return self._features_dc

    @property
    def features(self) -> torch.Tensor:
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def language_features(self) -> torch.Tensor:
        return self._language_features

    @property
    def opacity(self):
        return self.opacity_activation(self._opacity)

    def load_ply(self, path: Path) -> None:
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        f_rest_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        f_rest_names.sort(key=lambda x: int(x.split('_')[-1]))
        features_rest = np.zeros((xyz.shape[0], len(f_rest_names)))
        for idx, attr_name in enumerate(f_rest_names):
            features_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_rest = features_rest.reshape((features_rest.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        lf_names = [prop.name for prop in plydata.elements[0].properties if prop.name.startswith("lf_")]
        lf_names.sort(key=lambda x: int(x.split('_')[-1]))
        language_features = np.empty((xyz.shape[0], len(lf_names)), dtype=np.float32)
        for idx, attr_name in enumerate(lf_names):
            language_features[:, idx] = np.asarray(plydata.elements[0][attr_name])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., None]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names.sort(key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names.sort(key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda")
                                         .transpose(1, 2)
                                         .contiguous())
        self._features_rest = nn.Parameter(torch.tensor(features_rest, dtype=torch.float, device="cuda")
                                           .transpose(1, 2)
                                           .contiguous())
        self._language_features = nn.Parameter(torch.tensor(language_features, dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda"))

        self.active_sh_degree = self.max_sh_degree
