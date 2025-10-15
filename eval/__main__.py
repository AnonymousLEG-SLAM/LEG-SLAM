from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.nn.functional as F

from gaussian_model import GaussianModel
from render import render
from sh_utils import SH2RGB
from utils import MiniCam, focal2fov, get_world2view

def log_gaussian(scene_path: Path) -> None:
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    sh_degree = 3
    print("Reading Gaussian Splatting reconstruction...")
    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(scene_path / "point_cloud" / "iteration_1288" / "point_cloud.ply")

    point_colors = SH2RGB(gaussians.features_dc).squeeze().detach().cpu().numpy()
    points = gaussians.xyz.detach().cpu().numpy()

    with open(scene_path / "cameras.json", "r") as fin:
        camera_params = json.load(fin)

    width, height, fx, fy = camera_params[0]["width"], camera_params[0]["height"], camera_params[0]["fx"], \
    camera_params[0]["fy"]
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)

    rendered_images = []
    rendered_lang_feats_dist = []
    cameras = []

    text_embed = torch.tensor(
        [-0.0116534, -0.02225662, -0.02598745, 0.05149937, 0.03452285, -0.02354565, 0.02379932, -0.01006881,
         -0.00519741, -0.01826517, -0.04495092, 0.01896522, -0.01121542, -0.00211633, 0.00826379, 0.017287,
         0.06617628, 0.00882252, 0.01123158, -0.10463687, 0.00269056, 0.00820926, 0.0724728, -0.01628493,
         -0.05378885, -0.01450618, -0.03543069, -0.03008726, -0.00202858, -0.02311744, -0.00836567, 0.04784993]).cuda()

    text_embed = text_embed[..., None, None]

    for camera_param in camera_params:
        world_view_transform2 = get_world2view(np.array(camera_param["rotation"]), np.array(camera_param["position"]))
        cam = MiniCam(width, height, fovx, fovy, world_view_transform2)
        render_result = render(cam, gaussians, background)
        dist = F.cosine_similarity(render_result["rendered_lf"], text_embed, dim=0).detach().cpu().numpy()
        rendered_images.append(render_result["rendered_image"].permute(1, 2, 0).detach().cpu().numpy())
        rendered_lang_feats_dist.append(dist)
        cameras.append(cam)

    rendered_images = np.stack(rendered_images)
    rendered_images = np.uint8(np.clip(rendered_images * 255, 0, 255))

    rendered_lang_feats_dist = np.stack(rendered_lang_feats_dist)
    rendered_lang_feats_dist = 1 - ((rendered_lang_feats_dist - rendered_lang_feats_dist.min()) / (rendered_lang_feats_dist.max() - rendered_lang_feats_dist.min()))

    print("Building visualization by logging to Rerun")
    rr.log("points", rr.Points3D(points, colors=point_colors))

    for idx in range(rendered_images.shape[0]):
        rgb = rendered_images[idx]
        dist = rendered_lang_feats_dist[idx]
        rr.log("camera/image", rr.Image(rgb, color_model="RGB"))
        rr.log("camera/lf_dist", rr.DepthImage(dist, depth_range=(0, 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizes the LEGS-SLAM scene using Rerun SDK")
    parser.add_argument("--scene_path", type=Path, required=True)
    parser.add_argument("--categories_path", type=Path, required=True)

    rr.script_add_args(parser)
    args = parser.parse_args()

    blueprint = rrb.Vertical(
        rrb.Spatial3DView(name="3D", origin="/"),
        rrb.Horizontal(
            rrb.Spatial2DView(name="Camera", origin="/camera/image"),
            rrb.Spatial2DView(name="Language Features", origin="/camera/lf_dist"),
        ),
        row_shares=[2, 2],
    )

    rr.script_setup(args, "rerun_example_structure_from_motion", default_blueprint=blueprint)
    log_gaussian(args.scene_path)
    rr.script_teardown(args)

if __name__ == "__main__":
    main()
