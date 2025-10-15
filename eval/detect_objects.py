from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.nn.functional as F

from gaussian_model import GaussianModel
from render import render
from sh_utils import SH2RGB
from utils import MiniCam, focal2fov, get_world2view

import sys
import clip
import pickle
from omegaconf import OmegaConf
from onnxruntime import InferenceSession
import cv2

sys.path.insert(0, "eval/open_vocabulary_segmentation")
from models import build_model
from tqdm import tqdm

from utils import build_text_embedding

device = "cuda" if torch.cuda.is_available() else "cpu"


def find_bboxes(dist_img: np.ndarray, threshold: float) -> list[tuple[int, int, int, int]]:
    """Find bounding boxes of target objects in a distance image.

    Args:
        dist_img: Distance image where each pixel represents similarity score (0-1).
        threshold: Threshold value for binary segmentation (default: 0.7).

    Returns:
        List of bounding boxes as (x_min, y_min, x_max, y_max) tuples and binary mask.
    """
    # Apply larger kernel averaging filter to reduce noise while preserving boundaries
    scale = 30
    kernel = np.ones((scale, scale)) / (scale**2)
    dist_img_avg = cv2.filter2D(dist_img, -1, kernel)
    
    # Combine averaged and original to suppress noise but keep sharp boundaries
    dist_img_combined = 0.5 * (dist_img_avg + dist_img)
    
    # Threshold to get binary mask
    binary_mask = (dist_img_combined > threshold).astype(np.uint8) * 255

    # Find contours of connected components
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes from contours, filter small boxes
    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
    bboxes = [bbox for bbox in bboxes if bbox[2] > 20 and bbox[3] > 20]

    # Convert (x, y, w, h) to (x_min, y_min, x_max, y_max)
    return [(x, y, x + w, y + h) for x, y, w, h in bboxes], binary_mask


def log_gaussian(scene_path: Path, text_emb_compressed: torch.Tensor, dino_model: torch.nn.Module, video_folder: str, video_name: str, request: str) -> None:
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    sh_degree = 3
    print("Reading Gaussian Splatting reconstruction...")
    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(scene_path / "point_cloud" / "point_cloud.ply")

    point_colors = SH2RGB(gaussians.features_dc).squeeze().detach().cpu().numpy()
    points = gaussians.xyz.detach().cpu().numpy()

    with open(scene_path / "cameras.json", "r") as fin:
        camera_params = json.load(fin)

    width, height, fx, fy = camera_params[0]["width"], camera_params[0]["height"], camera_params[0]["fx"], \
    camera_params[0]["fy"]
    print("width, height, fx, fy", width, height, fx, fy)
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)

    rendered_images = []
    rendered_lang_feats_dist = []
    cameras = []

    text_embed = text_emb_compressed.cuda()
    text_embed = text_embed[0][..., None, None] # Take first request

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(f"{video_folder}/{video_name}_{request}.mp4", fourcc, 10.0, (width, height*2))

    for idx, camera_param in enumerate(tqdm(camera_params)):
        world_view_transform2 = get_world2view(np.array(camera_param["rotation"]), np.array(camera_param["position"]))

        cam = MiniCam(width, height, fovx, fovy, world_view_transform2)
        render_result = render(cam, gaussians, background)
        
        rendered_image = render_result["rendered_image"].permute(1, 2, 0).detach()
        rendered_lf = render_result["rendered_lf"].detach()
        rendered_image_pamr = rendered_image.permute(2, 0, 1).unsqueeze(0)
        
        dist = F.cosine_similarity(rendered_lf, text_embed, dim=0).detach()
        dist_pamr = dist.unsqueeze(0).unsqueeze(0)
        dist_pamr = dino_model.apply_pamr(rendered_image_pamr, dist_pamr)
        dist = dist_pamr.squeeze(0).squeeze(0)
    
        rendered_images.append(rendered_image.detach().cpu().numpy())
        rendered_lang_feats_dist.append(dist.detach().cpu().numpy())
        cameras.append(cam)

    rendered_images = np.stack(rendered_images)
    rendered_images = np.uint8(np.clip(rendered_images * 255, 0, 255))

    rendered_lang_feats_dist = np.stack(rendered_lang_feats_dist)
    rendered_lang_feats_dist = 1 - ((rendered_lang_feats_dist - rendered_lang_feats_dist.min()) / (rendered_lang_feats_dist.max() - rendered_lang_feats_dist.min()))

    print("Building visualization by logging to Rerun")
    rr.log("points", rr.Points3D(points, colors=point_colors))

    for idx in range(rendered_images.shape[0]):
        rgb = rendered_images[idx].copy()  # Create a copy to avoid modifying original array
        dist = rendered_lang_feats_dist[idx]
        
        bboxes, binary_mask = find_bboxes(dist, threshold=0.8)
        for bbox in bboxes:
            rgb = cv2.rectangle(rgb.copy(), (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
        dist_colored = cv2.applyColorMap(np.uint8(dist * 255), cv2.COLORMAP_JET)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        combined = np.vstack([bgr, dist_colored])
        
        video_out.write(combined)
            
        rr.log("camera/image", rr.Image(rgb, color_model="RGB"))
        rr.log("camera/lf_dist", rr.DepthImage(dist, depth_range=(0, 1)))
        rr.log("camera/binary_mask", rr.DepthImage(binary_mask, depth_range=(0, 1)))
    
    video_out.release()
        
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizes the LEGS-SLAM scene using Rerun SDK")
    parser.add_argument("--scene_path", type=Path, required=True)
    parser.add_argument("--debug_prompt", type=str, required=True)
    parser.add_argument("--video_name", type=str, required=True)
    
    args = parser.parse_args()
    
    video_folder = "ovs_videos"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    
    cfg = OmegaConf.load('eval/open_vocabulary_segmentation/talk2dino.yml')
    dino_model = build_model(cfg.model)
    dino_model.to(device).eval()
    
    with open('cfg/encoder/pca_text_emb64_scannet_test.yaml', 'r') as f:
        f.readline()  # Skip YAML header
        cfg = OmegaConf.load(f)
    pca_model_path = cfg["PixelwiseCompressor.Path"]
    
    pca_session = InferenceSession(pca_model_path)
    
    text_emb_compressed = build_text_embedding([args.debug_prompt], dino_model, pca_session)
    
    rr.script_add_args(parser)
    args = parser.parse_args()

    blueprint = rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Camera", origin="/camera/image"),
            rrb.Spatial2DView(name="Language Features", origin="/camera/lf_dist"),
        ),
        rrb.Spatial2DView(name="Binary Mask", origin="/camera/binary_mask"),
        row_shares=[1, 1],
    )

    rr.script_setup(args, "rerun_example_structure_from_motion", default_blueprint=blueprint)
    log_gaussian(args.scene_path, text_emb_compressed, dino_model, video_folder, args.video_name, args.debug_prompt)
    rr.script_teardown(args)

if __name__ == "__main__":
    main()
