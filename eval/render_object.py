import os
import sys
import subprocess
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from sklearn.cluster import DBSCAN
import cv2

# Add submodule paths for imports
# sys.path.append(str(Path(__file__).parent))
# sys.path.append(str(Path(__file__).parent / "open_vocabulary_segmentation"))

from gaussian_model import GaussianModel
from render import render
from utils import MiniCam, focal2fov, get_world2view, build_text_embedding
from onnxruntime import InferenceSession
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, "eval/open_vocabulary_segmentation")
from models import build_model

def train_scene_if_needed(scene_path, encoder_path):
    scene_name = os.path.basename(os.path.normpath(scene_path))
    encoder_name = os.path.splitext(os.path.basename(encoder_path))[0]
    out_dir = f"results/{scene_name}_{encoder_name}/"
    
    pointcloud_path = Path(out_dir) / "experiment" / "ply" / "point_cloud" / "point_cloud.ply"
    if os.path.exists(pointcloud_path):
        print(f"Scene {scene_name} already trained, skipping training.")
        return out_dir
    print(f"Training scene {scene_name}...")
    command = [
        "./bin/replica_rgbd",
        "./ORB-SLAM3/Vocabulary/ORBvoc.txt",
        f"./cfg/ORB_SLAM3/RGB-D/Replica/{scene_name}.yaml",
        encoder_path,
        "./cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml",
        scene_path,
        out_dir,
        "no_viewer"
    ]
    result = subprocess.run(" ".join(command), shell=True)
    if result.returncode == 0:
        return out_dir
    else:
        return None

def render_object_images(scene_path, text_request, video_name, encoder_path="cfg/encoder/pca_encoder_imagenet.yaml"):
    SEMANTIC_SIMILARITY_THRESHOLD = 0.94
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    sh_degree = 3
    print("Reading Gaussian Splatting reconstruction...")
    gaussians = GaussianModel(sh_degree=sh_degree)
    pointcloud_path = Path(scene_path) / "experiment" / "ply" / "point_cloud" / "point_cloud.ply"
    print(f"Loading pointcloud from {pointcloud_path}")
    gaussians.load_ply(pointcloud_path)

    # Load camera params
    with open(Path(scene_path) / "experiment" / "ply" / "cameras.json", "r") as fin:
        camera_params = json.load(fin)
    width, height, fx, fy = camera_params[0]["width"], camera_params[0]["height"], camera_params[0]["fx"], camera_params[0]["fy"]
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)

    # Load DINO model and text embedding
    cfg = OmegaConf.load('eval/open_vocabulary_segmentation/talk2dino.yml')
    dino_model = build_model(cfg.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model.to(device).eval()
    with open(encoder_path, 'r') as f:
        f.readline()
        encoder_cfg = OmegaConf.load(f)
    pca_model_path = encoder_cfg["PixelwiseCompressor.Path"]
    pca_session = InferenceSession(pca_model_path)
    text_emb_compressed = build_text_embedding([text_request], dino_model, pca_session, device=device)

    # Compute similarity between each Gaussian and text embedding
    gaussian_features = gaussians.language_features.squeeze().detach()
    gaussian_positions = gaussians.xyz.detach()
    text_embed = text_emb_compressed.cuda() if device == "cuda" else text_emb_compressed
    text_embed = text_embed[0][..., None, None]
    gaussian_features_norm = F.normalize(gaussian_features, dim=1)
    text_embed_norm = F.normalize(text_embed[:,0], dim=0)
    similarities = torch.matmul(gaussian_features_norm, text_embed_norm)
    similarities = 1 - ((similarities - similarities.min()) / (similarities.max() - similarities.min()))
    similarities_np = similarities.cpu().numpy()

    # Cluster high-similarity Gaussians
    high_sim_mask = similarities_np > SEMANTIC_SIMILARITY_THRESHOLD
    high_sim_mask = high_sim_mask.squeeze()
    high_sim_points = gaussian_positions[high_sim_mask].cpu().numpy()
    object_centers = []
    if len(high_sim_points) > 0:
        clustering = DBSCAN(eps=0.16, min_samples=5).fit(high_sim_points)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        for cluster_id in range(n_clusters):
            cluster_points = high_sim_points[labels == cluster_id]
            if len(cluster_points) == 0:
                continue
            center = cluster_points.mean(axis=0)
            object_centers.append(center)

    # For each object, render orbit video
    orbit_radius = 1.0
    for idx, center in enumerate(object_centers):
        gaussian_positions_np = gaussian_positions.cpu().numpy()
        dists = np.linalg.norm(gaussian_positions_np - center, axis=1)
        mask = dists < 0.1
        red_color = torch.tensor([4.0, 0.0, 0.0], dtype=gaussians.features_dc.dtype, device=gaussians.features_dc.device)
        gaussians_features_dc_orig = gaussians.features_dc.clone()
        # if np.sum(mask) > 0:
        #     gaussians.features_dc[mask] = red_color
        orbit_cameras = generate_spherical_trajectory(center, orbit_radius, axis='y')
        video_folder = "ovs_videos"
        os.makedirs(video_folder, exist_ok=True)
        video_path = f"{video_folder}/{video_name}_object_{idx}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        for cam_idx, cam_params in enumerate(tqdm(orbit_cameras, desc=f"Rendering object {idx}")):
            if cam_idx == 200:
                gaussians._features_dc = gaussians_features_dc_orig.clone()
            world_view_transform = get_world2view(
                np.array(cam_params["rotation"]),
                np.array(cam_params["position"])
            )
            cam = MiniCam(width, height, fovx, fovy, world_view_transform)
            render_result = render(cam, gaussians, background)
            rendered_image = render_result["rendered_image"].permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = np.uint8(np.clip(rendered_image * 255, 0, 255))
            bgr = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
            video_out.write(bgr)
        video_out.release()
        print(f"Saved orbit video for object {idx} to {video_path}")

def generate_spherical_trajectory(center: np.ndarray, radius: float, num_frames: int = 60, axis: str = 'z') -> list:
    cameras = []
    for i in range(num_frames):
        theta = 2 * np.pi * i / num_frames
        if axis == 'x':
            x = center[0]
            y = center[1] + radius * np.cos(theta)
            z = center[2] + radius * np.sin(theta)
        elif axis == 'y':
            x = center[0] + radius * np.cos(theta)
            y = center[1]
            z = center[2] + radius * np.sin(theta)
        else:
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            z = center[2]
        position = np.array([x, y, z])
        # Look at center
        forward = center - position
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        R = np.stack([right, up, forward], axis=1)
        cameras.append({"position": position, "rotation": R})
    return cameras

def main():
    parser = argparse.ArgumentParser(description="Train and render object images from a scene using Gaussian SLAM Splatting.")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to the scene directory (raw frames).")
    parser.add_argument("--text_request", type=str, required=True, help="Text prompt for object rendering.")
    parser.add_argument("--encoder_path", type=str, default="cfg/encoder/pca_encoder_imagenet.yaml", help="Path to the encoder config YAML file.")
    args = parser.parse_args()

    scene_name = os.path.basename(os.path.normpath(args.scene_path))
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = args.text_request.replace(' ', '_').replace('/', '_')
    video_name = f"{scene_name}_{os.path.splitext(os.path.basename(args.encoder_path))[0]}_{safe_text}_{now_str}"

    trained_out_dir = train_scene_if_needed(args.scene_path, args.encoder_path)
    if trained_out_dir is None:
        print(f"Training failed or was interrupted.")
        return
    with torch.no_grad():
        render_object_images(trained_out_dir, args.text_request, video_name, encoder_path=args.encoder_path)
    print(f"Rendering complete. Output video prefix: {video_name}")

if __name__ == "__main__":
    main() 