from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
from datetime import datetime

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.nn.functional as F
import math

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

from sklearn.cluster import DBSCAN
            
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


def generate_spherical_trajectory(center: np.ndarray, radius: float, num_frames: int = 60, axis: str = 'z') -> list[dict]:
    """Generate camera positions in a structured spherical grid around the center point.
    
    Args:
        center: 3D center point to orbit around
        radius: Distance from center to camera
        num_frames: Number of frames in the trajectory
        axis: Which axis to use as the primary rotation axis ('x', 'y', or 'z')
        
    Returns:
        List of camera parameters (position and rotation)
    """
    cameras = []
    
    # Number of layers in each dimension
    n_layers = 1000
    
    # Define coordinate permutations based on axis
    if axis == 'x':
        # x is up/down, y and z form the circle
        def permute_coords(x, y, z):
            return z, x, y
    elif axis == 'y':
        # y is up/down, x and z form the circle
        def permute_coords(x, y, z):
            return x, z, y
    else:  # 'z' is default
        # z is up/down, x and y form the circle
        def permute_coords(x, y, z):
            return x, y, z
    
    # Generate points in a structured spherical grid
    for i in range(1):
        # Vertical angle (theta) from top to bottom
        # theta = math.pi * (i + 0.5) / n_layers  # +0.5 to avoid poles
        # theta = math.pi * 1  # +0.5 to avoid poles
        theta = math.pi * 0.5  # +0.5 to avoid poles
        
        for j in range(n_layers):
            # Horizontal angle (phi) around the sphere
            phi = 2 * math.pi * j / n_layers
            
            # Convert spherical to Cartesian coordinates
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            
            # Permute coordinates based on chosen axis
            x, y, z = permute_coords(x, y, z)
            
            # Add center offset
            x += center[0]
            y += center[1]
            z += center[2]
            
            # Calculate camera rotation to look at center
            position = np.array([x, y, z])
            forward = center - position
            forward = forward / np.linalg.norm(forward)
            
            # Calculate up vector (assuming world up is [0, 1, 0])
            world_up = np.array([0, 1, 0])
            right = np.cross(forward, world_up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # Create rotation matrix
            rotation = np.column_stack([right, up, forward])
            
            cameras.append({
                "position": position.tolist(),
                "rotation": rotation.tolist()
            })
    
    return cameras


def render_gaussian_images(scene_path: Path, text_emb_compressed: torch.Tensor, dino_model: torch.nn.Module, video_folder: str, request: str, use_rerun: bool, visualize_trajectory: bool) -> None:
    SEMANTIC_SIMILARITY_THRESHOLD = 0.94
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    sh_degree = 3
    print("Reading Gaussian Splatting reconstruction...")
    gaussians = GaussianModel(sh_degree=sh_degree)
    pointcloud_path = scene_path / "experiment" / "ply" / "point_cloud" / "point_cloud.ply"
    print(f"Loading pointcloud from {pointcloud_path}")
    gaussians.load_ply(pointcloud_path)

    video_name = scene_path.name + "_" + request + "_" + str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    # Get Gaussian features and positions
    gaussian_features = gaussians.language_features.squeeze().detach()  # [N, 3]
    print("gaussian_features.shape", gaussian_features.shape)
    gaussian_positions = gaussians.xyz.detach()  # [N, 3]
    
    # Compute similarity between each Gaussian and text embedding
    text_embed = text_emb_compressed.cuda()
    # text_embed = text_embed[0]  # Take first request
    text_embed = text_embed[0][..., None, None] # Take first request
    
    # Normalize features for cosine similarity
    gaussian_features_norm = F.normalize(gaussian_features, dim=1)
    text_embed_norm = F.normalize(text_embed[:,0], dim=0)
    
    # Compute cosine similarity for each Gaussian
    similarities = torch.matmul(gaussian_features_norm, text_embed_norm)  # [N]
    similarities = 1 - ((similarities - similarities.min()) / (similarities.max() - similarities.min()))
    
    print("similarities.shape", similarities.shape)
    
    print("similarities.min(), similarities.max()", similarities.min(), similarities.max())
    # Set color to fully red for gaussians with similarities > 0
    red_color = torch.tensor([4.0, 0.0, 0.0], dtype=gaussians.features_dc.dtype, device=gaussians.features_dc.device)
    mask = similarities > SEMANTIC_SIMILARITY_THRESHOLD
    # gaussians.features_dc[mask] = red_color
    # print(f"Set {mask.sum().item()} gaussians to fully red because similarity > 0")
    
    # Convert to numpy for visualization
    similarities_np = similarities.cpu().numpy()

    with open(scene_path / "experiment" / "ply" / "cameras.json", "r") as fin:
        camera_params = json.load(fin)

    width, height, fx, fy = camera_params[0]["width"], camera_params[0]["height"], camera_params[0]["fx"], \
    camera_params[0]["fy"]
    print("width, height, fx, fy", width, height, fx, fy)
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    
    # Analyze Gaussians with high similarity, cluster into objects, calculate center of objects in 3D

    # Select Gaussians with high similarity (>0.8)
    high_sim_mask = similarities_np > SEMANTIC_SIMILARITY_THRESHOLD
    high_sim_mask = high_sim_mask.squeeze()  # Remove extra dimension
    high_sim_points = gaussian_positions[high_sim_mask].cpu().numpy()

    if len(high_sim_points) == 0:
        print(f"No Gaussians found with similarity > {SEMANTIC_SIMILARITY_THRESHOLD}.")
    else:
        # Cluster the high similarity points into objects using DBSCAN

        # DBSCAN parameters may need tuning depending on scene scale
        clustering = DBSCAN(eps=0.16, min_samples=5).fit(high_sim_points)
        labels = clustering.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Found {n_clusters} object clusters among high-similarity Gaussians.")

        object_centers = []
        for cluster_id in range(n_clusters):
            cluster_points = high_sim_points[labels == cluster_id]
            if len(cluster_points) == 0:
                continue
            center = cluster_points.mean(axis=0)
            object_centers.append(center)
            print(f"Object {cluster_id}: center at {center}")

        if len(object_centers) == 0:
            print("No valid object clusters found.")
        else:
            # For each center, color Gaussians within radius R green
            R = 0.1  # You can adjust this radius as needed
            red_color = torch.tensor([4.0, 0.0, 0.0], dtype=gaussians.features_dc.dtype, device=gaussians.features_dc.device)
            gaussian_positions_np = gaussian_positions.cpu().numpy()
            
            # Create video for each object
            for idx, center in enumerate(object_centers):
                print(f"Object {idx} 3D center: {center}")
                # Compute distances from all Gaussians to this center
                dists = np.linalg.norm(gaussian_positions_np - center, axis=1)
                mask = dists < R
                num_colored = np.sum(mask)
                gaussians_features_dc_orig = gaussians.features_dc.clone()
                if num_colored > 0:
                    gaussians.features_dc[mask] = red_color
                # print(f"Colored {num_colored} Gaussians green within radius {R} of center {center}")
                
                # Generate spherical trajectory around this object
                orbit_radius = 1.0  # Fixed radius of 1 unit
                orbit_cameras = generate_spherical_trajectory(center, orbit_radius, axis='y')
                # orbit_cameras = generate_spherical_trajectory(center, orbit_radius, axis='x')
                
                # Create video writer for this object
                video_path = f"{video_folder}/{video_name}_object_{idx}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                
                # Render frames from each camera position
                for cam_idx, cam_params in enumerate(tqdm(orbit_cameras, desc=f"Rendering object {idx}")):
                    if cam_idx == 200:
                        gaussians._features_dc = gaussians_features_dc_orig.clone()
                        
                    world_view_transform = get_world2view(
                        np.array(cam_params["rotation"]),
                        np.array(cam_params["position"])
                    )
                    
                    cam = MiniCam(width, height, fovx, fovy, world_view_transform)
                    render_result = render(cam, gaussians, background)
                    
                    # Get depth at center of frame
                    center_x, center_y = width // 2, height // 2
                    center_region = 15  # Check a small region around center
                    depth = render_result["rendered_depth"].detach().cpu().numpy().squeeze()
                    
                    center_depth = depth[
                        center_y - center_region:center_y + center_region,
                        center_x - center_region:center_x + center_region
                    ].mean()
                                        
                    # Skip frame if center depth is significantly less than radius
                    # This indicates something is in front of the object
                    
                    rendered_image = render_result["rendered_image"].permute(1, 2, 0).detach().cpu().numpy()
                    rendered_image = np.uint8(np.clip(rendered_image * 255, 0, 255))
                    # rendered_image[
                    #     center_y - center_region:center_y + center_region,
                    #     center_x - center_region:center_x + center_region
                    # ] = [0, 0, 255]
                    if center_depth < orbit_radius * 0.7:
                        # rendered_image[
                        #     center_y - center_region:center_y + center_region,
                        #     center_x - center_region:center_x + center_region
                        # ] = [255, 0, 0]
                        pass
                    else:
                        bgr = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
                        video_out.write(bgr)
                        
                        # Log frame to rerun for visualization
                        if use_rerun:
                            rr.log(f"object_0/orbit", rr.Image(rendered_image, color_model="RGB"))
                
                video_out.release()
                print(f"Saved orbit video for object {idx} to {video_path}")

    if visualize_trajectory:
        rendered_images = []
        rendered_lang_feats_dist = []
        cameras = []

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(f"{video_folder}/{video_name}_{request}_trajectory.mp4", fourcc, 10.0, (width, height*2))

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

        for idx in range(rendered_images.shape[0]):
            rgb = rendered_images[idx].copy()  # Create a copy to avoid modifying original array
            dist = rendered_lang_feats_dist[idx]
            
            bboxes, binary_mask = find_bboxes(dist, threshold=0.8)
                
            dist_colored = cv2.applyColorMap(np.uint8(dist * 255), cv2.COLORMAP_JET)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            combined = np.vstack([bgr, dist_colored])
            
            video_out.write(combined)
                
            if use_rerun:
                rr.log("camera/image", rr.Image(rgb, color_model="RGB"))
                rr.log("camera/lf_dist", rr.DepthImage(dist, depth_range=(0, 1)))
                rr.log("camera/binary_mask", rr.DepthImage(binary_mask, depth_range=(0, 1)))
        
        video_out.release()
        
        # Print statistics about semantic similarities
        print(f"\nSemantic similarity statistics:")
        print(f"Mean similarity: {np.mean(similarities_np):.3f}")
        print(f"Max similarity: {np.max(similarities_np):.3f}")
        print(f"Min similarity: {np.min(similarities_np):.3f}")
        print(f"Number of Gaussians with similarity > {SEMANTIC_SIMILARITY_THRESHOLD}: {np.sum(similarities_np > SEMANTIC_SIMILARITY_THRESHOLD)}")
        print(f"Number of Gaussians with similarity > 0.5: {np.sum(similarities_np > 0.5)}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizes the LEGS-SLAM scene using Rerun SDK")
    parser.add_argument("--scene_path", type=Path, required=True)
    parser.add_argument("--debug_prompt", type=str, required=True)
    parser.add_argument("--encoder_path", type=str, default="cfg/encoder/pca_encoder_imagenet.yaml")
    parser.add_argument("--use_rerun", type=bool, default=False)
    parser.add_argument("--visualize_trajectory", type=bool, default=False)
    
    args = parser.parse_args()
    
    video_folder = "ovs_videos"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    
    cfg = OmegaConf.load('eval/open_vocabulary_segmentation/talk2dino.yml')
    dino_model = build_model(cfg.model)
    dino_model.to(device).eval()
    
    with open(args.encoder_path, 'r') as f:
        f.readline()  # Skip YAML header
        cfg = OmegaConf.load(f)
    pca_model_path = cfg["PixelwiseCompressor.Path"]
    
    pca_session = InferenceSession(pca_model_path)
    
    text_emb_compressed = build_text_embedding([args.debug_prompt], dino_model, pca_session)
    
    if args.use_rerun:
        rr.script_add_args(parser)
    args = parser.parse_args()
    if args.use_rerun:
        blueprint = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Camera", origin="/camera/image"),
                rrb.Spatial2DView(name="Language Features", origin="/camera/lf_dist"),
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(name="Binary Mask", origin="/camera/binary_mask"),
                rrb.Spatial2DView(name="Object Orbits", origin="/object_0/orbit"),
            ),
            row_shares=[1, 1],
        )
        rr.script_setup(args, "rerun_example_", default_blueprint=blueprint)
    
    with torch.no_grad():
        render_gaussian_images(args.scene_path, text_emb_compressed, dino_model, video_folder, args.debug_prompt, args.use_rerun, args.visualize_trajectory)
    if args.use_rerun:
        rr.script_teardown(args)

if __name__ == "__main__":
    main()
