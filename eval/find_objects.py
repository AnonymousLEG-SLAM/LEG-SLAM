from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
from typing import List, Tuple, Optional

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

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

def compute_3d_center(points: np.ndarray, mask: np.ndarray, depth: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """Compute 3D center of object using depth and camera pose.

    Args:
        points: 2D points in image coordinates
        mask: Binary mask of object
        depth: Depth map
        camera_pose: Camera pose matrix

    Returns:
        3D center coordinates of object, or None if no valid points
    """
    # Get points where mask is 1
    y_coords, x_coords = np.where(mask > 0)

    if len(x_coords) == 0 or len(y_coords) == 0:
        print(f"No valid mask pixels found for object center computation. mask shape: {mask.shape}")
        return None

    valid_points = np.stack([x_coords, y_coords], axis=1)

    # Get corresponding depths
    try:
        valid_depths = depth[y_coords, x_coords]
    except Exception as e:
        print(f"Error indexing depth with mask coordinates: {e}")
        print(f"y_coords shape: {y_coords.shape}, x_coords shape: {x_coords.shape}, depth shape: {depth.shape}")
        return None

    # Convert to 3D points
    fx, fy = camera_pose[0, 0], camera_pose[1, 1]
    cx, cy = camera_pose[0, 2], camera_pose[1, 2]

    if valid_points.shape[0] == 0 or valid_depths.shape[0] == 0:
        print(f"No valid points after masking. valid_points shape: {valid_points.shape}, valid_depths shape: {valid_depths.shape}")
        return None

    try:
        X = (valid_points[:, 0] - cx) * valid_depths / fx
        Y = (valid_points[:, 1] - cy) * valid_depths / fy
        Z = valid_depths
    except Exception as e:
        print(f"Error computing 3D coordinates: {e}")
        print(f"valid_points shape: {valid_points.shape}, valid_depths shape: {valid_depths.shape}, fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
        return None

    points_3d = np.stack([X, Y, Z], axis=1)

    # Transform to world coordinates
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    try:
        points_3d_world = (R.T @ (points_3d - t).T).T
    except Exception as e:
        print(f"Error transforming to world coordinates: {e}")
        print(f"points_3d shape: {points_3d.shape}, R shape: {R.shape}, t shape: {t.shape}")
        return None

    # Compute weighted center
    weights = mask[y_coords, x_coords]
    if np.sum(weights) == 0:
        print("All mask weights are zero, cannot compute weighted center.")
        return None

    try:
        center = np.average(points_3d_world, weights=weights, axis=0)
    except Exception as e:
        print(f"Error computing weighted average for object center: {e}")
        print(f"points_3d_world shape: {points_3d_world.shape}, weights shape: {weights.shape}")
        return None

    print(f"Computed object center: {center} from {len(weights)} valid mask pixels.")
    return center

def generate_sphere_points(center: np.ndarray, radius: float, num_points: int) -> List[np.ndarray]:
    """Generate points on a sphere around the object center.
    
    Args:
        center: 3D center point
        radius: Sphere radius
        num_points: Number of points to generate
        
    Returns:
        List of camera positions
    """
    # Generate points on unit sphere
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    points = []
    
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        
        # Scale and translate
        point = np.array([x, y, z]) * radius + center
        points.append(point)
        
    return points

def compute_camera_pose(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute camera pose matrix looking at target from position.
    
    Args:
        position: Camera position
        target: Point to look at
        
    Returns:
        4x4 camera pose matrix
    """
    # Compute forward direction
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    
    # Compute right direction (assuming up is [0, 1, 0])
    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recompute up to ensure orthogonality
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Build rotation matrix
    R = np.stack([right, up, -forward], axis=1)
    
    # Build full pose matrix
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = position
    
    return pose

def is_valid_viewpoint(position: np.ndarray, target: np.ndarray, points: np.ndarray, 
                      min_distance: float = 0.1, max_distance: float = 10.0) -> bool:
    """Check if camera position is valid.
    
    Args:
        position: Camera position
        target: Target point
        points: Scene points
        min_distance: Minimum distance to points
        max_distance: Maximum distance to target
        
    Returns:
        True if viewpoint is valid
    """
    # Check distance to target
    dist_to_target = np.linalg.norm(position - target)
    if dist_to_target > max_distance:
        return False
        
    # Check distance to scene points
    dists_to_points = np.linalg.norm(points - position, axis=1)
    if np.any(dists_to_points < min_distance):
        return False
        
    return True

def find_bboxes(dist_img: np.ndarray, threshold: float) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
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

def log_gaussian(scene_path: Path, text_emb_compressed: torch.Tensor, dino_model: torch.nn.Module, 
                video_folder: str, video_name: str, request: str, num_views: int = 8) -> None:
    """Render object from multiple viewpoints.
    
    Args:
        scene_path: Path to scene data
        text_emb_compressed: Compressed text embeddings
        dino_model: DINO model for segmentation
        video_folder: Output video folder
        video_name: Output video name
        request: Text query
        num_views: Number of viewpoints to render
    """
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    sh_degree = 3
    print("Reading Gaussian Splatting reconstruction...")
    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(scene_path / "point_cloud" / "point_cloud.ply")

    points = gaussians.xyz.detach().cpu().numpy()

    with open(scene_path / "cameras.json", "r") as fin:
        camera_params = json.load(fin)

    width, height, fx, fy = camera_params[0]["width"], camera_params[0]["height"], camera_params[0]["fx"], \
    camera_params[0]["fy"]
    print(f"width={width}, height={height}, fx={fx}, fy={fy}")
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)

    # First pass: Find object center and collect all views
    print("Finding object center and collecting views...")
    max_high_intensity_area = 0
    best_frame_idx = -1
    best_mask = None
    best_depth = None
    best_camera_pose = None
    best_semantic_mask = None  # Store the best semantic mask
    
    rendered_images = []
    rendered_lang_feats_dist = []
    rendered_depths = []  # Store rendered depths
    cameras = []
    
    text_embed = text_emb_compressed.cuda()
    text_embed = text_embed[0][..., None, None]  # Take first request
    
    for idx, camera_param in enumerate(tqdm(camera_params)):
        world_view_transform2 = get_world2view(np.array(camera_param["rotation"]), np.array(camera_param["position"]))
        cam = MiniCam(width, height, fovx, fovy, world_view_transform2)
        render_result = render(cam, gaussians, background)
        
        rendered_image = render_result["rendered_image"].permute(1, 2, 0).detach()
        rendered_lf = render_result["rendered_lf"].detach()
        rendered_depth = render_result["rendered_depth"].detach()
        rendered_image_pamr = rendered_image.permute(2, 0, 1).unsqueeze(0)
        
        dist = F.cosine_similarity(rendered_lf, text_embed, dim=0).detach()
        dist_pamr = dist.unsqueeze(0).unsqueeze(0)
        dist_pamr = dino_model.apply_pamr(rendered_image_pamr, dist_pamr)
        dist = dist_pamr.squeeze(0).squeeze(0)
        
        # Store all rendered images and features
        rendered_images.append(rendered_image.detach().cpu().numpy())
        rendered_lang_feats_dist.append(dist.detach().cpu().numpy())
        rendered_depths.append(rendered_depth.detach().cpu().numpy())  # Store depth
        cameras.append(cam)
    
    # Process all language feature distances at once
    rendered_images = np.stack(rendered_images)
    rendered_images = np.uint8(np.clip(rendered_images * 255, 0, 255))
    
    rendered_lang_feats_dist = np.stack(rendered_lang_feats_dist)
    rendered_lang_feats_dist = 1 - ((rendered_lang_feats_dist - rendered_lang_feats_dist.min()) / 
                                  (rendered_lang_feats_dist.max() - rendered_lang_feats_dist.min()))
    
    # Now process each frame with normalized distances
    for idx in range(len(camera_params)):
        dist = rendered_lang_feats_dist[idx]
        
        # Find object mask
        bboxes, binary_mask = find_bboxes(dist, threshold=0.5)
        
        if len(bboxes) > 0:
            # Find bbox with largest area of high semantic correlation in this frame
            frame_high_intensity_area = 0
            
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                # Get the region of high semantic correlation
                bbox_mask = (dist > 0.5)[y_min:y_max, x_min:x_max]
                high_intensity_area = np.sum(bbox_mask)
                frame_high_intensity_area += high_intensity_area
            
            print(f"frame_high_intensity_area: {frame_high_intensity_area}")
            # Update global best if this frame has more high intensity area
            if frame_high_intensity_area > max_high_intensity_area:
                max_high_intensity_area = frame_high_intensity_area
                best_frame_idx = idx
                
                # Create binary mask for all high correlation pixels in this frame
                binary_mask = np.zeros_like(dist, dtype=np.uint8)
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max = bbox
                    binary_mask[y_min:y_max, x_min:x_max] = (dist > 0.5)[y_min:y_max, x_min:x_max] * 255
                
                best_mask = binary_mask
                best_depth = rendered_depths[idx]  # Use stored depth
                best_camera_pose = get_world2view(np.array(camera_params[idx]["rotation"]), 
                                               np.array(camera_params[idx]["position"]))
                best_semantic_mask = dist  # Store the semantic mask
    
    if best_frame_idx == -1:
        print("No object found! Skipping new viewpoint generation.")
    else:
        print(f"Found best frame {best_frame_idx} with high intensity area: {max_high_intensity_area}")
        # Compute object center
        object_center = compute_3d_center(points, best_mask, best_depth, best_camera_pose)
        print(f"Object center: {object_center}")
        
        # Generate viewpoints
        radius = 2.0  # Adjust based on scene scale
        viewpoints = generate_sphere_points(object_center, radius, num_views)
        
        # Filter valid viewpoints
        valid_viewpoints = []
        for pos in viewpoints:
            if is_valid_viewpoint(pos, object_center, points):
                valid_viewpoints.append(pos)
        
        if not valid_viewpoints:
            print("No valid viewpoints found!")
        else:
            # Render from viewpoints
            print("Rendering from viewpoints...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_out = cv2.VideoWriter(f"{video_folder}/{video_name}_{request}.mp4", fourcc, 10.0, (width, height*2))
            
            # Now render and log new viewpoints
            print("Rendering and logging new viewpoints...")
            for i, pos in enumerate(tqdm(valid_viewpoints)):
                # Compute camera pose
                camera_pose = compute_camera_pose(pos, object_center)
                world_view_transform2 = camera_pose
                
                # Render
                cam = MiniCam(width, height, fovx, fovy, world_view_transform2)
                render_result = render(cam, gaussians, background)
                
                rendered_image = render_result["rendered_image"].permute(1, 2, 0).detach()
                rendered_lf = render_result["rendered_lf"].detach()
                rendered_image_pamr = rendered_image.permute(2, 0, 1).unsqueeze(0)
                
                dist = F.cosine_similarity(rendered_lf, text_embed, dim=0).detach()
                dist_pamr = dist.unsqueeze(0).unsqueeze(0)
                dist_pamr = dino_model.apply_pamr(rendered_image_pamr, dist_pamr)
                dist = dist_pamr.squeeze(0).squeeze(0)
                
                # Process results
                rgb = rendered_image.cpu().numpy()
                dist = dist.cpu().numpy()
                bboxes, binary_mask = find_bboxes(dist, threshold=0.5)
                
                # Convert to uint8 for Rerun
                rgb_uint8 = np.uint8(np.clip(rgb * 255, 0, 255))
                dist_uint8 = np.uint8(np.clip(dist * 255, 0, 255))
                mask_uint8 = np.uint8(binary_mask)
                
                # Draw results for video
                dist_colored = cv2.applyColorMap(dist_uint8, cv2.COLORMAP_JET)
                bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
                combined = np.vstack([bgr, dist_colored])
                
                video_out.write(combined)
                
                # Log to Rerun
                print(f"logging new view {i}")
                rr.log("camera/image", rr.Image(rgb_uint8, color_model="RGB"))
                rr.log("camera/lf_dist", rr.DepthImage(dist, depth_range=(0, 1)))
                rr.log("camera/binary_mask", rr.DepthImage(binary_mask, depth_range=(0, 1)))
                if best_semantic_mask is not None:
                    best_semantic_uint8 = np.uint8(np.clip(best_semantic_mask * 255, 0, 255))
                    rr.log("camera/best_semantic_mask", rr.DepthImage(best_semantic_mask, depth_range=(0, 1)))
            
            video_out.release()
    
    # Log all original views
    print("Logging original views to Rerun...")
    for idx in range(rendered_images.shape[0]):
        rgb = rendered_images[idx].copy()
        dist = rendered_lang_feats_dist[idx]
        
        bboxes, binary_mask = find_bboxes(dist, threshold=0.5)
        
        # Convert to uint8 for Rerun
        rgb_uint8 = np.uint8(rgb)
        dist_uint8 = np.uint8(dist * 255)
        mask_uint8 = np.uint8(binary_mask)
        
        print(f"logging original view {idx}")
        rr.log("camera/image", rr.Image(rgb_uint8, color_model="RGB"))
        rr.log("camera/lf_dist", rr.DepthImage(dist, depth_range=(0, 1)))
        rr.log("camera/binary_mask", rr.DepthImage(binary_mask, depth_range=(0, 1)))
        if best_semantic_mask is not None:
            best_semantic_uint8 = np.uint8(np.clip(best_semantic_mask * 255, 0, 255))
            rr.log("camera/best_semantic_mask", rr.DepthImage(best_semantic_mask, depth_range=(0, 1)))

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizes the LEGS-SLAM scene using Rerun SDK")
    parser.add_argument("--scene_path", type=Path, required=True)
    parser.add_argument("--debug_prompt", type=str, required=True)
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=8, help="Number of viewpoints to render")
    
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
        rrb.Horizontal(
            rrb.Spatial2DView(name="Current Binary Mask", origin="/camera/binary_mask"),
            rrb.Spatial2DView(name="Best Semantic Mask", origin="/camera/best_semantic_mask"),
        ),
        row_shares=[1, 1],
    )

    rr.script_setup(args, "rerun_example_structure_from_motion_", default_blueprint=blueprint)
    log_gaussian(args.scene_path, text_emb_compressed, dino_model, video_folder, args.video_name, args.debug_prompt, args.num_views)
    rr.script_teardown(args)

if __name__ == "__main__":
    main()
