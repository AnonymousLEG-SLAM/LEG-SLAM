from __future__ import annotations

import json
from pathlib import Path
import os
from datetime import datetime
from typing import Optional

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
import math

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from gaussian_model import GaussianModel
from render import render
from utils import MiniCam, focal2fov, get_world2view

import sys
from omegaconf import OmegaConf
from onnxruntime import InferenceSession
import cv2

from sklearn.cluster import DBSCAN

sys.path.insert(0, "eval/open_vocabulary_segmentation")
from models import build_model
from tqdm import tqdm

from utils import build_text_embedding

device = "cuda" if torch.cuda.is_available() else "cpu"

dino_model = None
pca_session = None
cfg = None

app = FastAPI(title="LEGS-SLAM Object Finder API", version="1.0.0")

class ObjectFinderRequest(BaseModel):
    scene_path: str
    prompt: str
    encoder_path: Optional[str] = "cfg/encoder/pca_encoder_scannet.yaml"
    use_rerun: Optional[bool] = False
    visualize_trajectory: Optional[bool] = False

class LegsSlamRequest(BaseModel):
    vocabulary_path: str = "./ORB-SLAM3/Vocabulary/ORBvoc.txt"
    orb_settings_path: str = "./cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml"
    encoder_settings_path: str = "./cfg/encoder/pca_encoder_scannet.yaml"
    gaussian_settings_path: str = "./cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
    sequence_path: str
    output_path: str = "results/{SCENE_NAME}"

class ObjectFinderResponse(BaseModel):
    status: str
    video_paths: list[str]
    message: str

class LegsSlamResponse(BaseModel):
    status: str
    output_path: str
    message: str

def initialize_models():
    global dino_model, pca_session, cfg
    print("Initializing models...")
    cfg = OmegaConf.load('eval/open_vocabulary_segmentation/talk2dino.yml')
    dino_model = build_model(cfg.model)
    dino_model.to(device).eval()
    with open("cfg/encoder/pca_encoder_scannet.yaml", 'r') as f:
        f.readline()
        encoder_cfg = OmegaConf.load(f)
    pca_model_path = encoder_cfg["PixelwiseCompressor.Path"]
    pca_session = InferenceSession(pca_model_path)
    print("Models initialized successfully!")

def find_bboxes(dist_img: np.ndarray, threshold: float) -> list[tuple[int, int, int, int]]:
    scale = 30
    kernel = np.ones((scale, scale)) / (scale**2)
    dist_img_avg = cv2.filter2D(dist_img, -1, kernel)
    dist_img_combined = 0.5 * (dist_img_avg + dist_img)
    binary_mask = (dist_img_combined > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
    bboxes = [bbox for bbox in bboxes if bbox[2] > 20 and bbox[3] > 20]
    return [(x, y, x + w, y + h) for x, y, w, h in bboxes], binary_mask

def generate_spherical_trajectory(center: np.ndarray, radius: float, num_frames: int = 60, axis: str = 'z') -> list[dict]:
    cameras = []
    n_layers = 1000
    if axis == 'x':
        def permute_coords(x, y, z):
            return z, x, y
    elif axis == 'y':
        def permute_coords(x, y, z):
            return x, z, y
    else:
        def permute_coords(x, y, z):
            return x, y, z
    for i in range(1):
        theta = math.pi * 0.5
        for j in range(n_layers):
            phi = 2 * math.pi * j / n_layers
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            x, y, z = permute_coords(x, y, z)
            x += center[0]
            y += center[1]
            z += center[2]
            position = np.array([x, y, z])
            forward = center - position
            forward = forward / np.linalg.norm(forward)
            world_up = np.array([0, 1, 0])
            right = np.cross(forward, world_up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            rotation = np.column_stack([right, up, forward])
            cameras.append({
                "position": position.tolist(),
                "rotation": rotation.tolist()
            })
    return cameras

def render_gaussian_images(scene_path: Path, text_emb_compressed: torch.Tensor, video_folder: str, request: str, use_rerun: bool, visualize_trajectory: bool) -> list[str]:
    SEMANTIC_SIMILARITY_THRESHOLD = 0.94
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    sh_degree = 3
    print("Reading Gaussian Splatting reconstruction...")
    gaussians = GaussianModel(sh_degree=sh_degree)
    pointcloud_path = scene_path / "experiment" / "ply" / "point_cloud" / "point_cloud.ply"
    print(f"Loading pointcloud from {pointcloud_path}")
    gaussians.load_ply(pointcloud_path)
    video_name = scene_path.name + "_" + request + "_" + str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    video_paths = []
    gaussian_features = gaussians.language_features.squeeze().detach()
    print("gaussian_features.shape", gaussian_features.shape)
    gaussian_positions = gaussians.xyz.detach()
    text_embed = text_emb_compressed.cuda()
    text_embed = text_embed[0][..., None, None]
    gaussian_features_norm = F.normalize(gaussian_features, dim=1)
    text_embed_norm = F.normalize(text_embed[:,0], dim=0)
    similarities = torch.matmul(gaussian_features_norm, text_embed_norm)
    similarities = 1 - ((similarities - similarities.min()) / (similarities.max() - similarities.min()))
    print("similarities.shape", similarities.shape)
    print("similarities.min(), similarities.max()", similarities.min(), similarities.max())
    red_color = torch.tensor([4.0, 0.0, 0.0], dtype=gaussians.features_dc.dtype, device=gaussians.features_dc.device)
    mask = similarities > SEMANTIC_SIMILARITY_THRESHOLD
    similarities_np = similarities.cpu().numpy()
    with open(scene_path / "experiment" / "ply" / "cameras.json", "r") as fin:
        camera_params = json.load(fin)
    width, height, fx, fy = camera_params[0]["width"], camera_params[0]["height"], camera_params[0]["fx"], camera_params[0]["fy"]
    print("width, height, fx, fy", width, height, fx, fy)
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    high_sim_mask = similarities_np > SEMANTIC_SIMILARITY_THRESHOLD
    high_sim_mask = high_sim_mask.squeeze()
    high_sim_points = gaussian_positions[high_sim_mask].cpu().numpy()
    if len(high_sim_points) == 0:
        print(f"No Gaussians found with similarity > {SEMANTIC_SIMILARITY_THRESHOLD}.")
        return video_paths
    else:
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
            return video_paths
        else:
            R = 0.1
            red_color = torch.tensor([4.0, 0.0, 0.0], dtype=gaussians.features_dc.dtype, device=gaussians.features_dc.device)
            gaussian_positions_np = gaussian_positions.cpu().numpy()
            for idx, center in enumerate(object_centers):
                print(f"Object {idx} 3D center: {center}")
                dists = np.linalg.norm(gaussian_positions_np - center, axis=1)
                mask = dists < R
                num_colored = np.sum(mask)
                gaussians_features_dc_orig = gaussians.features_dc.clone()
                if num_colored > 0:
                    gaussians.features_dc[mask] = red_color
                orbit_radius = 1.0
                orbit_cameras = generate_spherical_trajectory(center, orbit_radius, axis='y')
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
                    center_x, center_y = width // 2, height // 2
                    center_region = 15
                    depth = render_result["rendered_depth"].detach().cpu().numpy().squeeze()
                    center_depth = depth[
                        center_y - center_region:center_y + center_region,
                        center_x - center_region:center_x + center_region
                    ].mean()
                    rendered_image = render_result["rendered_image"].permute(1, 2, 0).detach().cpu().numpy()
                    rendered_image = np.uint8(np.clip(rendered_image * 255, 0, 255))
                    if center_depth < orbit_radius * 0.7:
                        pass
                    else:
                        bgr = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
                        video_out.write(bgr)
                        if use_rerun:
                            rr.log(f"object_0/orbit", rr.Image(rendered_image, color_model="RGB"))
                video_out.release()
                video_paths.append(video_path)
                print(f"Saved orbit video for object {idx} to {video_path}")
    if visualize_trajectory:
        rendered_images = []
        rendered_lang_feats_dist = []
        cameras = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        trajectory_video_path = f"{video_folder}/{video_name}_{request}_trajectory.mp4"
        video_out = cv2.VideoWriter(trajectory_video_path, fourcc, 10.0, (width, height*2))
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
            rgb = rendered_images[idx].copy()
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
        video_paths.append(trajectory_video_path)
        print(f"\nSemantic similarity statistics:")
        print(f"Mean similarity: {np.mean(similarities_np):.3f}")
        print(f"Max similarity: {np.max(similarities_np):.3f}")
        print(f"Min similarity: {np.min(similarities_np):.3f}")
        print(f"Number of Gaussians with similarity > {SEMANTIC_SIMILARITY_THRESHOLD}: {np.sum(similarities_np > SEMANTIC_SIMILARITY_THRESHOLD)}")
        print(f"Number of Gaussians with similarity > 0.5: {np.sum(similarities_np > 0.5)}")
    return video_paths

@app.on_event("startup")
async def startup_event():
    initialize_models()

@app.get("/")
async def root():
    return {"message": "LEGS-SLAM Object Finder API is running"}

@app.post("/find_objects", response_model=ObjectFinderResponse)
async def find_objects(request: ObjectFinderRequest):
    try:
        scene_name = request.scene_path.split("/")[-1]
        scene_path = Path('results') / scene_name
        if not scene_path.exists():
            raise HTTPException(status_code=404, detail=f"Scene path {request.scene_path} does not exist")
        video_folder = "ovs_videos"
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        text_emb_compressed = build_text_embedding([request.prompt], dino_model, pca_session)
        with torch.no_grad():
            video_paths = render_gaussian_images(
                scene_path, 
                text_emb_compressed, 
                video_folder, 
                request.prompt, 
                request.use_rerun, 
                request.visualize_trajectory
            )
        return ObjectFinderResponse(
            status="success",
            video_paths=video_paths,
            message=f"Found {len(video_paths)} video(s) for prompt: '{request.prompt}'"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": dino_model is not None and pca_session is not None}

@app.post("/run_legs_slam", response_model=LegsSlamResponse)
async def run_legs_slam(request: LegsSlamRequest):
    try:
        import subprocess
        import os
        os.makedirs(request.output_path, exist_ok=True)
        ply_path = os.path.join(
            request.output_path, "experiment", "ply", "point_cloud", "point_cloud.ply"
        )
        print(f"ply_path: {ply_path}")
        if os.path.exists(ply_path):
            print(f"Point cloud already exists at: {ply_path}")
            return LegsSlamResponse(
                status="success",
                output_path=request.output_path,
                message=f"LEGS-SLAM output already exists at {ply_path}"
            )
        cmd = [
            "./bin/replica_rgbd",
            request.vocabulary_path,
            request.orb_settings_path,
            request.encoder_settings_path,
            request.gaussian_settings_path,
            request.sequence_path,
            request.output_path,
            'no_viewer'
        ]
        print(f"Running LEGS-SLAM command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=".",
            capture_output=True,
            text=True,
            timeout=3600
        )
        print(f"result: {result}")
        if result.returncode == 0:
            return LegsSlamResponse(
                status="success",
                output_path=request.output_path,
                message=f"LEGS-SLAM completed successfully. Output saved to {request.output_path}"
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"LEGS-SLAM failed with return code {result.returncode}. Error: {result.stderr}"
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="LEGS-SLAM processing timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running LEGS-SLAM: {str(e)}")

if __name__ == "__main__":
    initialize_models()
    uvicorn.run(app, host="0.0.0.0", port=8005)
