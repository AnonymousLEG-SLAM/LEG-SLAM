import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf
from onnxruntime import InferenceSession
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from utils import MiniCam, build_text_embedding, focal2fov, get_world2view
sys.path.insert(0, "eval/open_vocabulary_segmentation")
from gaussian_model import GaussianModel
from models import build_model
from render import render
import metric_utils
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_label_mapping(filename, label_from="id", label_to="nyu40id"):
    """Read label mapping from file and convert labels to specified format."""
    assert os.path.isfile(filename)
    mapping = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])

    def represents_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping

def create_scene_model(scene_name):
    """Create Gaussian model and load camera parameters for a scene."""
    scene_path = f"results/{EXP_NAME}/{scene_name}/"
    sh_degree = 3
    print("Reading Gaussian Splatting reconstruction...")
    gaussians = GaussianModel(sh_degree=sh_degree)
    ply_path = os.path.join(scene_path, "experiment", "ply", "point_cloud", "point_cloud.ply")
    gaussians.load_ply(ply_path)

    with open(os.path.join(scene_path, "experiment", "ply", "cameras.json"), "r") as fin:
        camera_params = json.load(fin)

    return gaussians, camera_params

def train_scenes(eval_scannet_scenes_path, test_scenes):
    training_times = []
    training_fps = []
    for scene_name in test_scenes:
        scene_path = os.path.join(eval_scannet_scenes_path, scene_name)
        if not os.path.isdir(scene_path):
            print(f"Scene {scene_name} does not exist")
            continue
        
        training_time, fps = train_model_on_scene(eval_scannet_scenes_path, scene_name)
        
        with open("eval_result.log", "a") as f:
            if training_time is not None:
                f.write(f"{scene_name}: {training_time} seconds, {fps} fps\n")
                training_times.append(training_time)
                training_fps.append(fps)
            else:   
                f.write(f"{scene_name}: failed\n")
    
    if training_times:
        mean_time = sum(training_times) / len(training_times)
        mean_fps = sum(training_fps) / len(training_fps)
        print(f"Mean training time: {mean_time:.2f} seconds, {mean_fps:.2f} fps")
        with open("eval_result.log", "a") as f:
            f.write(f"Mean training time: {mean_time:.2f} seconds, {mean_fps:.2f} fps\n")

def train_model_on_scene(eval_scannet_scenes_path, scene_name):
    """Train model on a single scene."""
    print(f"Training model on scene {scene_name}")
    
    out_dir = f"results/{EXP_NAME}/{scene_name}/"
    if os.path.exists(out_dir):
        print(f"Scene {scene_name} already exists")
        return None, None
    
    command = [
        "./bin/replica_rgbd",
        "./ORB-SLAM3/Vocabulary/ORBvoc.txt", 
        f"./cfg/ORB_SLAM3/RGB-D/ScanNet/{ORB_SLAM_CONFIG}.yaml",
        f"./cfg/encoder/{ENCODER_CONFIG}.yaml",
        f"./cfg/gaussian_mapper/RGB-D/ScanNet/{GMAPPER_CONFIG}.yaml",
        os.path.join(eval_scannet_scenes_path, scene_name),
        out_dir
    ]
    
    process = subprocess.Popen(" ".join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    training_time = None
    fps = None
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
            if "Average FPS:" in line:
                fps = float(line.split(' ')[2].strip())
            elif "Total time:" in line:
                training_time = float(line.split(' ')[2].strip())
        time.sleep(0.01)
    
        
    return training_time, fps

def calculate_metrics(gt_img, pred_img, loss_fn_alex):
    """Calculate PSNR, SSIM and LPIPS between two images.
    
    Args:
        gt_img, pred_img: uint8 images with values in range [0, 255]
    Returns:
        tuple: PSNR (dB), SSIM, LPIPS values
    """
    # PSNR
    gt_img_f = gt_img.astype(np.float32)
    pred_img_f = pred_img.astype(np.float32)
    mse = np.mean((gt_img_f - pred_img_f) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
    # SSIM
    ssim_score = ssim(gt_img, pred_img, channel_axis=2, data_range=255)
    
    # LPIPS
    gt_tensor = torch.from_numpy(gt_img).permute(2,0,1).unsqueeze(0).float() / 255.0
    pred_tensor = torch.from_numpy(pred_img).permute(2,0,1).unsqueeze(0).float() / 255.0
    gt_tensor = gt_tensor.to(device)
    pred_tensor = pred_tensor.to(device)
    lpips_score = loss_fn_alex(gt_tensor, pred_tensor).item()
    
    return psnr, ssim_score, lpips_score

def refine_label_image(label_tensor, min_area=5000, max_area=10000):
    """
    Refine a label image by removing small connected components while preserving the original labels.
    Also fills empty zones with the most common neighboring label.

    Parameters:
    - label_tensor: PyTorch tensor of shape (H, W), containing label assignments.
    - min_area: Minimum area threshold for keeping connected components.

    Returns:
    - refined_label_tensor: PyTorch tensor with small noisy regions removed and empty zones filled.
    """
    # Convert PyTorch tensor to NumPy array
    label_np = label_tensor.cpu().numpy().astype(np.int32)  # Keep original labels

    # Find connected components (excluding background)
    unique_labels = np.unique(label_np)
    refined_label = np.zeros_like(label_np)

    for label in unique_labels:
        if label == 0:  # Skip background
            continue

        # Create a binary mask for the current label
        binary_mask = (label_np == label).astype(np.uint8)

        # Find connected components for the specific label
        num_components, components, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        for i in range(1, num_components):  # Ignore background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                refined_label[components == i] = label  # Preserve original label

    # Fill empty zones by finding which label region they belong to
    empty_mask = (refined_label == 0)
    kernel = np.ones((3,3), np.uint8)
    
    # First find all connected empty regions
    num_empty, empty_components = cv2.connectedComponents(empty_mask.astype(np.uint8), connectivity=8)
    
    # For each empty region, find the label it's contained within
    for i in range(1, num_empty):
        curr_empty = (empty_components == i)
        area = np.sum(curr_empty)
        
        if area < max_area:
            # Dilate the empty region slightly to find neighboring labels
            dilated_empty = cv2.dilate(curr_empty.astype(np.uint8), kernel, iterations=1)
            neighbor_labels = refined_label[dilated_empty.astype(bool)]
            
            # Get most common non-zero neighbor label
            neighbor_labels = neighbor_labels[neighbor_labels != 0]
            if len(neighbor_labels) > 0:
                most_common = np.bincount(neighbor_labels).argmax()
                refined_label[curr_empty] = most_common

    # Convert back to PyTorch tensor
    refined_label_tensor = torch.from_numpy(refined_label).to(label_tensor.device)

    return refined_label_tensor

def evaluate_scenes(pca_model_path, label_file, eval_scannet_scenes_path, test_scenes):
    """Main evaluation function."""    
    cfg = OmegaConf.load('eval/open_vocabulary_segmentation/talk2dino.yml')
    dino_model = build_model(cfg.model)
    dino_model.to(device).eval()
    
    pca_session = InferenceSession(pca_model_path)
    
    """Evaluate scenes and generate label maps."""
    print(f"Evaluating scene dir {eval_scannet_scenes_path}")
    
    label_mapping = read_label_mapping(label_file, label_to="cocomapid")
    confusion = np.zeros((NUM_CLASSES + 1, NUM_CLASSES), dtype=np.ulonglong)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    palette, labelset = metric_utils.get_text_requests(DATASET_TYPE)
    text_emb_compressed = build_text_embedding(labelset, dino_model, pca_session, device=device)
    text_emb_compressed = text_emb_compressed[..., None].cuda()
    failed_scenes = []
    all_scenes_metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'mean_iou': [], 'mean_acc': []}
    loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    
    for scene_name in tqdm(test_scenes):
        print(f"Processing scene {scene_name}")
        confusion_scene = np.zeros((NUM_CLASSES + 1, NUM_CLASSES), dtype=np.ulonglong)
        scene_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            try:
                gaussians, camera_params = create_scene_model(scene_name)
            except FileNotFoundError as e:
                print(f"Scene {scene_name} does not exist: {e}")
                failed_scenes.append(scene_name)
                continue
            
            fovx = focal2fov(camera_params[0]["fx"], camera_params[0]["width"])
            fovy = focal2fov(camera_params[0]["fy"], camera_params[0]["height"])

            out_path = os.path.join(BASE_OUT_PATH, scene_name)
            os.makedirs(os.path.join(out_path, "labelmap_gt"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "labelmap_pred"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "color_gt"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "color_pred"), exist_ok=True)

            for i, camera_param in enumerate(tqdm(camera_params)):
                if i % EVERY_N_FRAME != 0:
                    continue
                
                world_view_transform2 = get_world2view(
                    np.array(camera_param["rotation"]), 
                    np.array(camera_param["position"])
                )

                cam = MiniCam(
                    camera_param["width"], 
                    camera_param["height"],
                    fovx, fovy, 
                    world_view_transform2
                )
                render_result = render(cam, gaussians, background)
                
                rendered_image = render_result["rendered_image"].permute(1, 2, 0).detach()
                rendered_lf = render_result["rendered_lf"].detach()
                
                rendered_lf = rendered_lf.permute(1, 2, 0)
                text_emb = text_emb_compressed.squeeze(-1).squeeze(0).squeeze(0)

                rendered_lf_reshaped = rendered_lf.unsqueeze(-2)
                text_emb_reshaped = text_emb.unsqueeze(0).unsqueeze(0)                
                dist = F.cosine_similarity(
                    rendered_lf_reshaped, 
                    text_emb_reshaped, 
                    dim=-1
                )
                
                cos_sim = (1 - dist) / 2
                # cos_sim = (1 - dist)
                label = torch.argmax(cos_sim, dim=2)
                max_cos_sim = torch.max(cos_sim, dim=2)[0]  # Get maximum cosine similarity values
                label[max_cos_sim < 0.7] = 0  # Set label to 0 where max cosine similarity is < 0.7
                # label[max_cos_sim < 0.8] = 0  # Set label to 0 where max cosine similarity is < 0.7
                # label = refine_label_image(label)
                
                gt_path = str(camera_param["img_name"])
                label_img = metric_utils.get_mapped_label(
                    camera_param["height"], 
                    camera_param["width"], 
                    gt_path, 
                    label_mapping
                )
                label_img = torch.from_numpy(label_img).int().cpu()

                sem_gt = metric_utils.render_palette(label_img, palette)
                sem = metric_utils.render_palette(label, palette)
                
                image_idx = os.path.basename(gt_path).split(".")[0]
                torchvision.utils.save_image(
                    sem, os.path.join(out_path, "labelmap_pred", f"{image_idx}.jpg")
                )
                torchvision.utils.save_image(
                    sem_gt, os.path.join(out_path, "labelmap_gt", f"{image_idx}.jpg")
                )
                
                shutil.copy(
                    gt_path, os.path.join(out_path, "color_gt", f"{image_idx}.jpg")
                )
                rendered_image_rgb = (rendered_image.cpu().numpy() * 255)
                rendered_image_rgb = np.uint8(np.clip(rendered_image_rgb, 0, 255))
                cv2.imwrite(
                    os.path.join(out_path, "color_pred", f"{image_idx}.png"),
                    cv2.cvtColor(rendered_image_rgb, cv2.COLOR_RGB2BGR)
                )
                
                gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
                pred_img = rendered_image_rgb
                psnr, ssim_score, lpips_score = calculate_metrics(gt_img, pred_img, loss_fn_alex)
                scene_metrics['psnr'].append(psnr)
                scene_metrics['ssim'].append(ssim_score)
                scene_metrics['lpips'].append(lpips_score)
                
                confusion += metric_utils.confusion_matrix(
                    label.cpu().numpy().reshape(-1), 
                    label_img.cpu().numpy().reshape(-1), 
                    NUM_CLASSES
                )
                confusion_scene += metric_utils.confusion_matrix(
                    label.cpu().numpy().reshape(-1),
                    label_img.cpu().numpy().reshape(-1), 
                    NUM_CLASSES
                )
        
        scene_mean_iou, scene_mean_acc = metric_utils.evaluate_confusion(scene_name, confusion_scene, stdout=True, dataset=DATASET_TYPE)
        
        avg_metrics = {k: np.mean(v) for k,v in scene_metrics.items()}
        for k,v in avg_metrics.items():
            all_scenes_metrics[k].append(v)
        all_scenes_metrics['mean_iou'].append(scene_mean_iou)
        all_scenes_metrics['mean_acc'].append(scene_mean_acc)
        
        with open("eval_result.log", "a") as f:
            print(f"Scene {scene_name} - PSNR: {avg_metrics['psnr']:.2f} dB, SSIM: {avg_metrics['ssim']:.4f}, LPIPS: {avg_metrics['lpips']:.4f}")
            f.write(f"Scene {scene_name} - PSNR: {avg_metrics['psnr']:.2f} dB, SSIM: {avg_metrics['ssim']:.4f}, LPIPS: {avg_metrics['lpips']:.4f}\n")
        
    metric_utils.evaluate_confusion('Average', confusion, stdout=True, dataset=DATASET_TYPE)
    
    overall_metrics = {k: np.mean(v) for k,v in all_scenes_metrics.items()}
    print(f"Average Rendering metrics across all scenes - PSNR: {overall_metrics['psnr']:.2f} dB, SSIM: {overall_metrics['ssim']:.4f}, LPIPS: {overall_metrics['lpips']:.4f}")
    with open("eval_result.log", "a") as f:
        f.write(f"Average Rendering metrics across all scenes - PSNR: {overall_metrics['psnr']:.2f} dB, SSIM: {overall_metrics['ssim']:.4f}, LPIPS: {overall_metrics['lpips']:.4f}\n")
    
    if failed_scenes:
        print(f"Failed scenes: {failed_scenes}")
    
    print('Summary of all scenes:')
    with open("eval_result.log", "a") as f:
        for idx in range(len(test_scenes)):
            print(f"{test_scenes[idx]}: "\
                f"PSNR: {all_scenes_metrics['psnr'][idx]:.2f} dB, "\
                f"SSIM: {all_scenes_metrics['ssim'][idx]:.4f}, "\
                f"LPIPS: {all_scenes_metrics['lpips'][idx]:.4f}, "\
                f"mIoU: {all_scenes_metrics['mean_iou'][idx]:.4f}, "\
                f"mAcc: {all_scenes_metrics['mean_acc'][idx]:.4f}")
            f.write(f"{test_scenes[idx]}: "\
                f"PSNR: {all_scenes_metrics['psnr'][idx]:.2f} dB, "\
                f"SSIM: {all_scenes_metrics['ssim'][idx]:.4f}, "\
                f"LPIPS: {all_scenes_metrics['lpips'][idx]:.4f}, "\
                f"mIoU: {all_scenes_metrics['mean_iou'][idx]:.4f}, "\
                f"mAcc: {all_scenes_metrics['mean_acc'][idx]:.4f}\n")

def create_comparison_video(scene_name, out_path):
    """Create video comparing labelmap and ground truth predictions in a 2x2 grid."""    
    labelmap_pred_dir = os.path.join(out_path, "labelmap_pred")
    labelmap_gt_dir = os.path.join(out_path, "labelmap_gt")
    color_pred_dir = os.path.join(out_path, "color_pred") 
    color_gt_dir = os.path.join(out_path, "color_gt")
    
    image_files = sorted(
        [f for f in os.listdir(labelmap_pred_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(x.split('.')[0])
    )

    if not image_files:
        print(f"No images found in {labelmap_pred_dir}")
        return
        
    first_img = cv2.imread(os.path.join(labelmap_pred_dir, image_files[0]))
    height, width = first_img.shape[:2]
    
    legend_width = 150
    output_width = width + legend_width
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(out_path, f"comparison_{scene_name}.mp4")
    overlay_video_path = os.path.join(out_path, f"overlay_{scene_name}.mp4")
    
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (width*2, height*2))
    overlay_out = cv2.VideoWriter(overlay_video_path, fourcc, 10.0, (output_width, height))
    palette, labelset = metric_utils.get_text_requests(DATASET_TYPE)
    
    for img_file in image_files:
        labelmap = cv2.imread(os.path.join(labelmap_pred_dir, img_file))
        gt_label = cv2.imread(os.path.join(labelmap_gt_dir, img_file))
        
        color_pred = cv2.imread(os.path.join(color_pred_dir, f"{img_file.split('.')[0]}.png"))
        color_gt = cv2.imread(os.path.join(color_gt_dir, img_file))
        
        overlay = color_pred.copy()
        cv2.addWeighted(labelmap, 0.5, overlay, 0.5, 0, overlay)
        
        legend = np.ones((height, legend_width, 3), dtype=np.uint8) * 255
        y_offset = 30
        for i, label in enumerate(labelset):
            color_bgr = [int(c) for c in palette[3*i:3*i+3]][::-1]
            cv2.rectangle(legend, (20, y_offset + i*15), (40, y_offset + i*15 + 12), 
                         color_bgr, -1)
            cv2.putText(legend, label, (50, y_offset + i*15 + 9),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            
        overlay_with_legend = np.hstack((overlay, legend))
        
        top = np.hstack((color_gt, color_pred))
        bottom = np.hstack((gt_label, labelmap))
        combined = np.vstack((top, bottom))
        
        out.write(combined)
        overlay_out.write(overlay_with_legend)
        
    out.release()
    overlay_out.release()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_config", type=str, default="ckpts_text_scannet_20_ae_shallow")
    args = parser.parse_args()
    
    NUM_CLASSES = 20
    EVERY_N_FRAME = 1
    GMAPPER_CONFIG = "scannet"
    ORB_SLAM_CONFIG = "scannet"
    ENCODER_CONFIG = args.encoder_config
    EXP_SUFFIX = "_ae_shallow"
    DATASET_TYPE = "cocomap"
    EXP_NAME = f"ScanNet_eval_{ORB_SLAM_CONFIG}_{GMAPPER_CONFIG}_{ENCODER_CONFIG}"
    BASE_OUT_PATH = f"eval_render_langsplat_every{EVERY_N_FRAME}_{DATASET_TYPE}_{EXP_NAME}{EXP_SUFFIX}"
    
    label_file = "eval/scannetv2-labels.modified.tsv"
    eval_scannet_scenes_path = "path"
    test_scenes = [
        "scene0050_02", "scene0144_01", "scene0221_01", "scene0300_01", 
        "scene0354_00", "scene0389_00", "scene0423_02", "scene0427_00", 
        "scene0494_00", "scene0616_00", "scene0645_02", "scene0693_00"
    ]
    
    with open(f'cfg/encoder/{ENCODER_CONFIG}.yaml', 'r') as f:
        f.readline()  # Skip YAML header
        cfg = OmegaConf.load(f)
    pca_model_path = cfg["PixelwiseCompressor.Path"]
    
    set_seed(seed=155)
    
    for filename in ["eval_result.log"]:
        if os.path.exists(filename):
            os.remove(filename)
    
    with open("eval_result.log", "w") as f:
        f.write(f"EXP_NAME: {EXP_NAME}\n")
        f.write(f"DATASET_TYPE: {DATASET_TYPE}\n") 
        f.write(f"ORB_SLAM_CONFIG: {ORB_SLAM_CONFIG}\n")
        f.write(f"ENCODER_CONFIG: {ENCODER_CONFIG}\n")
        f.write(f"GMAPPER_CONFIG: {GMAPPER_CONFIG}\n")
        f.write(f"pca_model_path: {pca_model_path}\n")
        f.write(f"label_file: {label_file}\n")
        f.write(f"eval_scannet_scenes_path: {eval_scannet_scenes_path}\n")
    
    train_scenes(eval_scannet_scenes_path, test_scenes)
    evaluate_scenes(pca_model_path, label_file, eval_scannet_scenes_path, test_scenes)
    
    shutil.move("eval_result.log", os.path.join(BASE_OUT_PATH, f"eval_result.log"))

    # Create comparison videos for each scene
    for scene_name in tqdm(os.listdir(BASE_OUT_PATH), desc="Creating comparison videos"):
        scene_path = os.path.join(BASE_OUT_PATH, scene_name)
        if os.path.isdir(scene_path):
            create_comparison_video(scene_name, scene_path)
