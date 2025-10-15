import cv2
import json
import numpy as np
import os
import random
import shutil
import subprocess
import time
import torch

from gaussian_model import GaussianModel
from render import render
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from utils import MiniCam, focal2fov, get_world2view

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_ate_rmse_and_mean(gt_poses, est_poses):
    """Calculate ATE RMSE and Mean between ground truth and estimated poses in centimeters.
    Aligns trajectories using the first pose as reference."""
    # Convert poses to 4x4 matrices
    gt_poses = np.array([np.reshape(pose, (4,4)) for pose in gt_poses])
    est_poses = np.array(est_poses)
    
    # Align trajectories using the first pose
    T_align = np.linalg.inv(est_poses[0]) @ gt_poses[0]
    est_poses_aligned = np.array([T_align @ pose for pose in est_poses])
    
    # Calculate translation error in centimeters
    trans_error = (gt_poses[:,:3,3] - est_poses_aligned[:,:3,3]) * 100  # Convert to cm
    
    # Standard ATE: treat each x,y,z component as individual measurement
    rmse = np.sqrt(np.mean(trans_error**2))
    mean = np.mean(np.abs(trans_error))  # Use absolute values for mean
    return rmse, mean

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

def train_scenes(eval_replica_scenes_path, test_scenes):
    fps_list = []
    training_time_list = []
    for scene_name in test_scenes:
        scene_path = os.path.join(eval_replica_scenes_path, scene_name)
        if not os.path.isdir(scene_path):
            print(f"Scene {scene_name} does not exist")
            continue
        
        training_time, fps = train_model_on_scene(eval_replica_scenes_path, scene_name)
        if fps is not None:
            fps_list.append(fps)
        training_time_list.append(training_time)
        
        with open("eval_result.log", "a") as f:
            if training_time is not None:
                f.write(f"{scene_name}: {training_time} seconds, {fps} fps\n")
            else:
                f.write(f"{scene_name}: failed\n")
                
    if len(fps_list) != 0:
        mean_fps = np.mean(fps_list)
        mean_training_time = np.mean(training_time_list)
        with open("eval_result.log", "a") as f:
            print(f"Mean FPS: {mean_fps:.2f}, Mean Training Time: {mean_training_time:.2f} seconds")
            f.write(f"Mean FPS: {mean_fps:.2f}, Mean Training Time: {mean_training_time:.2f} seconds\n")
        return mean_fps, mean_training_time
    else:
        return None, None

def train_model_on_scene(eval_replica_scenes_path, scene_name):
    """Train model on a single scene."""
    print(f"Training model on scene {scene_name}")
    
    out_dir = f"results/{EXP_NAME}/{scene_name}/"
    if os.path.exists(out_dir):
        print(f"Scene {scene_name} already exists")
        return None, None
    
    command = [
        "./bin/replica_rgbd",
        "./ORB-SLAM3/Vocabulary/ORBvoc.txt", 
        f"./cfg/ORB_SLAM3/RGB-D/Replica/{scene_name}.yaml",
        "./cfg/encoder/ckpts_text_scannet_20_ae_shallow.yaml",
        "./cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml",
        os.path.join(eval_replica_scenes_path, scene_name),
        out_dir
    ]
    
    process = subprocess.Popen(" ".join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    output = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
            output.append(line)
            if "Average FPS:" in line:
                fps = float(line.split(' ')[2].strip())
            elif "Total time:" in line:
                training_time = float(line.split(' ')[2].strip())
        time.sleep(0.05)
    
    output = "".join(output)
        
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

def calculate_depth_metrics(gt_depth, pred_depth):
    """Calculate depth metrics between two images.
    
    Args:
        gt_depth, pred_depth: uint16 images with values in range [0, 65535]
    Returns:
        float: L1 depth error in centimeters
    """
    # Convert to float and scale to meters
    gt_depth = gt_depth.astype(np.float32) / REPLICA_DEPTH_SCALE
    pred_depth = pred_depth.astype(np.float32) / REPLICA_DEPTH_SCALE
    
    # Create mask for valid depth values (non-zero and reasonable range)
    valid_mask = (gt_depth > 0.1) & (gt_depth < 10.0) & (pred_depth > 0.1) & (pred_depth < 10.0)
    
    if not np.any(valid_mask):
        return float('inf')  # Return infinity if no valid pixels
        
    # Calculate L1 error only on valid pixels
    l1_error = np.mean(np.abs(gt_depth[valid_mask] - pred_depth[valid_mask])) * 100
    
    return l1_error

def evaluate_scenes(eval_replica_scenes_path, test_scenes):
    """Main evaluation function."""
    print(f"Evaluating scene dir {eval_replica_scenes_path}")
    
    loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    failed_scenes = []
    all_scenes_metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'l1_error': [], 'ate_rmse': [], 'ate_mean': []}
    
    for scene_name in tqdm(test_scenes):
        print(f"Processing scene {scene_name}")
        scene_metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'l1_error': []}
        
        gt_poses_all = np.loadtxt(os.path.join(eval_replica_scenes_path, scene_name, "traj.txt"))
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            try:
                gaussians, camera_params = create_scene_model(scene_name)
            except FileNotFoundError as e:
                print(f"Scene {scene_name} does not exist: {e}")
                failed_scenes.append(scene_name)
                continue
            
            est_poses = []
            gt_poses = []
            for camera_param in camera_params:
                frame_id = int(os.path.basename(camera_param["img_name"]).replace("frame", "").replace(".jpg", ""))
                gt_poses.append(gt_poses_all[frame_id])
                
                R = np.array(camera_param["rotation"])
                t = np.array(camera_param["position"])
                T = np.eye(4)
                T[:3,:3] = R
                T[:3,3] = t
                est_poses.append(T)
            est_poses = np.array(est_poses)
            gt_poses = np.array(gt_poses)
            
            ate_rmse, ate_mean = calculate_ate_rmse_and_mean(gt_poses, est_poses)
            all_scenes_metrics['ate_rmse'].append(ate_rmse)
            all_scenes_metrics['ate_mean'].append(ate_mean)
            
            fovx = focal2fov(camera_params[0]["fx"], camera_params[0]["width"])
            fovy = focal2fov(camera_params[0]["fy"], camera_params[0]["height"])

            out_path = os.path.join(f"eval_render_{EXP_NAME}", scene_name)
            os.makedirs(os.path.join(out_path, "color_gt"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "color_pred"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "depth_pred"), exist_ok=True)

            for camera_param in tqdm(camera_params):
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
                rendered_depth = render_result["rendered_depth"].squeeze(0).detach()
                
                gt_path = str(camera_param["img_name"])
                image_idx = os.path.basename(gt_path).split(".")[0]
                shutil.copy(
                    gt_path, os.path.join(out_path, "color_gt", f"{image_idx}.jpg")
                )
                rendered_image_rgb = (rendered_image.cpu().numpy() * 255)
                rendered_image_rgb = np.uint8(np.clip(rendered_image_rgb, 0, 255))
                cv2.imwrite(
                    os.path.join(out_path, "color_pred", f"{image_idx}.png"),
                    cv2.cvtColor(rendered_image_rgb, cv2.COLOR_RGB2BGR)
                )
                gt_depth_path = gt_path.replace("frame", "depth").replace("jpg", "png")
                gt_depth_img = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
                rendered_depth_rgb = (rendered_depth.cpu().numpy() * REPLICA_DEPTH_SCALE)
                rendered_depth_rgb = np.uint16(np.clip(rendered_depth_rgb, 0, 65535))
                cv2.imwrite(
                    os.path.join(out_path, "depth_pred", f"{image_idx}.png"),
                    rendered_depth_rgb
                )
                l1_error = calculate_depth_metrics(gt_depth_img, rendered_depth_rgb)
                scene_metrics['l1_error'].append(l1_error)
                
                gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
                pred_img = rendered_image_rgb
                psnr, ssim_score, lpips_score = calculate_metrics(gt_img, pred_img, loss_fn_alex)
                scene_metrics['psnr'].append(psnr)
                scene_metrics['ssim'].append(ssim_score)
                scene_metrics['lpips'].append(lpips_score)
                break
        
        avg_metrics = {k: np.mean(v) for k,v in scene_metrics.items()}
        for k,v in avg_metrics.items():
            all_scenes_metrics[k].append(v)
            
        with open("eval_result.log", "a") as f:
            print(f"Scene {scene_name} - PSNR: {avg_metrics['psnr']:.2f} dB, SSIM: {avg_metrics['ssim']:.4f}, LPIPS: {avg_metrics['lpips']:.4f}, L1 Error: {avg_metrics['l1_error']:.2f} cm, ATE RMSE: {ate_rmse:.4f} cm, ATE Mean: {ate_mean:.4f} cm")
            f.write(f"Scene {scene_name} - PSNR: {avg_metrics['psnr']:.2f} dB, SSIM: {avg_metrics['ssim']:.4f}, LPIPS: {avg_metrics['lpips']:.4f}, L1 Error: {avg_metrics['l1_error']:.2f} cm, ATE RMSE: {ate_rmse:.4f} cm, ATE Mean: {ate_mean:.4f} cm\n")
    
    overall_metrics = {k: np.mean(v) for k,v in all_scenes_metrics.items()}
    print(f"Average metrics across all scenes - PSNR: {overall_metrics['psnr']:.2f} dB, SSIM: {overall_metrics['ssim']:.4f}, LPIPS: {overall_metrics['lpips']:.4f}, L1 Error: {overall_metrics['l1_error']:.2f} cm, ATE RMSE: {overall_metrics['ate_rmse']:.4f} cm, ATE Mean: {overall_metrics['ate_mean']:.4f} cm")
    with open("eval_result.log", "a") as f:
        f.write(f"Average metrics across all scenes - PSNR: {overall_metrics['psnr']:.2f} dB, SSIM: {overall_metrics['ssim']:.4f}, LPIPS: {overall_metrics['lpips']:.4f}, L1 Error: {overall_metrics['l1_error']:.2f} cm, ATE RMSE: {overall_metrics['ate_rmse']:.4f} cm, ATE Mean: {overall_metrics['ate_mean']:.4f} cm\n")
    
    if failed_scenes:
        print(f"Failed scenes: {failed_scenes}")
        
    print('Summary of all scenes:')
    with open("eval_result.log", "a") as f:
        for idx in range(len(test_scenes)):
            print(f"{test_scenes[idx]}: "\
                f"PSNR: {all_scenes_metrics['psnr'][idx]:.2f} dB, "\
                f"SSIM: {all_scenes_metrics['ssim'][idx]:.4f}, "\
                f"LPIPS: {all_scenes_metrics['lpips'][idx]:.4f}, "\
                f"L1 Error: {all_scenes_metrics['l1_error'][idx]:.2f} cm, "\
                f"ATE RMSE: {all_scenes_metrics['ate_rmse'][idx]:.4f} cm, "\
                f"ATE Mean: {all_scenes_metrics['ate_mean'][idx]:.4f} cm")
            f.write(f"{test_scenes[idx]}: "\
                f"PSNR: {all_scenes_metrics['psnr'][idx]:.2f} dB, "\
                f"SSIM: {all_scenes_metrics['ssim'][idx]:.4f}, "\
                f"LPIPS: {all_scenes_metrics['lpips'][idx]:.4f}, "\
                f"L1 Error: {all_scenes_metrics['l1_error'][idx]:.2f} cm, "\
                f"ATE RMSE: {all_scenes_metrics['ate_rmse'][idx]:.4f} cm, "\
                f"ATE Mean: {all_scenes_metrics['ate_mean'][idx]:.4f} cm\n")

if __name__ == "__main__":
    EXP_NAME = "Replica_eval_ae_shallow"
    
    eval_replica_scenes_path = "path"
    test_scenes = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]

    REPLICA_DEPTH_SCALE = 6553.5
    
    set_seed(seed=155)
    
    for filename in ["eval_result.log"]:
        if os.path.exists(filename):
            os.remove(filename)
    
    with open("eval_result.log", "w") as f:
        f.write(f"EXP_NAME: {EXP_NAME}\n")
        f.write(f"eval_replica_scenes_path: {eval_replica_scenes_path}\n")
    
    train_scenes(eval_replica_scenes_path, test_scenes)
    evaluate_scenes(eval_replica_scenes_path, test_scenes)
    
    shutil.move("eval_result.log", f"eval_result_{EXP_NAME}.log")
