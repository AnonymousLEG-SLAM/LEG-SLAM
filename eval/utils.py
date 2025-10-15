import typing as tp
import numpy as np
import torch
import clip
from pathlib import Path
from scipy.spatial.transform import Rotation
from torchvision import transforms
import cv2

class MiniCam:
    def __init__(self, width: int, height: int, fovx: float, fovy: float, world_view_transform: np.ndarray) -> None:
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy
        self.z_near = 0.01
        self.z_far = 100.0
        self.world_view_transform = torch.tensor(world_view_transform, dtype=torch.float32).transpose(0, 1).cuda()
        self.projection_matrix = self.get_projection_matrix(z_near=self.z_near, z_far=self.z_far, fovx=self.FoVx,
                                                            fovy=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform \
                                       .unsqueeze(0) \
                                       .bmm(self.projection_matrix.unsqueeze(0)) \
                                       .squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def get_projection_matrix(z_near: float, z_far: float, fovx: float, fovy: float) -> torch.Tensor:
        tan_half_fovy = np.tan((fovy / 2))
        tan_half_fovx = np.tan((fovx / 2))

        top = tan_half_fovy * z_near
        bottom = -top
        right = tan_half_fovx * z_near
        left = -right

        proj_mtx = torch.zeros(4, 4)

        z_sign = 1.0

        proj_mtx[0, 0] = 2.0 * z_near / (right - left)
        proj_mtx[1, 1] = 2.0 * z_near / (top - bottom)
        proj_mtx[0, 2] = (right + left) / (right - left)
        proj_mtx[1, 2] = (top + bottom) / (top - bottom)
        proj_mtx[3, 2] = z_sign
        proj_mtx[2, 2] = z_sign * z_far / (z_far - z_near)
        proj_mtx[2, 3] = -(z_far * z_near) / (z_far - z_near)
        return proj_mtx


def load_poses(path: Path) -> tuple[np.ndarray, float]:
    poses = []

    pose_data = np.loadtxt(path, delimiter=' ', dtype=np.unicode_)
    pose_vecs = pose_data[:, 1:].astype(np.float32)
    tstamp = pose_data[:, 0].astype(np.float64)

    for pose_vec in pose_vecs:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pose_vec[3:]).as_matrix()
        pose[:3, 3] = pose_vec[:3]
        poses.append(pose)

    return poses, tstamp


def get_world2view(R: np.ndarray,
                   t: np.ndarray,
                   translate: np.ndarray = np.array([.0, .0, .0]),
                   scale: float = 1.0) -> np.ndarray:
    C2W = np.eye(4)
    C2W[:3, :3] = R
    C2W[:3, 3] = t

    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center

    W2C = np.linalg.inv(C2W)

    return np.float32(W2C)

def focal2fov(focal: float, pixels: int) -> float:
    return 2 * np.arctan(pixels / (2 * focal))


def build_text_embedding(categories, dino_model, pca_session, device="cuda"):
    """Build text embeddings for given categories."""
    tokens = []
    templates = [
        "itap of a {}.",
        "a bad photo of a {}.",
        "a origami {}.", 
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ]
    
    for category in categories:
        tokens.append(
            clip.tokenize([template.format(category) for template in templates])
        )
    tokens = torch.stack(tokens)
    text_emb = dino_model.build_text_embedding(tokens)
    
    text_emb_encoded = pca_session.run(None, {"input_feat": text_emb.cpu().numpy()})
    text_emb_encoded = text_emb_encoded[0]
    text_emb_compressed = torch.from_numpy(text_emb_encoded).to(device)
        
    return text_emb_compressed

def build_image_embedding(image_path, dino_model, pca_session, device="cuda"):
    """Build projected image embedding for a given image, similar to text projection."""
    # Load image
    image = cv2.imread(image_path)
    print(f"Image shape: {image.shape}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to torch tensor and normalize to [0,1]
    image = torch.from_numpy(image).float() / 255.0
    
    # Permute channels from HWC to CHW
    image = image.permute(2, 0, 1)
    
    # Add batch dimension
    image = image.unsqueeze(0)
    image = image.to(device)

    # Preprocess image for CLIP
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    image = preprocess(image)

    # Get image embedding from CLIP
    print(f"Image shape after preprocessing: {image.shape}")
    image_emb = dino_model.clip_model.encode_image(image)
    print(f"Image embedding shape: {image_emb.shape}")
    
    # Project image embedding if projection layer exists (like text)
    if hasattr(dino_model, "proj") and dino_model.proj is not None:
        # Try to use the same projection logic as in build_text_embedding
        # If CLIPLastLayer, pass through .proj as in text
        if type(dino_model.proj).__name__ == "CLIPLastLayer":
            # For CLIPLastLayer, pass self_attn_maps=None and ret_embeds=True
            image_emb, _ = dino_model.proj(
                image_emb, 
                None, 
                ret_embeds=True, 
                self_attn_maps=None
            )
        elif type(dino_model.proj).__name__ in ["ProjectionLayer", "DoubleMLP"]:
            image_emb = dino_model.proj.project_clip_txt(image_emb)
            print(f"Image embedding shape after projection: {image_emb.shape}")
        # else: do nothing (no projection)
    
    # Normalize as in build_text_embedding
    import us
    image_emb = us.normalize(image_emb, dim=-1).detach()

    # PCA compress
    image_emb_encoded = pca_session.run(None, {"input_feat": image_emb.cpu().numpy()})
    image_emb_encoded = image_emb_encoded[0]
    image_emb_compressed = torch.from_numpy(image_emb_encoded).to(device)

    return image_emb_compressed
