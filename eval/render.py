import typing as tp
import torch
import math
from diff_gaussian_rasterization_legs_slam import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_model import GaussianModel
from utils import MiniCam


def render(viewpoint_camera: MiniCam,
           pc: GaussianModel,
           bg_color: torch.Tensor,
           scaling_modifier: float = 1.0,
           override_color: torch.Tensor = None) -> dict[str, tp.Any]:
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        include_language_features=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.xyz
    means2D = screenspace_points
    opacity = pc.opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    scales = pc.scaling
    rotations = pc.rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.features
    else:
        colors_precomp = override_color

    lang_feats = pc.language_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_lf, rendered_depth, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        lang_feats=lang_feats,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"rendered_image": rendered_image,
            "rendered_lf": rendered_lf,
            "rendered_depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}