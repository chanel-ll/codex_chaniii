"""Camera parameter loading from NeRF / D-NeRF transforms.json format."""

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class Camera:
    c2w: torch.Tensor     # [4, 4] camera-to-world transform
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    image_path: Optional[str] = None
    time: float = 0.0

    @property
    def fovx(self) -> float:
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def fovy(self) -> float:
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    @property
    def K(self) -> torch.Tensor:
        return torch.tensor([
            [self.fx,    0.0, self.cx],
            [  0.0,  self.fy, self.cy],
            [  0.0,     0.0,    1.0],
        ], dtype=torch.float32)

    @property
    def world_to_cam(self) -> torch.Tensor:
        """Invert c2w to get viewmat [4, 4]."""
        return torch.linalg.inv(self.c2w)


def load_cameras_from_transforms(
        json_path: str,
        image_root: str = None,
        width: int = 800,
        height: int = 800,
        device: str = "cpu",
) -> List[Camera]:
    """
    Load cameras from a NeRF/D-NeRF transforms.json file.

    Args:
        json_path:   Path to transforms_train.json (or transforms_test.json).
        image_root:  Optional root directory where rendered images are stored.
                     If None, image_path on each Camera will be None.
        width, height: Image resolution (default 800×800 for D-NeRF).
        device:      Device for tensors.

    Returns:
        List of Camera objects sorted by frame index / time.
    """
    with open(json_path) as f:
        meta = json.load(f)

    fovx = float(meta["camera_angle_x"])
    fx = 0.5 * width / math.tan(0.5 * fovx)
    fy = fx  # square pixels assumed
    cx = width  / 2.0
    cy = height / 2.0

    cameras = []
    for frame in meta["frames"]:
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)

        # D-NeRF uses OpenGL convention (y-up, -z forward). Convert to OpenCV (+y down, +z forward).
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1

        img_path = None
        if image_root is not None:
            base = frame.get("file_path", "")
            for ext in (".png", ".jpg", ".jpeg", ""):
                candidate = os.path.join(image_root, base + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        cameras.append(Camera(
            c2w=c2w.to(device),
            width=width, height=height,
            fx=fx, fy=fy, cx=cx, cy=cy,
            image_path=img_path,
            time=float(frame.get("time", 0.0)),
        ))

    # Sort by time
    cameras.sort(key=lambda c: c.time)
    return cameras


def load_gt_image(camera: Camera, device: str = "cpu") -> Optional[torch.Tensor]:
    """Load the GT image associated with a camera as [3, H, W] float in [0,1]."""
    if camera.image_path is None:
        return None
    try:
        from PIL import Image
        import numpy as np
        img = np.array(Image.open(camera.image_path).convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).to(device)
    except Exception:
        return None
