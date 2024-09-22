from typing import Iterator
import more_itertools
from torch.utils.data import Dataset
from r3d_dataset import PosedRGBDItem
from torch import Tensor
import tqdm
import torch
import open3d as o3d

def get_inv_intrinsics(intrinsics: Tensor) -> Tensor:
    # return intrinsics.double().inverse().to(intrinsics)
    fx, fy, ppx, ppy = intrinsics[..., 0, 0], intrinsics[..., 1, 1], intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    inv_intrinsics = torch.zeros_like(intrinsics)
    inv_intrinsics[..., 0, 0] = 1.0 / fx
    inv_intrinsics[..., 1, 1] = 1.0 / fy
    inv_intrinsics[..., 0, 2] = -ppx / fx
    inv_intrinsics[..., 1, 2] = -ppy / fy
    inv_intrinsics[..., 2, 2] = 1.0
    return inv_intrinsics


def get_xyz(depth: Tensor, mask: Tensor, pose: Tensor, intrinsics: Tensor) -> Tensor:
    """Returns the XYZ coordinates for a set of points.

    Args:
        depth: The depth array, with shape (B, 1, H, W)
        mask: The mask array, with shape (B, 1, H, W)
        pose: The pose array, with shape (B, 4, 4)
        intrinsics: The intrinsics array, with shape (B, 3, 3)

    Returns:
        The XYZ coordinates of the projected points, with shape (B, H, W, 3)
    """

    (bsz, _, height, width), device, dtype = depth.shape, depth.device, intrinsics.dtype

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=device, dtype=dtype),
        torch.arange(0, height, device=device, dtype=dtype),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1).flatten(0, 1).unsqueeze(0).repeat_interleave(bsz, 0)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Applies intrinsics and extrinsics.
    # xyz = xyz @ intrinsics.inverse().transpose(-1, -2)
    xyz = xyz @ get_inv_intrinsics(intrinsics).transpose(-1, -2)
    xyz = xyz * depth.flatten(1).unsqueeze(-1)
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[..., None, :3, 3]

    # Mask out bad depth points.
    xyz = xyz.unflatten(1, (height, width))
    xyz[mask.squeeze(1)] = 0.0

    return xyz


def get_pointcloud(ds: Dataset[PosedRGBDItem], file_name: str, chunk_size: int = 16, threshold:float = 0.9) -> Iterator[tuple[Tensor, Tensor]]:
    """Iterates XYZ points from the dataset.

    Args:
        ds: The dataset to iterate points from
        desc: TQDM bar description
        chunk_size: Process this many frames from the dataset at a time

    Yields:
        The XYZ coordinates, with shape (B, H, W, 3), and a mask where a value
        of True means that the XYZ coordinates should be ignored at that
        point, with shape (B, H, W)
    """

    #device = torch.device('cuda') if torch.cuda.is_available() else torch.deivce('cpu')
    ds_len = len(ds)  # type: ignore
    xyzs = []
    rgbs = []

    for inds in more_itertools.chunked(tqdm.trange(ds_len, desc='point cloud'), chunk_size):
        rgb, depth, mask, pose, intrinsics = (
            torch.stack(ts, dim=0)
            for ts in zip(
                #*((t.to(device) for t in (i.image, i.depth, i.mask, i.pose, i.intrinsics)) for i in (ds[i] for i in inds))
                *((t for t in (i.image, i.depth, i.mask, i.pose, i.intrinsics)) for i in (ds[i] for i in inds))
            )
        )
        rgb = rgb.permute(0, 2, 3, 1)
        xyz = get_xyz(depth, mask, pose, intrinsics)
        #mask = (~mask & (torch.rand(mask.shape, device=mask.device) > threshold))
        mask = (~mask & (torch.rand(mask.shape) > threshold))
        # mask = (torch.rand(mask.shape) > threshold)
        rgb, xyz = rgb[mask.squeeze(1)], xyz[mask.squeeze(1)]
        rgbs.append(rgb.detach().cpu())
        xyzs.append(xyz.detach().cpu())

    xyzs = torch.vstack(xyzs)
    rgbs = torch.vstack(rgbs)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(xyzs)
    merged_pcd.colors = o3d.utility.Vector3dVector(rgbs)
    # merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.03)

    o3d.io.write_point_cloud(file_name, merged_pcd)