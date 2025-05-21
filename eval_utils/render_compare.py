from torchvision import transforms
from tqdm import tqdm
import trimesh
import torch
from PIL import Image
import numpy as np
import os

from eval_utils.utils import normalize_mesh,render_mesh, project_texture

dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True).cuda()


def _cosine_sim(a, b):
    """
    Compute the cosine similarity between two tensors.
    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
    Returns:
        torch.Tensor: Cosine similarity between the two tensors.
    """
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)
    return torch.sum(a * b, dim=-1)


def compare_to_ground_truth(obj_path, gt_path, out_dir, args):
    """
    Compare the synthesized object to the ground truth image using DINOv2.
    Args:
        obj_path (str): Path to the synthesized object.
        gt_path (str): Path to the ground truth object.
        out_dir (str): Directory, where to save the output image.
        args (argparse.Namespace): The arguments passed to the script.
    Returns:
        float: The cosine similarity between the synthesized object and the ground truth image.
            If args.additional_rotations is True, the average similarity over all rotations is returned.
    """


    tf = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tf_mask = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
    ])

    angles = [0, np.pi/2, np.pi, -np.pi/2] if args.additional_rotations else [0]
    sims = []

    for i, rot in enumerate(angles):
        os.makedirs(os.path.join(out_dir, "ground_truth"), exist_ok=True)
        gt_img = os.path.join(out_dir, "ground_truth", f"gt_{i}.png")
        gt_mask = os.path.join(out_dir, "ground_truth", f"gt_{i}_mask.png")

        gt_mesh = trimesh.load_mesh(gt_path)
        gt_mesh.apply_transform(trimesh.transformations.rotation_matrix(rot, [0, 1, 0]))
        render_mesh(gt_mesh, resolution=512, output_path=gt_img)

        gt_img_tensor = tf(Image.open(gt_img).convert("RGB"))
        gt_mask_tensor = tf_mask(Image.open(gt_mask).convert("L"))
        gt_tensor = gt_img_tensor * gt_mask_tensor

        syn_img = os.path.join(out_dir, f"syn_{i}.png")
        syn_mask = os.path.join(out_dir, f"syn_{i}_mask.png")

        syn_mesh = trimesh.load_mesh(obj_path)
        syn_mesh.apply_transform(trimesh.transformations.rotation_matrix(rot, [0, 1, 0]))
        render_mesh(syn_mesh, resolution=512, output_path=syn_img, is_instantmesh=bool(args.from_meshes))

        syn_img_tensor = tf(Image.open(syn_img).convert("RGB"))
        syn_mask_tensor = tf_mask(Image.open(syn_mask).convert("L"))
        syn_tensor = syn_img_tensor * syn_mask_tensor

        inp = torch.stack([syn_tensor, gt_tensor]).cuda()
        with torch.no_grad():
            emb = dinov2_vitb14_reg(inp)
        sims.append(_cosine_sim(emb[0], emb[1]).item())

    return sum(sims) / len(sims)