import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "singapo"))

from generate import generate
from eval_data import EvaluationData
from utils.misc import load_config
from eval_utils.utils import normalize_mesh,render_mesh, project_texture
from eval_utils.render_compare import compare_to_ground_truth

import argparse
from tqdm import tqdm
import trimesh
import torch
from PIL import Image
import numpy as np

dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True).cuda()


class Args:
    def __init__(
        self,
        img='demo/demo_input.png',
        ckpt='exps/singapo/final/ckpts/last.ckpt',
        config='exps/singapo/final/config/parsed.yaml',
        use_example=False,
        out='demo/demo_output',
        gt_root='../data',
        n=1,
        omega=0.5,
        denoise_steps=100,
    ):
        self.img_path = img
        self.ckpt_path = ckpt
        self.config_path = config
        self.use_example_graph = use_example
        self.save_dir = out
        self.gt_data_root = gt_root
        self.n_samples = n
        self.omega = omega
        self.n_denoise_steps = denoise_steps


def synthesize_objects(data, args):
    """
    Synthesize objects using Singapo.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """

    print("Synthesizing with Singapo...")

    for item in tqdm(data.get_data_items()):
        out_dir = os.path.join(item.output_path, "singapo", "0")

        if args.use_cached and os.path.exists(os.path.join(out_dir, "object.obj")):
            item.set_singapo_obj_path(os.path.join(out_dir, "object.obj"))
            continue

        s_args = Args(
            img=item.img_path,
            ckpt=args.singapo_ckpt_path,
            config=args.singapo_config_path,
            gt_root=args.singapo_gt_data_root,
            out=os.path.join(item.output_path, "singapo"),
        )

        generate(s_args)

        mesh = trimesh.load(os.path.join(out_dir, "object.ply"), force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"{out_dir}/object.ply failed to load")
            continue

        normalize_mesh(mesh)
        mesh.export(os.path.join(out_dir, "object.obj"), file_type='obj', include_texture=False)
        item.set_singapo_obj_path(os.path.join(out_dir, "object.obj"))


def texture_objects(data, args):
    """
    Texture objects using Easi-Tex.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """

    print("Texturing with Easi-Tex...")

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "easi-tex")
    prev = os.getcwd()
    os.chdir(root)

    def get_cmd(item):
        return (
            f'python scripts/generate_texture.py '
            f'--input_dir "{os.path.dirname(item.singapo_obj_path)}" '
            f'--output_dir "{os.path.join(item.output_path, "easitex")}" '
            f'--obj_file "{os.path.basename(item.singapo_obj_path)}" '
            f'--prompt "{item.description}" '
            f'--style_img "{item.img_path}" '
            f'--style_img_bg_color 255 255 255 '
            f'--ip_adapter_path "./ip_adapter" '
            f'--ip_adapter_strength 1.0 '
            f'--ip_adapter_n_tokens 16 '
            f'--controlnet_cond "canny" '
            f'--controlnet_strength 1.0 '
            f'--use_cc_edges True '
            f'--use_depth_edges True '
            f'--use_normal_edges True '
            f'--add_view_to_prompt '
            f'--ddim_steps 50 '
            f'--guidance_scale 10 '
            f'--new_strength 1 '
            f'--update_strength 0.4 '
            f'--view_threshold 0.1 '
            f'--blend 0 '
            f'--dist 0.8 '
            f'--num_viewpoints 36 '
            f'--viewpoint_mode predefined '
            f'--use_principle '
            f'--update_steps 20 '
            f'--update_mode heuristic '
            f'--seed 42 '
            f'--post_process '
            f'--tex_resolution "1k" '
            f'--use_objaverse'
        )

    for item in tqdm(data.get_data_items()):
        tex_path = os.path.join(
            item.output_path, "easitex", "canny", f"0-{item.id}",
            "42-ip1.0-cn1.0-dist0.8-gs10.0-p36-h20-us0.4-vt0.1", "update", "mesh", "19_post.obj"
        )
        item.set_easitex_obj_path(tex_path)

        if args.use_cached and os.path.exists(tex_path):
            continue

        os.system(get_cmd(item))

    os.chdir(prev)


def evaluate(data, args):
    """
    Evaluate the synthesized objects.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """

    print("Evaluating...")

    for item in tqdm(data.get_data_items()):
        sim = compare_to_ground_truth(item.easitex_obj_path, item.obj_path, item.output_path, args)
        item.set_cosine_similarity(sim)

        if args.add_no_texture:
            out_no_tex = os.path.join(item.output_path, "no_easitex")
            os.makedirs(out_no_tex, exist_ok=True)
            sim = compare_to_ground_truth(item.singapo_obj_path, item.obj_path, out_no_tex, args)
            item.set_cosine_similarity_no_easitex(sim)

        if args.add_naive_texturing:
            out_naive = os.path.join(item.output_path, "naive_texturing")
            os.makedirs(out_naive, exist_ok=True)
            sim = compare_to_ground_truth(item.naive_texturing_path, item.obj_path, out_naive, args)
            item.set_naive_cosine_similarity(sim)


def display_results(data, args):
    """
    Display the results of the evaluation.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """


    total_sim = 0
    total_sim_no_tex = 0
    total_sim_naive = 0


    for item in data.get_data_items():
        if item.valid:
            print(f"Item ID: {item.id}")
            print(f"Cosine Similarity: {item.cosine_similarity}")
            print("-" * 20)
            total_sim += item.cosine_similarity
            if args.add_no_texture:
                print(f"Cosine Similarity (No Texture): {item.cosine_similarity_no_easitex}")
                total_sim_no_tex += item.cosine_similarity_no_easitex
                print("-" * 20)
            if args.add_naive_texturing:
                print(f"Cosine Similarity (Naive Projection): {item.naive_cosine_similarity}")
                total_sim_naive += item.naive_cosine_similarity
                print("-" * 20)


    
    print(f"Average Cosine Similarity: {total_sim / len(data.get_data_items())}")
    if args.add_no_texture:
        print(f"Average Cosine Similarity (No Texture): {total_sim_no_tex / len(data.get_data_items())}")
    if args.add_naive_texturing:
        print(f"Average Cosine Similarity (Naive Projection): {total_sim_naive / len(data.get_data_items())}")



def evaluate_meshes(data,args):
    """
    Evaluate the meshes.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """

    print("Evaluating meshes...")

    for item in tqdm(data.get_data_items()):

        similarity = compare_to_ground_truth(os.path.join(args.from_meshes,item.id,"mesh.obj"),item.obj_path,os.path.join(args.from_meshes,item.id),args)

        item.set_cosine_similarity(similarity)

def texture_naive(data, args):
    """
    Texture objects using naive projection.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """

    print("Texturing objects using naive projection...")

    for item in tqdm(data.get_data_items()):

        if args.use_cached and os.path.exists(item.naive_texturing_path):
            continue

        # Load the mesh
        mesh = trimesh.load(item.singapo_obj_path)
        project_texture(mesh, item.img_path, item.mask_path, item.naive_texturing_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--singapo_ckpt_path", type=str, default='singapo/exps/singapo/final/ckpts/last.ckpt', help="path to the checkpoint of the model")
    parser.add_argument("--singapo_config_path", type=str, default='singapo/exps/singapo/final/config/parsed.yaml', help="path to the config file")
    parser.add_argument("--singapo_gt_data_root", type=str, default='data', help="the root directory of the original data, used for part mesh retrieval")
    parser.add_argument("--eval_data_path", type=str, default='eval_data', help="path to the data to be evaluated")
    parser.add_argument("--output_path", type=str, default='output', help="path to save the output")
    parser.add_argument("--use_cached", action="store_true", help="whether to use cached objects")
    parser.add_argument("--add_no_texture", action="store_true", help="additionally evaluate the objects without any texturing")
    parser.add_argument("--add_naive_texturing", action="store_true", help="additionally evaluate the objects with naive texturing instead of Easi-Tex")
    parser.add_argument("--from_meshes", type=str, default=None, help="path to the meshes to be evaluated")
    parser.add_argument("--additional_rotations", action="store_true", help="evaluate the objects with additional rotations")

    args = parser.parse_args()

    #All paths used in singapo and easi-tex should be absolute
    args.output_path = os.path.abspath(args.output_path)
    args.singapo_ckpt_path = os.path.abspath(args.singapo_ckpt_path)
    args.singapo_config_path = os.path.abspath(args.singapo_config_path)
    args.singapo_gt_data_root = os.path.abspath(args.singapo_gt_data_root)

    data = EvaluationData(args.eval_data_path,args.output_path,args.use_cached)


    if args.from_meshes is None:
        synthesize_objects(data,args)
        texture_objects(data,args)
        if args.add_naive_texturing:
            texture_naive(data,args)
        evaluate(data,args)
        display_results(data,args)
    else:
        print("Evaluating meshes from", args.from_meshes)
        evaluate_meshes(data,args)
        display_results(data,args)








