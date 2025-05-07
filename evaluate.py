import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "singapo"))

from generate import generate
from eval_data import EvaluationData
from utils.misc import load_config
from texture_singapo_utils import normalize_mesh

import argparse
from tqdm import tqdm
import trimesh



class SingapoArgs:
    def __init__(
        self,
        img_path: str = 'demo/demo_input.png',
        ckpt_path: str = 'exps/singapo/final/ckpts/last.ckpt',
        config_path: str = 'exps/singapo/final/config/parsed.yaml',
        use_example_graph: bool = False,
        save_dir: str = 'demo/demo_output',
        gt_data_root: str = '../data',
        n_samples: int = 1,
        omega: float = 0.5,
        n_denoise_steps: int = 100,
    ):
        self.img_path = img_path
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.use_example_graph = use_example_graph
        self.save_dir = save_dir
        self.gt_data_root = gt_data_root
        self.n_samples = n_samples
        self.omega = omega
        self.n_denoise_steps = n_denoise_steps

def synthesize_objects(data,args):
    """
    Synthesize objects using Singapo.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """

    print("Synthesising objects using Singapo...")

    for item in tqdm(data.get_data_items()):

        obj_dir = os.path.join(item.output_path,"singapo","0")

        if args.use_cached and os.path.exists(os.path.join(obj_dir,"object.obj")):
            item.set_singapo_obj_path(os.path.join(obj_dir,"object.obj"))
            continue

        singapo_args = SingapoArgs(
            img_path=item.img_path,
            ckpt_path=args.singapo_ckpt_path,
            config_path=args.singapo_config_path,
            gt_data_root=args.singapo_gt_data_root,
            save_dir=os.path.join(item.output_path, "singapo"),
        )

        generate(singapo_args)

        #convert ply to obj and normalize
        mesh = trimesh.load(os.path.join(obj_dir,"object.ply"), force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"{os.path.join(obj_dir,'object.ply')} did not load as a Trimesh object")
            continue

        normalize_mesh(mesh) #important for easi-tex to work
        mesh.export(os.path.join(obj_dir,"object.obj"), file_type='obj', include_texture=False)

        item.set_singapo_obj_path(os.path.join(obj_dir,"object.obj"))

def texture_objects(data,args):
    """
    Texture objects using Easi-Tex.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """
    

    print("Texturing objects using Easi-Tex...")

    prev_dir = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"easi-tex"))

    def make_command(item):
        cmd = (
            f'python scripts/generate_texture.py '
            f'--input_dir "{os.path.dirname(item.singapo_obj_path)}" '
            f'--output_dir "{os.path.join(item.output_path,"easitex")}" '
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
        return cmd

    for item in tqdm(data.get_data_items()):

        item.set_easitex_obj_path(os.path.join(item.output_path,"easitex","canny",f"0-{item.id}","42-ip1.0-cn1.0-dist0.8-gs10.0-p36-h20-us0.4-vt0.1","update","19_post.obj"))
        if args.use_cached and os.path.exists(item.easitex_obj_path):
            continue

        os.system(make_command(item))
    
    os.chdir(prev_dir)

def cosine_sim(a,b):
    """
    Calculate the cosine similarity between two tensors.
    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.
    Returns:
        torch.Tensor: The cosine similarity between the two tensors.
    """


    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)
    return torch.sum(a * b, dim=-1)


def evaluate(data,args):
    """
    Evaluate the synthesized objects.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """

    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True).cuda()

    transform = transforms.Compose([
        transforms.Resize((518, 518)),  # DINOv2 expects 518x518
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((518, 518)), 
        transforms.ToTensor(),
    ])

    print("Evaluating objects...")

    for item in tqdm(data.get_data_items()):
        synthesized_image_path = os.path.join(args.output_path,"synthesized.png")
        synthesized_mask_path = os.path.join(args.output_path,"synthesized_mask.png")

        if not (args.use_cached and os.path.exists(synthesized_image_path)):
            #render the synthesized mesh
            mesh = trimesh.load_mesh(args.easitex_obj_path)

            render_mesh(mesh, resolution=512,output_path=synthesized_image_path)

        synthesized_image = Image.open(synthesized_image_path).convert("RGB")
        synthesized_mask = Image.open(args.mask_path).convert("L")

        ground_truth_image = Image.open(item.img_path).convert("RGB")
        ground_truth_mask = Image.open(item.mask_path).convert("L")

        synthesized_image_tensor = transform(synthesized_image)
        ground_truth_image_tensor = transform(ground_truth_image)
        synthesized_mask_tensor = transform_mask(synthesized_mask)
        ground_truth_mask_tensor = transform_mask(ground_truth_mask)

        # Apply mask (assuming binary 0 or 1 in mask)
        synthesized_image_tensor_masked = synthesized_image_tensor * synthesized_mask_tensor
        ground_truth_image_tensor_masked = ground_truth_image_tensor * ground_truth_mask_tensor

        # Prepare batch
        input_tensor = torch.stack([synthesized_image_tensor_masked, ground_truth_image_tensor_masked]).cuda()
        # Inference
        with torch.no_grad():
            embedding = dinov2_vitb14_reg(input_tensor)
        # Calculate cosine similarity
        similarity = cosine_sim(embedding[0], embedding[1])

        item.set_cosine_similarity(similarity.item())

        if args.add_no_easitex:
            synthesized_image_path = os.path.join(args.output_path,"synthesized_no_easitex.png")
            synthesized_mask_path = os.path.join(args.output_path,"synthesized_no_easitex_mask.png")

            # Render the synthesized mesh without Easi-Tex
            mesh = trimesh.load_mesh(item.singapo_obj_path)
            render_mesh(mesh, resolution=512,output_path=synthesized_image_path)

            synthesized_image = Image.open(synthesized_image_path).convert("RGB")
            synthesized_mask = Image.open(args.mask_path).convert("L")

            synthesized_image_tensor = transform(synthesized_image)
            synthesized_mask_tensor = transform_mask(synthesized_mask)

            # Apply mask (assuming binary 0 or 1 in mask)
            synthesized_image_tensor_masked = synthesized_image_tensor * synthesized_mask_tensor

            # Prepare batch
            input_tensor = torch.stack([synthesized_image_tensor_masked, ground_truth_image_tensor_masked]).cuda()
            # Inference
            with torch.no_grad():
                embedding = dinov2_vitb14_reg(input_tensor)
            # Calculate cosine similarity
            similarity = cosine_sim(embedding[0], embedding[1])

            item.set_cosine_similarity_no_easitex(similarity.item())



def display_results(data,args):
    """
    Display the results of the evaluation.
    Args:
        data (EvaluationData): The evaluation data object.
        args (argparse.Namespace): The arguments passed to the script.
    """

    total_cosine_similarity = 0
    total_cosine_similarity_no_easitex = 0

    for item in data.get_data_items():
        if item.valid:
            print(f"Item ID: {item.id}")
            print(f"Cosine Similarity: {item.cosine_similarity}")
            print("-" * 20)
            total_cosine_similarity += item.cosine_similarity
            if args.add_no_easitex:
                print(f"Cosine Similarity (No Easi-Tex): {item.cosine_similarity_no_easitex}")
                total_cosine_similarity_no_easitex += item.cosine_similarity_no_easitex
                print("-" * 20)
    
    print(f"Average Cosine Similarity: {total_cosine_similarity / len(data.get_data_items())}")
    if args.add_no_easitex:
        print(f"Average Cosine Similarity (No Easi-Tex): {total_cosine_similarity_no_easitex / len(data.get_data_items())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--singapo_ckpt_path", type=str, default='singapo/exps/singapo/final/ckpts/last.ckpt', help="path to the checkpoint of the model")
    parser.add_argument("--singapo_config_path", type=str, default='singapo/exps/singapo/final/config/parsed.yaml', help="path to the config file")
    parser.add_argument("--singapo_gt_data_root", type=str, default='data', help="the root directory of the original data, used for part mesh retrieval")
    parser.add_argument("--eval_data_path", type=str, default='eval_data', help="path to the data to be evaluated")
    parser.add_argument("--output_path", type=str, default='output', help="path to save the output")
    parser.add_argument("--use_cached", action="store_true", help="whether to use cached objects")
    parser.add_argument("--add_no_easitex", action="store_true", help="additionally evaluate the objects without Easi-Tex texturing")

    args = parser.parse_args()

    #All paths used in singapo and easi-tex should be absolute
    args.output_path = os.path.abspath(args.output_path)
    args.singapo_ckpt_path = os.path.abspath(args.singapo_ckpt_path)
    args.singapo_config_path = os.path.abspath(args.singapo_config_path)
    args.singapo_gt_data_root = os.path.abspath(args.singapo_gt_data_root)

    data = EvaluationData(args.eval_data_path,args.output_path,args.use_cached)
    synthesize_objects(data,args)
    texture_objects(data,args)

    evaluate(data,args)
    display_results(data,args)








