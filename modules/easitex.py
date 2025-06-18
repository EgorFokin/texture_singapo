from modules.module import EvalModule
import os
from tqdm import tqdm
from eval_utils.utils import split_by_reference
from eval_utils.render_compare import compare_to_ground_truth
from eval_utils.articulate import articulate
import trimesh

class EasiTexModule(EvalModule):
    def __init__(self, system_args):
        super().__init__(system_args)

    def _texture_objects(self,data):
        """
        Texture objects using Easi-Tex.
        Args:
            data (EvaluationData): The evaluation data object.
        """

        print("Texturing with Easi-Tex...")

        root = os.path.join(os.getcwd(), "easi-tex")
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


            if self.system_args.use_cached and os.path.exists(tex_path):
                continue

            os.system(get_cmd(item))

        os.chdir(prev)

    def generate(self, data):
        """
        Generate textures for the objects in the dataset using Easi-Tex.
        Args:
            data (EvaluationData): The evaluation data object.
        """
        self._texture_objects(data)
    
    def evaluate(self, data):
        """
        Evaluate the generated textures.
        Args:
            data (EvaluationData): The evaluation data object.
        """
        print("Evaluating Easi-Tex textures...")

        for item in tqdm(data.get_data_items()):


            singapo_mesh = trimesh.load(item.singapo_obj_path,group_material=False)
            easitex_mesh = trimesh.load(item.easitex_obj_path)
            easitex_mesh = split_by_reference(singapo_mesh,easitex_mesh)
            gt_mesh = trimesh.load(item.scene_path,group_material=False)

            if self.system_args.articulated:
                articulate(easitex_mesh, item.singapo_dict)
                articulate(gt_mesh, item.gt_dict)


            sim = compare_to_ground_truth(easitex_mesh, gt_mesh, item.output_path, self.system_args)
            item.set_cosine_similarity(sim)

        