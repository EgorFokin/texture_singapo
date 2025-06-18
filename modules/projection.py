from modules.module import EvalModule
import os
from tqdm import tqdm
import trimesh
from eval_utils.render_utils import project_texture, split_by_reference
from eval_utils.render_compare import compare_to_ground_truth
from eval_utils.articulate import articulate

class ProjectionModule(EvalModule):
    def __init__(self, system_args):
        super().__init__(system_args)

    def _texture_naive(self,data):
        """
        Texture objects using naive projection.
        Args:
            data (EvaluationData): The evaluation data object.
        """

        print("Texturing objects using naive projection...")

        for item in tqdm(data.get_data_items()):

            if self.system_args.use_cached and os.path.exists(item.naive_texturing_path):
                continue

            # Load the mesh
            mesh = trimesh.load(item.singapo_obj_path,group_material=False)
            mesh = trimesh.util.concatenate(mesh.dump())
            project_texture(mesh, item.img_path, item.mask_path, item.naive_texturing_path)

    def generate(self, data):
        """
        Generate textures for the objects in the dataset using naive projection.
        Args:
            data (EvaluationData): The evaluation data object.
        """
        self._texture_naive(data)

    def evaluate(self, data):
        """
        Evaluate the generated textures.
        Args:
            data (EvaluationData): The evaluation data object.
        """
        print("Evaluating naive projection textures...")

        for item in tqdm(data.get_data_items()):

            out_naive = os.path.join(item.output_path, "naive_texturing")
            os.makedirs(out_naive, exist_ok=True)

            singapo_mesh = trimesh.load(item.singapo_obj_path,group_material=False)
            naive_tex_mesh = trimesh.load(item.naive_texturing_path)
            naive_tex_mesh = split_by_reference(singapo_mesh, naive_tex_mesh)
            gt_mesh = trimesh.load(item.scene_path,group_material=False)

            if self.system_args.articulated:
                articulate(naive_tex_mesh, item.singapo_dict)
                articulate(gt_mesh, item.gt_dict)


            sim = compare_to_ground_truth(naive_tex_mesh, gt_mesh, out_naive, self.system_args)
            item.set_naive_cosine_similarity(sim)

        



