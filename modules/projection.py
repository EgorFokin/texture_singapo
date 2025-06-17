from modules.module import EvalModule
import os
from tqdm import tqdm
import trimesh
from eval_utils.render_utils import project_texture

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

        



