from modules.module import EvalModule
import os
from tqdm import tqdm
import tempfile
import shutil

class TEXTureModule(EvalModule):
    def __init__(self, system_args):
        super().__init__(system_args)

    yaml_template ="""
log:
  exp_name: eval
guide:
  text: "A photo of %s, {} view"
  append_direction: True
  shape_path: %s
optim:
  seed: 3
"""

    def _get_cmd(self, tmp_yaml):
        """
        Generate the command to run TEXTure for a given item.
        Args:
            tmp_yaml (str): Path to the temporary YAML file.
        Returns:
            str: The command to run TEXTure.
        """
        return (f'python -m scripts.run_texture --config_path={tmp_yaml} ')

    def _texture_objects(self,data):
        """
        Texture objects using TEXTure.
        Args:
            data (EvaluationData): The evaluation data object.
        """

        root = os.path.join(os.getcwd(), "TEXTurePaper")
        prev = os.getcwd()
        os.chdir(root)

        print("Texturing with TEXTure...")

        for item in tqdm(data.get_data_items()):
            output_folder = os.path.join(item.output_path, "texture")
            item.set_TEXTure_path(os.path.join(output_folder, f"mesh.obj"))

            if self.system_args.use_cached and os.path.exists(item.texture_path) :
                continue

            with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w+') as tmp:
                tmp.write(self.yaml_template % (item.description, item.singapo_obj_path))
                tmp.flush()

                command = self._get_cmd(tmp.name)
                os.system(command)

                

                os.makedirs(output_folder, exist_ok=True)

                experiment_folder = os.path.join("experiments", "eval", "mesh")

                for filename in os.listdir(experiment_folder):
                    source_path = os.path.join(experiment_folder, filename)
                    destination_path = os.path.join(output_folder, filename)

                    # Check if it's a file (skip directories)
                    if os.path.isfile(source_path):
                        shutil.move(source_path, destination_path)

                

        os.chdir(prev)

                





    def generate(self, data):
        """
        Generate textures for the objects in the dataset using TEXTure
        Args:
            data (EvaluationData): The evaluation data object.
        """
        self._texture_objects(data)

        