from modules.module import EvalModule
import os
from tqdm import tqdm
from generate import generate
from eval_utils.utils import normalize_mesh
import trimesh

class SingapoModule(EvalModule):
    def __init__(self, system_args):
        super().__init__(system_args)

    class Args:
        def __init__(
            self,
            img='demo/demo_input.png',
            ckpt='exps/singapo/final/ckpts/last.ckpt',
            config='config/parsed.yaml',
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


    def _synthesize_objects(self,data):
        """
        Synthesize objects using Singapo.
        Args:
            data (EvaluationData): The evaluation data object.
        """

        print("Synthesizing with Singapo...")


        for item in tqdm(data.get_data_items()):
            out_dir = os.path.join(item.output_path, "singapo", "0")

            if self.system_args.use_cached and os.path.exists(os.path.join(out_dir, "object.obj")):
                item.set_singapo_obj_path(os.path.join(out_dir, "object.obj"))
                item.set_singapo_dict(os.path.join(out_dir, "object.json"))
                continue

            s_args = self.Args(
                img=item.img_path,
                ckpt=self.system_args.singapo_ckpt_path,
                config=self.system_args.singapo_config_path,
                gt_root=self.system_args.singapo_gt_data_root,
                out=os.path.join(item.output_path, "singapo"),
            )

            generate(s_args)

            scene = trimesh.Scene()

            for file in os.listdir(os.path.join(out_dir, "plys")):
                if file.endswith(".ply"):
                    mesh = trimesh.load(os.path.join(out_dir,"plys", file), force='mesh')
                    
                    normalize_mesh(mesh)
            
                    scene.add_geometry(mesh,node_name=file.split('.')[0])
            
            scene.export(os.path.join(out_dir, "object.obj"), file_type='obj', include_texture=False)
            item.set_singapo_obj_path(os.path.join(out_dir, "object.obj"))

            item.set_singapo_dict(os.path.join(out_dir, "object.json"))

    def generate(self, data):
        """
        Generate objects using Singapo.
        Args:
            data (EvaluationData): The evaluation data object.
        """
        self._synthesize_objects(data)

        


