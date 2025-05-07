import os
from pathlib import Path
import trimesh
from tqdm import tqdm
import shutil
from texture_singapo_utils import get_image_description, get_object_mask

class DataItem:
    def __init__(self, path,output_path,use_cached=True):
        self.valid = True
        self.id = os.path.basename(path)
        self.path = path
        self.output_path = os.path.join(output_path,self.id)
        self.use_cached = use_cached
        self.singapo_obj_path = None
        self.easitex_obj_path = None
        self.cosine_similarity = None
        self.cosine_similarity_no_easitex = None

        

        if not os.path.exists(os.path.join(path,"imgs","00.png")):
            self.valid = False
            print(f"Invalid data item: {path}")
            return

        self.img_path = os.path.join(self.output_path,"image.png")

        os.makedirs(self.output_path, exist_ok=True)
        
        shutil.copy(os.path.join(path,"imgs","00.png"), self.img_path)

        self.parts_path = os.path.join(path,"objs")

        if not os.path.exists(self.parts_path):
            self.valid = False
            print(f"Invalid data item: {path}")
            return

        self.obj_path = self._construct_full_obj()

        if use_cached and os.path.exists(os.path.join(self.output_path,"description.txt")):
            with open(os.path.join(self.output_path,"description.txt"), "r") as f:
                self.description = f.read()
        else:
            self.description = get_image_description(self.img_path)
            with open(os.path.join(self.output_path,"description.txt"), "w") as f:
                f.write(self.description)
        
        self.mask_path = os.path.join(self.output_path,"image_mask.png")
        if not (use_cached and os.path.exists(self.mask_path)):
            get_object_mask(self.img_path,self.mask_path)
    
    def _construct_full_obj(self):
        """
        Constructs the full object mesh from the parts.

        Returns:
            str: Path to the constructed full object mesh.
        """

        obj_path = os.path.abspath(os.path.join(self.output_path,"full_obj",f"{self.id}.obj"))
        if self.use_cached and os.path.exists(obj_path):
            return obj_path

        os.makedirs(os.path.join(self.output_path,"full_obj"), exist_ok=True)
        
        parts = []

        for part in os.listdir(self.parts_path):
            if part.endswith(".mtl"):
                continue
            part_path = os.path.join(self.parts_path,part)
            if os.path.exists(part_path):
                mesh = trimesh.load(part_path, force='mesh')  # force loading as Trimesh
                if not isinstance(mesh, trimesh.Trimesh):
                    self.valid = False
                    print(f"{part_path} did not load as a Trimesh object")
                    return None
                parts.append(mesh)
        
        if len(parts) == 0:
            self.valid = False
            print(f"No valid parts found in {self.parts_path}")
            return None
        
        combined_mesh = trimesh.util.concatenate(parts)
        combined_mesh.export(obj_path, file_type='obj', include_texture=False)
            
        return obj_path
    
    def set_singapo_obj_path(self, path):
        """
        Set the path to the Singapo generated object.

        Args:
            path (str): Path to the Singapo generated object.
        """
        self.singapo_obj_path = path

    def set_easitex_obj_path(self, path):
        """
        Set the path to the Easi-Tex generated object.

        Args:
            path (str): Path to the Easi-Tex generated object.
        """
        self.easitex_obj_path = path
    
    def set_cosine_similarity(self, similarity):
        """
        Set the cosine similarity between the generated object and the original object.

        Args:
            similarity (float): Cosine similarity value.
        """
        self.cosine_similarity = similarity

    def set_cosine_similarity_no_easitex(self, similarity):
        """
        Set the cosine similarity between the generated object and the original object without Easi-Tex.

        Args:
            similarity (float): Cosine similarity value.
        """
        self.cosine_similarity_no_easitex = similarity
    



class EvaluationData:
    def __init__(self, data_path,output_path,use_cached=True):
        self.data_path = data_path
        self.output_path = output_path
        self.use_cached = use_cached
        self.items = []

        self._load_items()

    def _load_items(self):
        """
        Load all items in the dataset directory.
        """

        print(f"Loading data from {self.data_path}...")

        root = Path(self.data_path)

        item_paths = []

        for dataset in root.iterdir():
            if dataset.is_dir():
                for class_dir in dataset.iterdir():
                    if class_dir.is_dir():
                        for item in class_dir.iterdir():
                            if item.is_dir():
                                item_paths.append(item)
        
        for item_path in tqdm(item_paths):
            item = DataItem(item_path,self.output_path,self.use_cached)
            if item.valid:
                self.items.append(item)
            else:
                print(f"Invalid item: {item_path}, skipping...")


    def get_data_items(self):
        return self.items
