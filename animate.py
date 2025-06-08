import numpy as np
import pyrender
import trimesh
import imageio
from eval_utils.utils import *
from eval_utils.articulate import articulate


singapo_mesh = trimesh.load("/home/edfokin/Projects/TextureSingapo/texture_singapo/output/11f8b552a802c6233a1332713568f05f901b725c/singapo/0/object.obj",group_material=False)
easitex_mesh = trimesh.load("/home/edfokin/Projects/TextureSingapo/texture_singapo/output/11f8b552a802c6233a1332713568f05f901b725c/easitex/canny/0-11f8b552a802c6233a1332713568f05f901b725c/42-ip1.0-cn1.0-dist0.8-gs10.0-p36-h20-us0.4-vt0.1/update/mesh/19_post.obj")
easitex_mesh = split_by_reference(singapo_mesh,easitex_mesh)
dict_path = "/home/edfokin/Projects/TextureSingapo/texture_singapo/output/11f8b552a802c6233a1332713568f05f901b725c/singapo/0/object.json"

frames = []
for i in range(30):
    mesh = easitex_mesh.copy()
    articulate(mesh, dict_path, joint_state=i/30.0)
    color = render_mesh(mesh, resolution=512, output_path=None)
    frames.append(color)

reverse = frames[::-1]

frames = frames + reverse

# Save as GIF
imageio.mimsave("articulated.gif", frames, duration=0.05)
