from openai import OpenAI
import base64
from dotenv import load_dotenv
from rembg import remove
import pyrender
import trimesh
from PIL import Image
import numpy as np
load_dotenv()

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_image_description(image_path):
    """
    Get a description of the image using OpenAI's GPT-4o model.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Description of the image.
    """
    # Encode the image
    base64_image = encode_image(image_path)

    # Create a chat completion request
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Provide a brief description of an object. Example response: A wooden cabinet with metal handles."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=250,
    )

    return response.choices[0].message.content

def get_object_mask(image_path,output_path):
    """
    Get the object mask from the image using rembg library.
    
    Args:
        image_path (str): Path to the image file.
        output_path (str): Path to save the output image.
    """

    with open(image_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input,only_mask=True)
            o.write(output)
        

def render_mesh(mesh, resolution=512, output_path=None, is_instantmesh=False):
    """
    Render a mesh to a 2D image, centering and scaling it so the bounding box is [-1, 1].

    Args:
        mesh (trimesh.Trimesh): The mesh to render.
        resolution (int): The resolution of the output image.
        output_path (str, optional): If specified, saves the image and mask to this path.

    Returns:
        np.ndarray: The rendered image.
    """

    render_mesh = mesh.copy()

    # Normalize mesh to fit bounding box [-1, 1]
    bbox_min = render_mesh.bounds[0]
    bbox_max = render_mesh.bounds[1]
    center = (bbox_min + bbox_max) / 2
    scale = 2.0 / np.max(bbox_max - bbox_min)  # scale to fit in [-1, 1]
    
    render_mesh.apply_translation(-center)
    render_mesh.apply_scale(scale)


    if is_instantmesh:
        # Flip the mesh, to better represent the original image
        render_mesh.apply_transform([
            [-1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]
        ])
        render_mesh.apply_transform(trimesh.transformations.rotation_matrix(
            angle= np.pi,
            direction=[0, 1, 0]
        ))
        render_mesh.apply_transform(trimesh.transformations.rotation_matrix(
            angle= np.pi / 10,
            direction=[1, 0, 0]
        ))
    
    else:
        # Slightly rotate the mesh for better visualization
        render_mesh.apply_transform(trimesh.transformations.rotation_matrix(
            angle= -np.pi / 6,
            direction=[0, 1, 0]
        ))
        render_mesh.apply_transform(trimesh.transformations.rotation_matrix(
            angle= np.pi / 10,
            direction=[1, 0, 0]
        ))
    
    # Create a scene
    scene = pyrender.Scene()

    intensity = 20
    
    if not hasattr(render_mesh.visual, 'material'):
        # Create a grey material if no texture exists
        grey_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.5, 0.5, 0.5, 1.0],
            metallicFactor=0.1,
            roughnessFactor=0.7
        )
        intensity = 5
        mesh_node = pyrender.Mesh.from_trimesh(render_mesh, material=grey_material)
    else:
        mesh_node = pyrender.Mesh.from_trimesh(render_mesh)
    scene.add(mesh_node)
    
    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 3.0  # Move the camera back
    scene.add(camera, pose=camera_pose)
    
    # Set up the light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
    scene.add(light, pose=camera_pose)
    
    # Render the scene
    r = pyrender.OffscreenRenderer(resolution, resolution)
    color, depth = r.render(scene)
    r.delete()

    mask = (depth > 0).astype(np.uint8) * 255  # Convert to 0 or 255
    
    # Save the rendered image
    if output_path:
        image = Image.fromarray(color)
        mask_img = Image.fromarray(mask)
        image.save(output_path)
        mask_img.save(output_path.replace('.png', '_mask.png'))
    
    return color

def normalize_mesh(mesh):
    """
    Normalize the mesh by:
    1. Removing degenerate faces
    2. Merging and welding vertices that are very close
    3. Ensuring consistent winding of faces
    Args:
        mesh (trimesh.Trimesh): The mesh to normalize.
    """
    
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-5))

    # 2. Merge and weld vertices that are very close
    mesh.merge_vertices()  # merges vertices within tolerance
    mesh.remove_unreferenced_vertices()

    # 3. Ensure consistent winding of faces
    mesh.fix_normals()


def project_texture(mesh, image_path, mask_path, output_path):
    """
    Project a texture onto a mesh using UV mapping.
    
    Args:
        mesh (trimesh.Trimesh): The mesh to project the texture onto.
        image (PIL.Image): The texture image.
        mask (PIL.Image): The mask image.
        output_path (str): Path to save the textured mesh.
    """
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path)

    mask_np = np.array(mask)
    
    # Find the bounding box of the mask
    masked_pixels = np.where(mask_np > 150)
    
    top_y = masked_pixels[0].min()
    bottom_y = masked_pixels[0].max()
    
    left_x = masked_pixels[1].min()
    right_x = masked_pixels[1].max()
    
    # Crop the image to the bounding box of the mask
    image = image.crop((left_x, top_y, right_x, bottom_y))
    
    # Orthographic projection
    vertices = mesh.vertices.copy()
    uvs = vertices[:, :2] 
    
    # Normalize UVs to [0, 1]
    uvs[:, 0] = (uvs[:, 0] - uvs[:, 0].min()) / (uvs[:, 0].max() - uvs[:, 0].min())
    uvs[:, 1] = (uvs[:, 1] - uvs[:, 1].min()) / (uvs[:, 1].max() - uvs[:, 1].min())
    
    # Flip V to match image coordinates
    # uvs[:, 1] = 1.0 - uvs[:, 1]
    
    visual = trimesh.visual.texture.TextureVisuals(uv=uvs, image=image)
    mesh.visual = visual
    
    mesh.export(output_path)
