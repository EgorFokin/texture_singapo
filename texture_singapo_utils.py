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
        

def render_mesh(mesh, resolution=512,output_path=None):
    """
    Render a mesh to a 2D image.
    
    Args:
        mesh (trimesh.Trimesh): The mesh to render.
        resolution (int): The resolution of the output image.
    
    Returns:
        np.ndarray: The rendered image.
    """
    # Create a scene
    scene = pyrender.Scene()
    
    # Add the mesh to the scene
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)
    
    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 2.0  # Move the camera back
    scene.add(camera, pose=camera_pose)
    
    # Set up the light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    scene.add(light, pose=camera_pose)
    
    # Render the scene
    r = pyrender.OffscreenRenderer(resolution, resolution)
    color, depth = r.render(scene)

    mask = (depth > 0).astype(np.uint8) * 255  # Convert to 0 or 255
    
    # Save the rendered image
    if output_path:
        image = Image.fromarray(color)
        mask = Image.fromarray(mask)
        image.save(output_path)
        mask.save(output_path.replace('.png', '_mask.png'))
    
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
