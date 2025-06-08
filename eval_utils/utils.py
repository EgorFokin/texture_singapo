from openai import OpenAI
import base64
from dotenv import load_dotenv
from rembg import remove
import pyrender
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
import numpy as np
import xatlas
import os
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

def make_double_sided(mesh):
    # Flip faces winding
    flipped_faces = mesh.faces[:, ::-1]
    
    # Combine vertices
    vertices = np.vstack((mesh.vertices, mesh.vertices))
    
    # Combine faces: original + flipped (with offset)
    faces = np.vstack((mesh.faces, flipped_faces + len(mesh.vertices)))

    normals = np.vstack((mesh.vertex_normals, -mesh.vertex_normals))
    
    if not hasattr(mesh.visual,"material"):
        return trimesh.Trimesh(vertices=vertices, faces=faces,vertex_normals=normals, process=False)

    uv = np.vstack((mesh.visual.uv, mesh.visual.uv))
    
    # Build visual with preserved texture image and UVs
    visual = mesh.visual.copy()
    visual.uv = uv

    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual,vertex_normals=normals, process=False)



    return new_mesh

        

def render_mesh(mesh, resolution=512, output_path=None, is_instantmesh=False, as_scene=False):
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

    intensity = 5

    if as_scene:

        intensity = 2
        
        # Normalize mesh to fit bounding box [-1, 1]
        bbox_min = render_mesh.bounds[0]
        bbox_max = render_mesh.bounds[1]
        center = (bbox_min + bbox_max) / 2
        scale = 2.0 / np.max(bbox_max - bbox_min)  # scale to fit in [-1, 1]
        
        render_mesh.apply_translation(-center)
        render_mesh.apply_scale(scale)


        render_mesh.apply_transform(trimesh.transformations.rotation_matrix(
            angle= -np.pi / 6,
            direction=[0, 1, 0]
        ))
        render_mesh.apply_transform(trimesh.transformations.rotation_matrix(
            angle= np.pi / 10,
            direction=[1, 0, 0]
        ))

        scene = pyrender.Scene()

        for name, geom in render_mesh.geometry.items():
            # Disable backface culling using material properties
            if not hasattr(geom.visual, 'material'):
                # Create a grey material if no texture exists
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.5, 0.5, 0.5, 1.0],
                    metallicFactor=0.1,
                    roughnessFactor=0.7,
                    doubleSided=True
                )
            else:
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorTexture=geom.visual.material.baseColorTexture,
                    metallicFactor=0.1,
                    roughnessFactor=0.7,
                    doubleSided=True
                )
            
            # Wrap the trimesh geometry in a pyrender.Mesh
            mesh = pyrender.Mesh.from_trimesh(geom, material=material, smooth=False)

            transform = render_mesh.graph[name][0]

            scene.add(mesh, pose=transform)

        #scene = pyrender.Scene.from_trimesh_scene(render_mesh)
    else:

        render_mesh = trimesh.util.concatenate(render_mesh.dump())
    
        # render_mesh.unmerge_vertices()

        #render_mesh = make_double_sided(render_mesh)


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

        
        
        if not hasattr(render_mesh.visual, 'material'):
            # Create a grey material if no texture exists
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.5, 0.5, 0.5, 1.0],
                metallicFactor=0.1,
                roughnessFactor=0.7
            )
        else:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorTexture=render_mesh.visual.material.image,
                baseColorFactor=render_mesh.visual.material.main_color,
                metallicFactor=0.1,
                roughnessFactor=0.7
            )
            #intensity = 20
        mesh_node = pyrender.Mesh.from_trimesh(render_mesh,material=material)
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


def normalize_scene_to_unit_cube(scene):
    # Get the axis-aligned bounding box of the whole scene
    bounds = scene.bounds
    min_corner, max_corner = bounds

    # Compute translation and scale
    extent = max_corner - min_corner
    scale_factors = 1.0 / extent

    for geom in scene.geometry.values():
        # Move min corner to origin
        geom.apply_translation(-min_corner)
        # Scale to unit cube (non-uniform if needed)
        geom.apply_scale(scale_factors)

def normalize_mesh_to_unit_cube(mesh):
    v = mesh.vertices
    bmin, bmax = v.min(0), v.max(0)
    scale = np.where(bmax - bmin == 0, 1, bmax - bmin)
    v_norm = (v - bmin) / scale
    return trimesh.Trimesh(vertices=v_norm, faces=mesh.faces, process=False)


def resize_to_reference(scene_to_resize, reference_scene):
    """
    Scales and translates `scene_to_resize` so its bounding box matches that of `reference_scene`.
    """
    # Compute bounding boxes
    ref_min, ref_max = reference_scene.bounds
    tgt_min, tgt_max = scene_to_resize.bounds

    ref_size = ref_max - ref_min
    tgt_size = tgt_max - tgt_min

    # Compute scale factors
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(tgt_size != 0, ref_size / tgt_size, 1.0)

    # Use uniform scale (minimum to fit inside ref box)
    uniform_scale = scale.min()

    # Apply transformations
    for geom in scene_to_resize.geometry.values():
        # Move target min corner to origin
        geom.apply_translation(-tgt_min)
        # Uniform scale to match reference size
        geom.apply_scale(uniform_scale)
        # Translate to reference min corner
        geom.apply_translation(ref_min)
    

def split_by_reference(reference_mesh,target_mesh):
    """
    Split the mesh into two parts based on the reference mesh.
    
    Args:
        reference_mesh (trimesh.Trimesh): The reference mesh with parts.
        target_mesh (trimesh.Trimesh): The target mesh to be split.
    """
    from scipy.spatial import KDTree

    reference_mesh_norm = reference_mesh.copy()

    normalize_scene_to_unit_cube(reference_mesh_norm)
    target_mesh_norm = normalize_mesh_to_unit_cube(target_mesh)

    target_vertices = target_mesh_norm.vertices

    tree = KDTree(target_mesh_norm.vertices)

    threshold = 0.0001

    part_to_vertices = {}

    for part_name, part_geometry in reference_mesh_norm.geometry.items():
        part_vertices = part_geometry.vertices
        matched_vertex_indices = set()  # Avoid duplicates
        
        # Find all target vertices close to this part's vertices
        for vertex in part_vertices:
            nearby_indices = tree.query_ball_point(vertex, r=threshold)

            matched_vertex_indices.update(nearby_indices)
        
        part_to_vertices[part_name] = list(matched_vertex_indices)

    # Now, assign faces to new parts based on vertex membership
    new_parts = {}  # { part_name → trimesh.Trimesh }

    for part_name, vertex_indices in part_to_vertices.items():
        # Get faces where ALL vertices are in `vertex_indices`
        mask = np.all(np.isin(target_mesh_norm.faces, vertex_indices), axis=1)

        face_indices = np.where(mask)[0]
        
        # Extract the submesh
        submesh = target_mesh.submesh([face_indices], append=True)

        new_parts[part_name] = submesh

    split_mesh = trimesh.Scene(new_parts)

    resize_to_reference(split_mesh,reference_mesh)

    return split_mesh


