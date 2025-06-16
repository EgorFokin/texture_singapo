import trimesh
import numpy as np
import pyrender
from PIL import Image
from eval_utils.utils import make_double_sided, split_by_reference


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

        render_mesh = make_double_sided(render_mesh)


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