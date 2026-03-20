from scannetpp.common.utils.rasterize import get_fisheye_cameras_batch, get_opencv_cameras_batch
import torch
import json
from pathlib import Path

import open3d as o3d
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    MeshRasterizer, MeshRenderer, RasterizationSettings, 
    HardPhongShader, AmbientLights, TexturesVertex
)
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer

import numpy as np
from lmdeploy import pipeline, GenerationConfig
from lmdeploy.vl import load_image

class LMDeploy_Model:
    def __init__(self, model_name: str):
        self.pipe = pipeline(model_name)
        print(f'Pipeline created with {model_name}')

    def __call__(self, prompt_text, image_path=None, sample=False):
        gen_config = GenerationConfig(do_sample=sample)

        # VLM
        if image_path is not None:
            image = load_image(image_path)
            response = self.pipe((prompt_text, image), gen_config=gen_config)
        # LLM
        else:
            response = self.pipe(prompt_text, gen_config=gen_config)

        if type(response) == list: # batched
            return [r.text for r in response]
        else: # single prompt
            return response.text
    
class LMDeploy_Dummy_Model:
    def __init__(self) -> None:
        pass

    def __call__(self, prompt_text, image_path=None, sample=False):
        if type(prompt_text) == list:
            return ['DUMMY RESPONSE'] * len(prompt_text)
        else:
            return 'DUMMY RESPONSE'

def o3d_mesh_to_pytorch3d(o3d_mesh, device):
    vertices = torch.tensor(np.asarray(o3d_mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(o3d_mesh.triangles), dtype=torch.int64, device=device)
    
    # Check if the mesh has vertex colors
    if len(o3d_mesh.vertex_colors) > 0:
        verts_rgb = torch.tensor(np.asarray(o3d_mesh.vertex_colors), dtype=torch.float32, device=device)
    else:
        # Default to gray if no vertex colors are present
        verts_rgb = torch.full(vertices.shape, 0.7, dtype=torch.float32, device=device)
    
    textures = TexturesVertex(verts_features=[verts_rgb])  # Assign vertex colors as texture
    return Meshes(verts=[vertices], faces=[faces], textures=textures)

def rasterize_o3d_mesh_to_image(o3d_mesh, campose, intrinsic, img_width, img_height, distort_params):
    '''
    rasterize open3d mesh to an image at given camera pose using pytorch3d
    '''
    rendered_img = None
    # gpu 
    device = torch.device("cuda:0") 
    torch.cuda.empty_cache()
    torch.backends.cuda.preferred_linalg_library(backend='magma')
    
    # _, _, meshes = prep_pt3d_inputs(o3d_mesh)
    meshes = o3d_mesh_to_pytorch3d(o3d_mesh, device)
    poses = torch.Tensor(campose).unsqueeze(0)
    # directly get undistorted images -> wrong?
    # cameras_batch = get_opencv_cameras_batch(poses, img_height, img_width, intrinsic).to(device)
    # render distorted, undistort it later if needed
    cameras_batch = get_fisheye_cameras_batch(poses, img_height, img_width, intrinsic, distort_params).to(device)
    # Define rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(img_height, img_width),
        blur_radius=0.0,
        faces_per_pixel=1
    )
    
    # Define ambient lighting (soft uniform illumination)
    ambient_light = AmbientLights(device=device)  # No directional lighting
    
    # Define renderer with HardPhongShader and ambient light
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras_batch, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras_batch, lights=ambient_light)
    )
    
    # Render image
    # 1, W, H, 4
    rendered_image = renderer(meshes)
    rendered_img = rendered_image[0].detach().cpu().numpy()
    # pick first 3 rgb channels
    rendered_img = rendered_img[..., :3]
    # 0-1 -> 0-255 ccolors, then uint8
    rendered_img = (rendered_img * 255).astype(np.uint8)

    # write to tmp file using o3d
    # o3d_img = o3d.geometry.Image(rendered_img)
    # o3d.io.write_image('tmp_rendered_img.png', o3d_img)
    
    return rendered_img


def load_instance_preds(scene_id, instance_preds_dir, sem_classes_file):
    '''
    load in the same format as gt anno with 
        objects: dict of id -> {id, label}
        and 
        vertex_obj_ids (0=nothing)
    '''
    # read file
    sem_classes = open(sem_classes_file).readlines()
    sem_classes = [sem_class.strip() for sem_class in sem_classes]

    preds_dir = Path(instance_preds_dir)
    vtx_obj_ids = np.load(preds_dir / f'{scene_id}.npy')
    # make negative -> 0
    vtx_obj_ids[vtx_obj_ids < 0] = 0
    with open(preds_dir / f'{scene_id}.json', 'r') as f:
        instance_info = json.load(f)
    objects = {}
    for obj_id, instance in instance_info.items():
        label_id = instance['label_id']
        label = sem_classes[label_id]
        # make key int here, to be the same as real anno
        objects[int(obj_id)] = {'id': obj_id, 'label': label}

    return {'objects': objects, 'vertex_obj_ids': vtx_obj_ids}
    

def create_global_parts_prompt(obj_label):
    prompt = f'Describe the different parts of the {obj_label}. '
    prompt += 'Do not mention the image, camera or viewpoint. Refer to the object and the parts by their names. '
    prompt += f'Describe the shape, structure, color and texture of the parts of the {obj_label}. '
    prompt += f'Describe the parts using information available from the image, and not common-sense facts. '
    prompt += 'Do not describe anything other than the parts. Do not describe the lack of any features or contrast with the surroundings. '

    return prompt

def create_img_prompt(obj_label, obj_location_3d, obj_location_2d, obj_dims, near_objs_prompt, cfg, local_only=False, global_only=False):
    '''
    combine full image prompt with prompt for a crop - local and global
    '''
    # local info - doesnt use any 2d or 3d info
    local_prompt = f'Describe the main object in the image, the {obj_label} with dimensions {obj_dims} in meters. '
    local_prompt += 'Do not mention the image, camera or viewpoint. Refer to the object by its name. '
    local_prompt += 'Give details about the shape, structure, color and texture of the object. '
    local_prompt += 'Do not mention the numerical coordinates or dimensions of any of the objects in the output description. '
    local_prompt += f'Describe only the {obj_label} using information available from the image, and not common-sense facts about any the objects. '
    local_prompt += f'Do not describe any objects other than the {obj_label}. '

    if local_only:
        return local_prompt
    
    # global info
    if cfg.global_prompt_obj_locations == '3d':
        obj_location_str = f'located at {obj_location_3d}'
    elif cfg.global_prompt_obj_locations == '2d':
        obj_location_str = f'located at {obj_location_2d}'
    elif cfg.global_prompt_obj_locations == '3d_2d':
        obj_location_str = f'located at {obj_location_3d} in 3D and at {obj_location_2d} in 2D'
    elif cfg.global_prompt_obj_locations is None:
        obj_location_str = '' #empty 

    if cfg.global_prompt_obj_bbox:
        bbox_str = 'indicated with a red bounding box'
        if obj_location_str:
            bbox_str = ' and ' + bbox_str
    else:
        bbox_str = ''

    global_prompt = f'Describe the location of the {obj_label} {obj_location_str} {bbox_str} with respect to the whole 3D scene. '
    # might be empty if there are no nearby objects in the image
    if near_objs_prompt:
        global_prompt += f'Include spatial relations to the most prominent of these reference objects: {near_objs_prompt}. '
        if cfg.global_prompt_obj_locations == '3d':
            global_prompt += 'The object locations are given in 3D coordinates in meters. '
        elif cfg.global_prompt_obj_locations == '2d':
            global_prompt += 'The object locations are given in 2D pixel coordinates. '
            global_prompt += 'The top left of the image is [X, Y] = [0, 0] and the horizontal axis is Y, vertical axis is X. '
        elif cfg.global_prompt_obj_locations == '3d_2d':
            global_prompt += '3D locations are coordinates in meters, and 2D locations are in pixels. '
            global_prompt += 'The top left of the image is [X, Y] = [0, 0] and the horizontal axis is Y, vertical axis is X. '
    if cfg.global_prompt_obj_locations:
        global_prompt += 'Do not mention the numerical coordinates or dimensions of any of the objects in the output description. '
    global_prompt += 'Do not mention spatial relations such as left and right with reference to the camera view or the viewer. '
    global_prompt += 'Keep only relations such as between, on, in front of, behind, above, below, near and next to that are not dependent on the viewpoint. '
    global_prompt += f'Describe only the {obj_label} using information available from the image, and not common-sense facts. '
    global_prompt += "Do not mention any reference objects that are not seen in the image. "
    global_prompt += 'Do not mention the image, camera, bounding box, red box, viewpoint or any part of the image such as foreground or background. '

    if global_only:
        return global_prompt

    return local_prompt + global_prompt

def create_part_prompt(obj_name, bbox=False):
    '''
    combine object crop caption with part prompt
    '''
    bbox_str = 'indicated by a red box' if bbox else ''

    prompt = f'Describe this part of an {obj_name} {bbox_str}. Refer to the part by its name. '
    prompt += 'Do not mention the image, camera, or viewpoint. '
    prompt += 'Describe the shape, structure, color and texture of the part. '
    prompt += 'Start with the name of the part. '
    prompt += f'Describe the part of the {obj_name} using only information from the image, and not using any common-sense facts. '
    prompt += 'Do not describe anything other than the part. Do not describe the lack of any features or contrast with the surroundings. '

    return prompt

# deprecated!
def create_summary_prompt(captions, obj_name, single_image=False):
    '''
    deprecated. use different prompts for local and global
    '''
    generated_from_str = 'a single viewpoint' if single_image else 'different viewpoints'
    # create a prompt to summarize individual captions
    prompt = f'These are descriptions of a {obj_name} in a scene generated from {generated_from_str}.'
    for cap in captions:
        prompt += f'{cap}. '

    prompt += 'Summarize the descriptions into a single description. '
    prompt += 'Retain all the details that are likely to be correct in the output description. '
    prompt += 'Output only the summarized description that describes the object, without mentioning the image or using the word "object". '
    prompt += 'Do not output anything other than the final description. '

    return prompt

# combine local captions from multiview
def create_local_multiview_summary_prompt(captions, obj_name, single_image=False):
    viewpoint_str = 'a single viewpoint' if single_image else 'different viewpoints'

    prompt = f'These are descriptions of the appearance of a {obj_name} from {viewpoint_str}. '
    for cap in captions:
        prompt += f'{cap}. '

    prompt += f'Output a summarized description containing the object name. Output only the summarized description and nothing else. '
    prompt += f'Keep only the most likely and uniquely identifying information about the color, texture, material and shape of the object. '
    prompt += f'Remove any common sense information about the object such as its likely purpose in the scene. '
    prompt += f'Remove the numerical dimensions or numerical location of the object, if present. '
    return prompt

# combine global captions from multiview
def create_global_multiview_summary_prompt(captions, obj_name, single_image=False):
    viewpoint_str = 'a single viewpoint' if single_image else 'different viewpoints'

    prompt = f'These are descriptions of the location of a {obj_name} in a scene from {viewpoint_str}. '
    for cap in captions:
        prompt += f'{cap}. '

    # prompt += f'Output a summarized description starting with {obj_name}. Output only the summarized description and nothing else. '
    prompt += f'Output a summarized description containing the object name. Output only the summarized description and nothing else. '
    prompt += f'Keep only the most likely and uniquely identifying information about the location of the object within the scene and its relation to nearby objects. '
    prompt += f'Remove relations such as left and right with respect to the camera view or the viewer. '
    prompt += f'Keep only relations such as between, on, in front of, behind, above, below, near, next to, adjacent to and touching that are not dependent on the viewpoint. '
    prompt += f'Remove any common sense information about the object such as its likely purpose in the scene. '
    prompt += f'Remove the numerical dimensions or numerical location of the object, if present. '
    return prompt

# final summarization to required length
def get_obj_local_summary_prompt(caption, obj_name, max_len):
    prompt = f'Summarize this description of the appearance of a {obj_name} in a scene to less than {max_len} words. '
    prompt += caption
    prompt += f'Output only the summarized description containing the object name. Output only the summarized description and nothing else. '
    prompt += f'Keep only uniquely identifying information about the color, texture, material and shape of the object. '
    prompt += f'Remove any common sense information about the object such as its likely purpose in the scene. '
    prompt += f'Remove the numerical dimensions or numerical location of the object, if present. '
    return prompt

# final summarization to required length
def get_obj_global_summary_prompt(caption, obj_name, max_len):
    prompt = f'Summarize this description of the location of a {obj_name} in a scene to less than {max_len} words. '
    prompt += caption
    # prompt += f'Output only the summarized description starting with {obj_name}. Output only the summarized description and nothing else. '
    prompt += f'Output only the summarized description containing the object name. Output only the summarized description and nothing else. '
    prompt += f'Keep only uniquely identifying information about the location of the object within the scene and its relation to nearby objects. '
    prompt += f'Remove relations such as left and right with respect to the camera view or the viewer. '
    prompt += f'Keep only relations such as between, on, in front of, behind, above, below, near, next to and adjacent to that are not dependent on the viewpoint. '
    prompt += f'Remove any common sense information about the object such as its likely purpose in the scene. '
    prompt += f'Remove the numerical dimensions or numerical location of the object, if present. '
    return prompt

# final summarization to required length
def get_obj_summary_prompt(caption, obj_name, max_len):
    prompt = f'Summarize this description of a {obj_name} in a scene to less than {max_len} words. '
    prompt += caption
    prompt += f'Output only the summarized description containing the object name. Output only the summarized description and nothing else. '
    prompt += f'Keep all uniquely identifying information such as the color, texture, material and location of the object within the scene. '
    prompt += f'Remove any common sense information about the object such as its likely purpose in the scene. '
    prompt += f'Remove the numerical dimensions or numerical location of the object, if present. '
    return prompt

def get_joint_parts_summary_prompt(part_caps, obj_name, max_len=None):
    prompt = f'These are descriptions of parts of a single {obj_name}.'

    for part_cap in part_caps:
        prompt += f'{part_cap}. '

    if max_len:
        prompt += f'Summarize the descriptions into a single description of all the parts of the {obj_name} with less than {max_len} words. '
    else:
        prompt += f'Summarize the descriptions into a single description of all the parts of the {obj_name}.'

    prompt += 'Output only the summarized description that describes the parts, without mentioning the images or using the word "scene" or "object". '
    prompt += 'Retain all the details that are likely to be correct in the output description. '
    prompt += 'Keep all uniquely identifying information such as the color, texture, material of the parts. '
    prompt += 'Do not output anything other than the description. Remove any common sense information about the parts.'
    prompt += f'Remove the numerical dimensions or numerical location of the object, if present. '

    return prompt

def get_single_part_summary_prompt(caption, obj_name, max_len):
    prompt = f'Summarize this description of a single part of a {obj_name} to less than {max_len} words. '
    prompt += f'Output only the summarized description. '
    prompt += f'Keep uniquely identifying information such as the color, texture, material of the part. '
    prompt += f'Remove any common sense information about the part. '
    prompt += caption
    return prompt

def get_part_caps_summary_prompt(part_caps, obj_name):
    prompt = f'These are descriptions of a single part of a {obj_name} in a scene, generated from multiple views of the part. '
    
    for cap in part_caps:
        prompt += f'{cap}. '

    prompt += 'Summarize the descriptions into a single description. '
    prompt += 'Retain all the details that are likely to be correct in the output description. '
    prompt += 'Output only the summarized description that describes the part, without mentioning the images or using the word "scene" or "object". '
    prompt += 'Do not output anything other than the description.'

    return prompt


def create_scene_summary_prompt(caps):
    # summarize all image_captions into a single scene caption
    prompt = 'These are descriptions of a single 3D indoor scene generated from different viewpoints.'
    for img_cap in caps:
        prompt += f'{img_cap}. '

    prompt += 'Summarize the descriptions into a single description. '
    prompt += 'Retain all the details that are likely to be correct in the output description. '
    prompt += 'Output only the summarized description that describes the scene, without mentioning the images or using the word "scene". '
    prompt += 'Do not output anything other than the description.'

    return prompt

def summarize_captions(caption_data, summary_model, cfg):
    '''
    caption_data: caption data for one scene
    '''
    # summarize for each object
    for obj_id in caption_data['objects']:
        obj_name = caption_data['objects'][obj_id]['label']

        cap_types = []
        if cfg.gen_global:
            cap_types.append('global_caption')
        if cfg.gen_local:
            cap_types.append('local_caption')

        # summarize within each view
        if cfg.gen_multiple_per_view and cfg.summarize_per_view: 
            # summarize within local_caption_samples, global_caption_samples
            for cap_type in cap_types:
                for image_name in caption_data['objects'][str(obj_id)][f'{cap_type}_samples']:
                    caps = caption_data['objects'][str(obj_id)][f'{cap_type}_samples'][image_name]
                    if 'local' in cap_type:
                        prompt = create_local_multiview_summary_prompt(caps, obj_name, single_image=True)
                    elif 'global' in cap_type:
                        prompt = create_global_multiview_summary_prompt(caps, obj_name, single_image=True)
                    summarized_cap = summary_model(prompt)
                    caption_data['objects'][str(obj_id)][cap_type][image_name] = summarized_cap

        # summarize over all views
        for cap_type in cap_types:
            # summarize all the local caps
            if cfg.summarize_per_view: 
                # got 1 caption per view, summarize all together
                local_caps = list(caption_data['objects'][str(obj_id)][cap_type].values())
                local_summary_prompt = create_local_multiview_summary_prompt(local_caps, obj_name)
                local_summary = summary_model(local_summary_prompt)
                caption_data['objects'][str(obj_id)][f'summarized_{cap_type}'] = local_summary
                for max_len in cfg.summarize_length:
                    caption_data['objects'][str(obj_id)][f'summarized_{cap_type}_{max_len}'] = summary_model(get_obj_local_summary_prompt(local_summary, obj_name, max_len))
            else: 
                # got n caption samples per view, summarize i-th from each view into the i-th final caption
                caption_data['objects'][str(obj_id)][f'summarized_{cap_type}'] = []
                for max_len in cfg.summarize_length:
                    caption_data['objects'][str(obj_id)][f'summarized_{cap_type}_{max_len}'] = []
                for sample_ndx in range(cfg.num_gen_per_view):
                    caps = [caption_data['objects'][str(obj_id)][f'{cap_type}_samples'][image_name][sample_ndx] for image_name in caption_data['objects'][str(obj_id)][f'{cap_type}_samples']]
                    prompt = create_local_multiview_summary_prompt(caps, obj_name)
                    summarized_cap = summary_model(prompt)
                    caption_data['objects'][str(obj_id)][f'summarized_{cap_type}'].append(summarized_cap)
                    for max_len in cfg.summarize_length:
                        caption_data['objects'][str(obj_id)][f'summarized_{cap_type}_{max_len}'].append(summary_model(get_obj_local_summary_prompt(summarized_cap, obj_name, max_len)))

        # check if the object has parts and sumarize all the part captions
        # pseudoGT parts / instance pred parts
        if 'parts' in caption_data['objects'][obj_id] or 'parts_captions' in caption_data['objects'][obj_id]:
            # used pseudoGT part segments -> each part has a caption
            if not cfg.use_instance_preds:
                for part_id in caption_data['objects'][obj_id]['parts']:
                    part_crop_caps = list(caption_data['objects'][obj_id]['parts'][part_id]['crop_captions'].values())
                    part_full_caps = list(caption_data['objects'][obj_id]['parts'][part_id]['full_captions'].values())

                    prompt = get_part_caps_summary_prompt(part_crop_caps + part_full_caps, obj_name)

                    # get 1 summary per part
                    summarized_cap = summary_model(prompt)
                    caption_data['objects'][obj_id]['parts'][part_id]['summarized_caption'] = summarized_cap

                    for max_len in cfg.summarize_length:
                        caption_data['objects'][obj_id]['parts'][part_id][f'summarized_caption_{max_len}'] = summary_model(get_single_part_summary_prompt(summarized_cap, obj_name, max_len)) 

                # 1 summary over all the parts 
                all_parts_caps = [caption_data['objects'][obj_id]['parts'][part_id]['summarized_caption'] for part_id in caption_data['objects'][obj_id]['parts']]
            else:
                # when using instance preds, we already have the summarized_parts_caption
                # summarize the "parts_captions" over each view
                all_parts_caps = [caption_data['objects'][obj_id]['parts_captions'].values()]

            # doesnt matter if the descriptions are of different parts, or different views of all parts -> prompt is a list of captions
            caption_data['objects'][obj_id]['summarized_parts_caption'] = summary_model(get_joint_parts_summary_prompt(all_parts_caps, obj_name))

            for max_len in cfg.summarize_length:
                caption_data['objects'][obj_id][f'summarized_parts_caption_{max_len}'] = summary_model(get_joint_parts_summary_prompt(all_parts_caps, obj_name, max_len))