'''
Generate object captions using VLM

for each scene, go through the subsample list of images
rasterize the mesh onto the image to get 3d-2d mapping
go through the list of objects in the scene (ignore the specified objects)
check if obj_visible_thresh fraction of the object is visible in the image
get the 2d bbox of the object in that image and crop it 
run the cropped image through the VLM model to get the caption
for each scene save:
    {objid -> {imageid -> caption}}
'''

from codetiming import Timer

from scipy.spatial import KDTree
from omegaconf import DictConfig
from collections import defaultdict
from pathlib import Path
import hydra
from omegaconf import DictConfig

from tqdm import tqdm
import open3d as o3d
import torch
import numpy as np
import cv2

from datagen.caption_utils import LMDeploy_Dummy_Model, LMDeploy_Model, create_global_parts_prompt, create_img_prompt, create_part_prompt, load_instance_preds, rasterize_o3d_mesh_to_image, summarize_captions
from utils.img_utils import save_img, load_image, get_img_crop


from scannetpp.common.utils.dslr import compute_undistort_intrinsic
from scannetpp.common.utils.colmap import get_camera_images_poses, camera_to_intrinsic
from scannetpp.common.utils.anno import get_bboxes_2d, get_visiblity_from_cache, get_vtx_prop_on_2d, load_anno_wrapper
from scannetpp.common.file_io import load_json, read_txt_list, write_json
from scannetpp.common.scene_release import ScannetppScene_Release

def save_image(image, path):# 
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def viz_ids(img, img_ids, out_path):
    viz_img = np.zeros_like(img)
    unique_ids = np.unique(img_ids)
    # ignore negative and 0
    unique_ids = unique_ids[unique_ids > 0]

    for obj_id in unique_ids:
        mask = img_ids == obj_id
        color = np.random.randint(0, 255, size=3)
        viz_img[mask] = color
    
    save_image(viz_img, out_path)

@hydra.main(version_base=None, config_path="../configs", config_name="gen_captions")
def main(cfg : DictConfig) -> None:
    print('Config:', cfg)

    run_name = ''
    # get scene list
    scene_list = read_txt_list(cfg.scene_list_file)
    print('Scenes in list:', len(scene_list))

    if cfg.get('filter_scenes'):
        scene_list = [s for s in scene_list if s in cfg.filter_scenes]
        print('Filtered scenes:', len(scene_list))
    if cfg.get('exclude_scenes'):
        scene_list = [s for s in scene_list if s not in cfg.exclude_scenes]
        print('After excluding scenes:', len(scene_list))

    if cfg.use_existing_dir is not None:
        print('Use existing dir')
        save_dir = Path(cfg.use_existing_dir)
    else:
        # root + runname + savedir
        save_dir_prefix = f'{run_name}_' if run_name else ''
        save_dir = Path(cfg.save_dir_root) / f'{save_dir_prefix}{cfg.save_dir}'
        save_dir.mkdir(parents=True, exist_ok=True)
        print('Save to dir:', save_dir)

    img_crop_dir =  save_dir / 'img_crops'
    bbox_img_dir =  save_dir / 'img_bbox'
    part_crop_dir =  save_dir / 'part_crops'
    part_bbox_img_dir =  save_dir / 'part_bbox'
    perscene_out_path = save_dir / 'scene_captions'
    viz_obj_ids_dir = save_dir / 'viz_obj_ids'

    for dir in [img_crop_dir, bbox_img_dir, part_crop_dir, part_bbox_img_dir, perscene_out_path]:
        dir.mkdir(parents=True, exist_ok=True)

    # keep iphone and dslr data separate
    rasterout_dir = Path(cfg.rasterout_dir) / cfg.image_type
    rasterout_dir.mkdir(parents=True, exist_ok=True)

    object_parts_dir = Path(cfg.object_parts_dir) if cfg.object_parts_dir else None

    # create models
    with Timer():
        caption_model = LMDeploy_Dummy_Model() if cfg.dbg.dont_load_models else LMDeploy_Model(cfg.caption_model) 

    # go through scenes
    for scene_id in tqdm(scene_list, desc='scene'):
        print(f'Running on scene: {scene_id}')
        caption_out_path = perscene_out_path / f'{scene_id}.json'
        if cfg.skip_existing and caption_out_path.exists():
            print('Skipping existing:', caption_out_path)
            continue

        scene = ScannetppScene_Release(scene_id, data_root=cfg.data_root)
        # get object ids on the mesh vertices
        # objects, vertex_obj_ids
        if cfg.use_instance_preds:
            anno = load_instance_preds(scene_id, cfg.instance_preds_dir, cfg.sem_classes_file)
        else:
            anno = load_anno_wrapper(scene)

        vtx_obj_ids = anno['vertex_obj_ids']
        if cfg.filter_objects:
            # keep only these in vtx_obj_ids and anno['objects']
            keep_ndx = np.isin(vtx_obj_ids, cfg.filter_objects)
            vtx_obj_ids[~keep_ndx] = 0
            anno['objects'] = {obj_id: v for obj_id, v in anno['objects'].items() if int(obj_id) in cfg.filter_objects}

        if cfg.check_visibility:
            visibility_data = get_visiblity_from_cache(scene, rasterout_dir, cfg.visiblity_cache_dir, cfg.image_type, cfg.subsample_factor, cfg.undistort_dslr, anno=anno)
            if cfg.create_visiblity_cache_only:
                print(f'Created visibility cache for {scene_id}')
                continue

        # read mesh
        mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path)) 

        # set vtx_obj_ids to 0 for obj ids with names in cfg.ignore_object_labels
        if cfg.ignore_object_labels:
            for obj_id, obj in anno['objects'].items():
                if obj['label'] in cfg.ignore_object_labels:
                    vtx_obj_ids[vtx_obj_ids == int(obj_id)] = 0
        
        obj_ids = np.unique(vtx_obj_ids)
        # remove 0
        obj_ids = sorted(obj_ids[obj_ids != 0])

        # need 3d object locations and neighbors
        if cfg.global_prompt_obj_locations:
            # get nearest objects for each object
            # store all obj locations here
            obj_id_locations = {obj_id: anno['objects'][obj_id]['obb']['centroid'] for obj_id in obj_ids}
            obj_id_dims = {obj_id: anno['objects'][obj_id]['obb']['axesLengths'] for obj_id in obj_ids}
        
            # obj ids and locations in a fixed order for building kdtree
            obj_locations = np.array([obj_id_locations[obj_id] for obj_id in obj_ids])
            obj_tree = KDTree(obj_locations)

            # get all nearest neighbors in 3d space
            _, nearest_obj_ndx = obj_tree.query(obj_locations, k=len(obj_ids))
            nearest_obj_ndx = nearest_obj_ndx[:, 1:]
            # store in dict, i = index into obj_ids, n = each obj ndx in nearest obj ndxs
            nearest_objects = {obj_id: [obj_ids[n] for n in nearest_obj_ndx[i]] for i, obj_id in enumerate(obj_ids)}

        if cfg.limit_objects is not None:
            # pick the objects to caption
            caption_limit_obj_ids = np.unique(vtx_obj_ids)
            # exclude 0
            caption_limit_obj_ids = sorted(caption_limit_obj_ids[caption_limit_obj_ids != 0])
            # keep only the first cfg.limit_objects
            caption_limit_obj_ids = caption_limit_obj_ids[:cfg.limit_objects]
        elif cfg.filter_objects is not None:
            # keep only the objects that are in the list
            caption_limit_obj_ids = list(set(obj_ids).intersection(set(cfg.filter_objects)))

        # get the list of iphone/dslr images and poses
        colmap_camera, image_list, poses, distort_params = get_camera_images_poses(scene, cfg.subsample_factor, cfg.image_type)
        distort_params_orig = distort_params.copy()
        
        all_poses = {image_name: pose for image_name, pose in zip(image_list, poses)}
        # keep first 4 elements for cv2 undistortion
        # use original distort_params for pytorch3d rendering
        distort_params = distort_params[:4]

        intrinsic = camera_to_intrinsic(colmap_camera)
        img_height, img_width = colmap_camera.height, colmap_camera.width

        undistort_map1, undistort_map2 = None, None
        if cfg.image_type == 'dslr' and cfg.undistort_dslr:
            undistort_intrinsic = compute_undistort_intrinsic(intrinsic, img_height, img_width, distort_params)
            undistort_map1, undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                intrinsic, distort_params, np.eye(3), undistort_intrinsic, (img_width, img_height), cv2.CV_32FC1
            )

        scene_captions = defaultdict(dict)

        if cfg.gen_part_caps and not cfg.use_instance_preds:
            # load scene_id.json, then load scene_id.pth
            obj_to_parts = load_json(object_parts_dir / f'{scene_id}.json')
            vtx_part_ids = torch.load(object_parts_dir / f'{scene_id}.pth')

        if cfg.limit_images:
            print(f'Limiting images to: {cfg.limit_images}')
            image_list = image_list[:cfg.limit_images]

        print(f'Num images: {len(image_list)}')
        # go through list of images
        for _, image_name in enumerate(tqdm(image_list, desc='image', leave=False)):
            if cfg.image_type == 'iphone':
                image_dir = scene.iphone_rgb_dir
            elif cfg.image_type == 'dslr':
                image_dir = scene.dslr_resized_dir
            # load the image H, W, 3
            img_path = str(image_dir / image_name)
            if not Path(img_path).exists():
                print('Image not found:', img_path)
                continue
            
            print(f'Processing image: {image_name}')
            img = load_image(img_path) 

            rasterout_path = rasterout_dir / scene_id / f'{image_name}.pth'
            raster_out_dict = torch.load(rasterout_path)

            # if dimensions dont match, raster is from downsampled image
            # upsample using nearest neighbor
            pix_to_face = raster_out_dict['pix_to_face'].squeeze().cpu()
            rasterized_dims = list(pix_to_face.shape)

            if rasterized_dims != [img_height, img_width]:
                # upsample pixtoface and zbuf
                pix_to_face = torch.nn.functional.interpolate(pix_to_face.unsqueeze(0).unsqueeze(0).float(),
                                                              size=(img_height, img_width), mode='nearest').squeeze().squeeze().long()
            pix_to_face = pix_to_face.numpy()

            if cfg.use_rasterized_mesh_image:
                # rasterize the mesh onto this camera pose and use that instead of the dslr / iphone image
                # need to rasterize it again!
                pose = all_poses[image_name]
                # NOTE! overwrites "img" with the rasterized image
                img = rasterize_o3d_mesh_to_image(mesh, pose, intrinsic, img_width, img_height, distort_params_orig)

            if undistort_map1 is not None and undistort_map2 is not None:
                # apply undistortion to rasterization (nearest neighbor), zbuf (linear) and image (linear)
                pix_to_face = cv2.remap(pix_to_face, undistort_map1, undistort_map2, 
                    interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
                )
                # img is np
                img = cv2.remap(img, undistort_map1, undistort_map2,
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )

            # get object IDs on image
            pix_obj_ids = get_vtx_prop_on_2d(pix_to_face, vtx_obj_ids, mesh)

            if cfg.dbg.viz_obj_ids:
                out_path = viz_obj_ids_dir / scene_id / f'{image_name}.png'
                viz_ids(img, pix_obj_ids, out_path)

            # get objid -> bbox x,y,w,h after upsampling rasterization
            bboxes_2d = get_bboxes_2d(pix_obj_ids)

            # keep only bboxes with w and h atleast cfg.bbox_min_side_pix
            bboxes_2d = {k: [x, y, w, h] for k, [x, y, w, h] in bboxes_2d.items() if w >= cfg.bbox_min_side_pix and h >= cfg.bbox_min_side_pix} 

            # keep only objid > 0
            bboxes_2d = {k: v for k, v in bboxes_2d.items() if k > 0}

            if cfg.limit_objects is not None or cfg.filter_objects is not None:
                # keep only obj ids in caption_limit_obj_ids
                bboxes_2d = {k: v for k, v in bboxes_2d.items() if k in caption_limit_obj_ids}

            print(f'Num objects in image: {len(bboxes_2d)}')

            # go through each object that has a bbox -> excludes the objects from earlier
            for _, (obj_id, obj_bbox) in enumerate(tqdm(bboxes_2d.items(), desc='obj', leave=False)):
                obj_label = anno['objects'][obj_id]['label']
                # check if enough % of the objs vertices are in the image
                obj_mask_3d = vtx_obj_ids == obj_id

                if cfg.check_visibility:
                    # enough of the object is seen
                    if visibility_data['images'][image_name]['objects'][obj_id].get('visible_vertices_frac', 0) < cfg.obj_visible_thresh:
                        print(f'Object {obj_id} not visible enough in image {image_name}')
                        continue
                    
                    # check if obj occupies enough % of the image
                    if visibility_data['images'][image_name]['objects'][obj_id].get('visible_pixels_frac', 0) < cfg.obj_pixel_thresh:
                        print(f'Object {obj_id} occupies too few pixels in image {image_name}')
                        continue

                    if visibility_data['images'][image_name]['objects'][obj_id].get('zbuf_min', 9999) > cfg.obj_dist_thresh:
                        # object is too far away from camera
                        print(f'Object {obj_id} too far away in image {image_name}')
                        continue
                
                    if cfg.visibility_topk is not None:
                        images_visibilites = []
                        for i_name in visibility_data['images']:
                            if obj_id in visibility_data['images'][i_name]['objects'] and 'visible_vertices_frac' in visibility_data['images'][i_name]['objects'][obj_id]:
                                images_visibilites.append((i_name, visibility_data['images'][i_name]['objects'][obj_id]['visible_vertices_frac']))
                        # sort descending by visibility
                        images_visibilites.sort(key=lambda x: x[1], reverse=True)
                        top_images = [i_name for i_name, _ in images_visibilites][:cfg.visibility_topk]
                        print(f'Top images for obj {obj_id}: {top_images}')
                        print(f'Num images and visibilities for obj {obj_id} and image {image_name}: {len(images_visibilites)}')
                        # dont consider this object in this image
                        if image_name not in top_images: 
                            print(f'Object {obj_id} not in top {cfg.visibility_topk} images in image {image_name}')
                            continue
                ########################## create placeholders for this obj's data ##########################
                if str(obj_id) not in scene_captions['objects']:
                    scene_captions['objects'][str(obj_id)] = defaultdict(dict)
                ########################## part caps ##########################
                # cant generate part caps with instance preds, use a global prompt instead
                if cfg.gen_part_caps and (not cfg.use_instance_preds) and str(obj_id) in obj_to_parts:
                    # get part ids for only current object using obj_mask_3d, set everything else to -1
                    vtx_part_ids_copy = vtx_part_ids.copy()
                    vtx_part_ids_copy[~obj_mask_3d] = -1
                    # part ids in current image
                    pix_part_ids = get_vtx_prop_on_2d(pix_to_face, vtx_part_ids_copy, mesh)
                    # get part id -> bbox 
                    part_bboxes_2d = get_bboxes_2d(pix_part_ids)

                    # go over all the parts
                    for part_id, part_bbox in tqdm(part_bboxes_2d.items(), desc='part', leave=False):
                        # create crop of the part
                        part_crop = get_img_crop(img, part_bbox, 0.05, expand_bbox=True)
                        if part_crop.size == 0: # part not visible in the image
                            continue
                        part_crop_path = part_crop_dir / scene_id / f'{image_name}_obj{obj_id}_part{part_id}.png'
                        if not cfg.dbg.dont_load_models: # dont need to save images for debugging
                            save_img(part_crop, part_crop_path)
                        # get the caption for the part using the object label
                        part_crop_prompt = create_part_prompt(obj_label)
                        part_crop_caption = caption_model(part_crop_prompt, str(part_crop_path))

                        # draw part bbox on full image
                        img_copy = img.copy()
                        x, y, w, h = part_bbox
                        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                        cv2.rectangle(img_copy, (y, x), (y+h, x+w), (0, 0, 255), 2)
                        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                        # crop out only the object
                        img_copy = get_img_crop(img_copy, obj_bbox, 0.05, expand_bbox=True)
                        part_full_image_path = part_bbox_img_dir / scene_id / f'{image_name}_obj{obj_id}_part{part_id}.png'
                        if not cfg.dbg.dont_load_models: 
                            save_img(img_copy, part_full_image_path)

                        # save to file
                        part_full_prompt = create_part_prompt(obj_label, bbox=True)
                        part_full_caption = caption_model(part_full_prompt, str(part_full_image_path))

                        if 'parts' not in scene_captions['objects'][str(obj_id)]:
                            scene_captions['objects'][str(obj_id)]['parts'] = defaultdict(dict)
                        if str(part_id) not in scene_captions['objects'][str(obj_id)]['parts']:
                            scene_captions['objects'][str(obj_id)]['parts'][str(part_id)]['bboxes_2d'] = defaultdict(dict)
                            scene_captions['objects'][str(obj_id)]['parts'][str(part_id)]['crop_prompt'] = defaultdict(dict)
                            scene_captions['objects'][str(obj_id)]['parts'][str(part_id)]['crop_captions'] = defaultdict(dict)
                            scene_captions['objects'][str(obj_id)]['parts'][str(part_id)]['full_prompt'] = defaultdict(dict)
                            scene_captions['objects'][str(obj_id)]['parts'][str(part_id)]['full_captions'] = defaultdict(dict)

                        scene_captions['objects'][str(obj_id)]['parts'][str(part_id)]['bboxes_2d'][image_name] = part_bbox

                        scene_captions['objects'][str(obj_id)]['parts'][str(part_id)]['crop_prompt'][image_name] = part_crop_prompt
                        scene_captions['objects'][str(obj_id)]['parts'][str(part_id)]['crop_captions'][image_name] = part_crop_caption

                        scene_captions['objects'][str(obj_id)]['parts'][str(part_id)]['full_prompt'][image_name] = part_full_prompt
                        scene_captions['objects'][str(obj_id)]['parts'][str(part_id)]['full_captions'][image_name] = part_full_caption
                        # one part of the object done
                ################### part caps done ############################
                
                ################### obj caps ############################
                if not cfg.gen_obj_caps: 
                    continue

                print(f'Captioning obj: {obj_id} in image: {image_name}')
                # crop the image
                img_crop = get_img_crop(img, obj_bbox, cfg.bbox_expand_factor, expand_bbox=True)
                img_crop_path = img_crop_dir / scene_id / f'{image_name}_{obj_id}.png'
                if not cfg.dbg.dont_load_models:
                    save_img(img_crop, img_crop_path)

                # prepare prompt
                if cfg.global_prompt_obj_locations:
                    obj_location_3d = np.round(obj_id_locations[obj_id], 2).tolist()
                    obj_dims = np.round(obj_id_dims[obj_id], 2).tolist()
                    all_nearest_obj_ids = nearest_objects[obj_id]
                    # keep only the ones visible in the image and not the object itself
                    # keep the same order as in nearest_objects
                    current_nearest_obj_ids = [oid for oid in all_nearest_obj_ids if oid in objs_in_image and oid != obj_id]
                else:
                    obj_location_3d, obj_dims, current_nearest_obj_ids = None, None, []

                x, y, w, h = obj_bbox
                # center of the bbox
                obj_location_2d = np.round([x + w/2, y + h/2]).tolist()
                # get nearest object ids and locations visible in this image
                # all objects visible in this image, ordered by distance
                objs_in_image = bboxes_2d.keys()

                if cfg.nearest_objects != 'all':
                    # atmost top k images **visible in this image**
                    current_nearest_obj_ids = current_nearest_obj_ids[:cfg.nearest_objects]
                
                current_nearest_obj_names = [anno['objects'][oid]['label'] for oid in current_nearest_obj_ids]
                if cfg.global_prompt_obj_locations:
                    current_nearest_obj_locations3d = [obj_id_locations[obj_id] for obj_id in current_nearest_obj_ids]

                # use 2d locations (bbox centers in the image) in the prompt 
                current_nearest_obj_locations2d = []
                for oid in current_nearest_obj_ids:
                    x, y, w, h = bboxes_2d[oid]
                    # center of the bbox
                    current_nearest_obj_locations2d.append([x + w/2, y + h/2])

                # NOTE: might not have any context objects in the image
                near_objs_prompt = ''
                if cfg.global_prompt_obj_locations:
                    for obj_ndx, (obj_name, obj_loc3d, obj_loc2d) in enumerate(zip(current_nearest_obj_names, current_nearest_obj_locations3d, current_nearest_obj_locations2d)):
                        # round loc to 2 decimal places
                        obj_loc_round_3d = np.round(obj_loc3d, 2).tolist()
                        obj_loc_round_2d = np.round(obj_loc2d, 2).tolist()

                        # use 2d, 3d or 2d+3d locations in the prompt
                        near_objs_prompt += f' {obj_name}' 
                        if cfg.global_prompt_obj_locations == '3d':
                            near_objs_prompt += f' located at {obj_loc_round_3d}'
                        elif cfg.global_prompt_obj_locations == '3d_2d':
                            near_objs_prompt += f' located at {obj_loc_round_3d} in 3D and at {obj_loc_round_2d} in 2D'
                        elif cfg.global_prompt_obj_locations == '2d':
                            near_objs_prompt += f' located at {obj_loc_round_2d}'

                        if obj_ndx != len(current_nearest_obj_names) - 1:
                            near_objs_prompt += ","

                scene_captions['objects'][str(obj_id)]['label'] = obj_label
                
                # update per-image properties: bbox of this object in this image
                scene_captions['objects'][str(obj_id)]['bboxes_2d'][image_name] = obj_bbox

                # which image to use for global prompt
                if cfg.global_prompt_obj_bbox:
                    # create new image with bbox of the object draw on the full image
                    img_copy = img.copy()
                    x, y, w, h = obj_bbox
                    # BGR?
                    # convert image RGB to BGR
                    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(img_copy, (y, x), (y+h, x+w), (0, 0, 255), 2)
                    # convert back to RGB
                    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                    global_prompt_img_path = bbox_img_dir / scene_id / f'{image_name}_{obj_id}.png'
                    # save it to file
                    if not cfg.dbg.dont_load_models:
                        save_img(img_copy, global_prompt_img_path)
                else:
                    global_prompt_img_path = img_path

                ##### run VLM and get caption #####
                # get prompts
                if cfg.gen_local:
                    local_prompt = create_img_prompt(obj_label, obj_location_3d, obj_location_2d, obj_dims, None, cfg, local_only=True)
                    scene_captions['objects'][str(obj_id)]['local_prompt'][image_name] = local_prompt
                if cfg.gen_global:
                    global_prompt = create_img_prompt(obj_label, obj_location_3d, obj_location_2d, obj_dims, near_objs_prompt, cfg, global_only=True)
                    scene_captions['objects'][str(obj_id)]['global_prompt'][image_name] = global_prompt

                # get captions
                if cfg.gen_multiple_per_view:
                    if cfg.gen_local:
                        local_caption_samples = [caption_model(local_prompt, str(img_crop_path), sample=True) for _ in range(cfg.num_gen_per_view)]
                        scene_captions['objects'][str(obj_id)]['local_caption_samples'][image_name] = local_caption_samples
                    if cfg.gen_global:
                        global_caption_samples = [caption_model(global_prompt, str(global_prompt_img_path), sample=True) for _ in range(cfg.num_gen_per_view)]
                        scene_captions['objects'][str(obj_id)]['global_caption_samples'][image_name] = global_caption_samples
                else:
                    if cfg.gen_local:
                        local_caption = caption_model(local_prompt, str(img_crop_path))
                        scene_captions['objects'][str(obj_id)]['local_caption'][image_name] = local_caption
                    if cfg.gen_global:
                        global_caption = caption_model(global_prompt, str(global_prompt_img_path))
                        scene_captions['objects'][str(obj_id)]['global_caption'][image_name] = global_caption

                if cfg.use_instance_preds and cfg.gen_part_caps:
                    # generate a single parts caption directly, no part segments
                    global_parts_prompt = create_global_parts_prompt(obj_label)
                    global_parts_caption = caption_model(global_parts_prompt, str(img_crop_path))
                    # save it to parts_captions which has one caption for each image
                    scene_captions['objects'][str(obj_id)]['parts_captions'][image_name] = global_parts_caption
                # one object done in this image
            # one image done

        # summarize everything
        with Timer():
            del caption_model
            torch.cuda.empty_cache()
            print(f'Deleted caption model: {cfg.caption_model}')

            summary_model = LMDeploy_Dummy_Model() if cfg.dbg.dont_load_models else LMDeploy_Model(cfg.summary_model)

            summarize_captions(scene_captions, summary_model, cfg)
            print(f'Summarized captions for scene: {scene_id}')
            del summary_model
            torch.cuda.empty_cache()
            print(f'Deleted summary model: {cfg.summary_model}')

            caption_model = LMDeploy_Dummy_Model() if cfg.dbg.dont_load_models else LMDeploy_Model(cfg.caption_model)
        
        # save captions for this scene
        print('Saving captions to:', caption_out_path)
        print(f'Num objects with captions: {len(scene_captions["objects"])}')
        write_json(caption_out_path, scene_captions)

if __name__ == "__main__":
    main()