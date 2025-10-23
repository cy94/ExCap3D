'''
Project dino features to 3D
'''
from codetiming import Timer
from sklearn.decomposition import PCA
import open3d as o3d
import cv2
from tqdm import tqdm
import numpy as np
from backproj_dino_feats import project_image_to_pc
from scannetpp.common.utils.colmap import camera_to_intrinsic, get_camera_intrinsics
from scannetpp.common.utils.dslr import adjust_intrinsic_matrix
from scannetpp.common.utils.rasterize import undistort_rasterization
from scannetpp.dslr.undistort import compute_undistort_intrinsic
from scannetpp.common.utils.anno import get_best_views_from_cache
from third_party.dino_vit.extractor import ViTExtractor
import torch
from pathlib import Path
from scannetpp.common.scene_release import ScannetppScene_Release
from scipy.spatial import KDTree
import pytorch_lightning as pl


class Dino(pl.LightningModule):
    def __init__(self,
                data_root,
                raster_dir,
                n_images,
                image_type,
                subsample_factor,
                layer,
                feat_dim,
                downsample_factor,
                nn_thresh,
                model_type,
                stride,
                load_size,
                nn_cache_dir,
                colmap_cache_dir,
                dbg_viz,
                dbg_viz_dir,
                best_view_cache_dir
                ):
        super().__init__()
        # store all the params
        self.dbg_viz = dbg_viz
        self.dbg_viz_dir = Path(dbg_viz_dir)

        self.data_root = Path(data_root)
        self.raster_dir = Path(raster_dir)
        self.best_view_cache_dir = Path(best_view_cache_dir) / image_type

        self.n_images = n_images
        self.image_type = image_type
        self.subsample_factor = subsample_factor
        self.feat_dim = feat_dim
        self.max_feats_per_point = self.n_images # accumulate everything
        self.downsample_factor = downsample_factor
        self.nn_thresh = nn_thresh
        self.load_size = load_size
        self.layer = layer
        self.nn_cache_dir = Path(nn_cache_dir)
        self.colmap_cache_dir = Path(colmap_cache_dir)
        # dont save to ckpt!
        self.feature_extractor = ViTExtractor(model_type, stride)
        print(f'Created dino model: {model_type}')

    def forward(self, batchdata, scene_ids):
        # batch: contains the original points and mapping to voxelized points
        dino_feats_3d = []

        hires_coords_all = batchdata.original_coordinates
        unique_maps = batchdata.unique_maps

        zip_list = zip(scene_ids, hires_coords_all)
        # each scene
        for (scene_id, hires_coords) in zip_list:
            scene = ScannetppScene_Release(scene_id, self.data_root)
            colmap_camera, distort_params = get_camera_intrinsics(scene, self.image_type)    
            colmap_cache_data = torch.load(self.colmap_cache_dir / f'{scene_id}.pth') # dont read the huge colmap file, cache poses and image list
            orig_image_list, orig_poses = colmap_cache_data['image_names'], colmap_cache_data['poses']
            image_to_pose = {image: pose for image, pose in zip(orig_image_list, orig_poses)}

            # use undistorted dslr
            image_list = get_best_views_from_cache(scene, self.best_view_cache_dir, self.raster_dir / self.image_type, self.image_type, self.subsample_factor, True)
            # get poses according to this image list
            poses = [image_to_pose[image] for image in image_list]

            distort_params = distort_params[:4]
            intrinsic = camera_to_intrinsic(colmap_camera)
            img_height, img_width = colmap_camera.height, colmap_camera.width
            # keep atleast 1 image
            selected_images = image_list[:self.n_images]
            selected_poses = poses[:self.n_images]   
            img_dir = scene.dslr_resized_dir

            # rasterization is on the downsampled image
            rasterized_dims = img_height // self.downsample_factor, img_width // self.downsample_factor

            undistort_intrinsic = compute_undistort_intrinsic(intrinsic, rasterized_dims[0], rasterized_dims[1], distort_params)
            undistort_map1, undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                adjust_intrinsic_matrix(intrinsic, self.downsample_factor), distort_params, np.eye(3), 
                undistort_intrinsic, (rasterized_dims[1], rasterized_dims[0]), cv2.CV_32FC1
            )

            # init empty feats on points
            feats3d = torch.zeros((hires_coords.shape[0], self.max_feats_per_point, self.feat_dim)).to(self.device)
            has_feats_3d = torch.zeros((hires_coords.shape[0], self.max_feats_per_point)).to(self.device)

            for image_ndx, (image_name, image_pose) in enumerate(zip(selected_images, selected_poses)):
                img_path = img_dir / f'{image_name}'

                img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB) # h,w,3 -> orig = 1168,1752,3
                
                # resize image to raster size
                img = cv2.resize(img, rasterized_dims[::-1], interpolation=cv2.INTER_LANCZOS4) # 292, 438, 3

                img = cv2.remap(img, undistort_map1, undistort_map2,
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )

                # resize image to load size
                img = cv2.resize(img, self.load_size[::-1], interpolation=cv2.INTER_LANCZOS4) # 224,336,3

                # load image 
                image_batch = self.feature_extractor.preprocess_image(img) #1,3,224,336

                # get dino feats
                with torch.no_grad():
                    descriptors = self.feature_extractor.extract_descriptors(image_batch.to(self.device), self.layer, 'key', False) # keep on gpu, as tensor
                    
                num_patches = self.feature_extractor.num_patches # gets set after calling extract features!
                # reshape to h,w, dim
                features2d = descriptors.reshape(1, num_patches[0], num_patches[1], self.feat_dim) # 1hwc

                # upsample feats to raster dimensions, input is nchw
                features2d = torch.nn.functional.interpolate(features2d.permute(0, 3, 1, 2), size=rasterized_dims, mode='bilinear').squeeze() # chw
                features2d = features2d.permute(1, 2, 0) # h,w,c

                nn_cache_path = self.nn_cache_dir / scene_id / f'{image_name}.pth'

                if nn_cache_path.exists():
                    (ndx, valid_ndx, img_coords) = torch.load(nn_cache_path)
                else:
                    raster_data = torch.load(self.raster_dir / self.image_type / scene_id / f'{image_name}.pth', weights_only=True)
                    pix_to_face = raster_data['pix_to_face'].squeeze()
                    zbuf = raster_data['zbuf'].squeeze()
                    # undistort
                    pix_to_face, zbuf = undistort_rasterization(pix_to_face.cpu().numpy(), zbuf.cpu().numpy(), undistort_map1, undistort_map2)
                    pix_to_face = torch.LongTensor(pix_to_face).to(self.device)
                    zbuf = torch.Tensor(zbuf).to(self.device)
                    # 3d points and corresponding 2d indices, use tensors
                    points3d, img_coords = project_image_to_pc(image_pose, undistort_intrinsic, zbuf, pix_to_face)

                    if self.dbg_viz:
                        pc = o3d.geometry.PointCloud()
                        pc.points = o3d.utility.Vector3dVector(np.array(points3d.cpu()))
                        out_path = self.dbg_viz_dir / f'{scene_id}_{image_name}_m3dpoints.ply'
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        print(f'Saving viz m3dpoints to: {out_path}')
                        o3d.io.write_point_cloud(str(out_path), pc)

                    # build tree over points3d 
                    tree = KDTree(points3d.cpu().numpy())
                    _, ndx = tree.query(hires_coords, distance_upper_bound=self.nn_thresh)
                    valid_ndx = ndx != points3d.shape[0] # numpoints in data -> points which have a neighbor

                    # save to cache
                    nn_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save((ndx, valid_ndx, img_coords), nn_cache_path)

                # get feats for these image coords
                selected_feats = features2d[img_coords[:, 1], img_coords[:, 0]]
                # get the feats for these points and store
                try:
                    feats3d[valid_ndx, image_ndx] = selected_feats[ndx[valid_ndx]]
                    has_feats_3d[valid_ndx, image_ndx] = 1
                except IndexError: # something wrong with the saved index, doesnt correspond to the points in the scene -> when 3D scene is cropped!
                    continue

            # max pool features over all views
            feats3d = torch.max(feats3d, dim=1).values
            has_feats_3d = torch.max(has_feats_3d, dim=1).values
            # get feats on low rescoords
            feats3d_lowres = feats3d[unique_maps]
            dino_feats_3d.append(feats3d_lowres)

            if self.dbg_viz:
                # pca of points that have features and viz
                # PC and viz feats3d on points and mesh, only for points that have feats
                has_feats_3d_np = has_feats_3d.cpu().numpy().astype(bool).flatten()
                feats3d_np = feats3d.cpu().numpy()[has_feats_3d_np]
                pca = PCA(n_components=3)
                pca.fit(feats3d_np)
                pca_feats_3d = pca.transform(feats3d_np)
                pca_feats_3d = (pca_feats_3d - pca_feats_3d.min()) / (pca_feats_3d.max() - pca_feats_3d.min()) # 0-1 colors

                # write to PC 
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(np.array(hires_coords))
                pc_colors = np.ones_like(np.array(hires_coords)) * 0.5
                pc_colors[has_feats_3d_np] = pca_feats_3d # insert colors only for points that have features
                pc.colors = o3d.utility.Vector3dVector(pc_colors)
                out_path = self.dbg_viz_dir / f'{scene_id}_feats3d_m3dpoints.ply'
                out_path.parent.mkdir(parents=True, exist_ok=True)
                print(f'Saving viz feats3d to: {out_path}')
                o3d.io.write_point_cloud(str(out_path), pc)

        return dino_feats_3d