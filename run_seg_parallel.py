
import json



from pathlib import Path
import subprocess
import tempfile
import yaml
import open3d as o3d
from scannetpp.common.scene_release import ScannetppScene_Release
import numpy as np
from fire import Fire
from joblib import Parallel, delayed
from loguru import logger
import multiprocessing


def run_with_cfg(executable, cfg):
    with tempfile.NamedTemporaryFile('w') as f:
        yaml.dump(cfg, f)

        cmd = f'{executable} {f.name}'
        # run the executable
        subprocess.run(cmd, shell=True)


def read_txt_list(path):
    with open(path, 'r') as f:
        return f.read().splitlines()

class BasePreprocessing:
    def __init__(self,
                 exec_path='/rhome/cyeshwanth/scannetpp/scan_processing_cpp/build/scan_processing', 
                 data_root='/cluster/eriador/cyeshwanth/scannetpp_download/data/',
                 segmentThresh=0.005,
                 segmentMinVertex=80,
                 list_file='/cluster/eriador/cyeshwanth/caption3d/spp_meta/split_public/train_new_val.txt',
                 out_dir='/cluster/eriador/cyeshwanth/caption3d/mask3d/segments_0.005_80/',
                 viz_segments=False,
                 viz_dir='/cluster/eriador/cyeshwanth/caption3d/mask3d/segments_0.005_640_viz/',
                 use_small_mesh=False
                 ):
        self.scene_ids = read_txt_list(list_file)
        self.n_jobs = 8
        self.out_dir = Path(out_dir)
        self.data_root = Path(data_root)
        self.exec_path = exec_path
        self.viz_segments = viz_segments
        self.viz_dir = Path(viz_dir)
        self.segmentThresh = segmentThresh
        self.segmentMinVertex = segmentMinVertex
        self.use_small_mesh = use_small_mesh

    @logger.catch
    def preprocess(self):
        self.n_jobs = (
            multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
        )

        logger.info(f"Tasks: {len(self.scene_ids)}")

        _ = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.process_file)(scene_id)
            for scene_id in self.scene_ids
        )

    # process one scene id
    def process_file(self, scene_id):
        scene = ScannetppScene_Release(scene_id, data_root=self.data_root)
        if self.use_small_mesh:
            mesh_file = scene.scan_small_mesh_path
        else:
            mesh_file = scene.scan_mesh_path

        self.out_dir.mkdir(exist_ok=True, parents=True)
        if self.viz_segments:
            self.viz_dir.mkdir(exist_ok=True, parents=True)

        output_file = self.out_dir / f'{scene_id}.segs.json'

        scene_cfg = {
            'tasks': ['segment_to_file'],
            'segment_to_file': {
                'segmentMeshFile': str(mesh_file),
                'segmentThresh': self.segmentThresh,
                'segmentMinVertex': self.segmentMinVertex,
                'outputFile': str(output_file)
            }
        }

        run_with_cfg(self.exec_path, scene_cfg)
        
        if self.viz_segments:
            mesh = o3d.io.read_triangle_mesh(str(mesh_file))

            with open(output_file) as f:
                seg_data = json.load(f)

            segments = np.array(seg_data['segIndices'], dtype=np.uint32)

            # map segment IDs to 0-N values
            unique_segs = np.unique(segments)
            n_segments = len(unique_segs)
            new_seg_ids = np.arange(n_segments)
            mapping = dict(zip(unique_segs, new_seg_ids))

            for i in range(len(segments)):
                segments[i] = mapping[segments[i]]

            print(f'Vertices: {len(mesh.vertices)}, segment IDs: {len(segments)}, segments: {n_segments}')
            
            colors = np.random.randint(0, 256, (n_segments, 3))
            vtx_colors = colors[segments]
            # viz seg colors on vertices and save to viz dir
            mesh.vertex_colors = o3d.utility.Vector3dVector(vtx_colors / 255.0)
            # save to viz_dir/<seg file name>.ply
            out_path = self.viz_dir / f'{scene_id}.ply'
            o3d.io.write_triangle_mesh(str(out_path), mesh)
            print('Viz mesh:', out_path)


if __name__ == "__main__":
    Fire(BasePreprocessing)
