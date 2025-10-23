from pathlib import Path
import numpy as np
from fire import Fire
from natsort import natsorted
from loguru import logger
import torch

from datasets.preprocessing.base_preprocessing import BasePreprocessing


def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()

    return lines

class ScannetppPreprocessing(BasePreprocessing):
    '''
    Create Scannetpp dataset for mask3d from PTH files
    '''
    def __init__(
            self,
            data_dir: str,
            save_dir: str,
            train_list: str,
            val_list: str,
            labels_path: str = '/cluster/eriador/cyeshwanth/caption3d/spp_meta/top100.txt',
            instance_labels_path: str = "/cluster/eriador/cyeshwanth/caption3d/spp_meta/top17_instance_from_100_v2.txt",
            modes: tuple = ("train", "validation"),
            n_jobs: int = -1,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.lists = {
            'train': read_txt_list(train_list),
            'validation': read_txt_list(val_list),
        }
        
        self.labels = read_txt_list(labels_path)
        self.instance_labels = read_txt_list(instance_labels_path)
        self.palette = np.random.randint(0, 256, (200, 3))
        
        self.create_label_database()

        for mode in self.modes:
            filepaths = []
            for scene_id in self.lists[mode]:
                # path to pth file
                path = Path(data_dir) / f'{scene_id}.pth'
                if path.is_file():
                    filepaths.append(path)
            self.files[mode] = natsorted(filepaths)
            print(f'Found {len(self.files[mode])} files for {mode}')

    def create_label_database(self):
        labeldb = {}

        for label_ndx, label in enumerate(self.labels):
            validation = True if label in self.instance_labels else False

            labeldb[label_ndx] = {
                'color': self.palette[label_ndx].tolist(),
                'name': label,
                'validation': validation
            }
            
        self._save_yaml(self.save_dir / "label_database.yaml", labeldb)
        return labeldb

    def process_file(self, filepath, mode):
        """process_file.
        Read prepare data from pth files and create npy files, label database, train and val database, color mean files

        Args:
            filepath: path to the pth file file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        scene = filepath.stem
        filebase = {
            "filepath": filepath,
            "scene": scene,
            "sub_scene": None,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        
        pth_data = torch.load(filepath)
        # read everything from pth file
        coords = pth_data['vtx_coords']
        colors = pth_data['vtx_colors']
        normals = pth_data['vtx_normals']
        segment_ids = pth_data['vtx_segment_ids']
        semantic_labels = pth_data['vtx_labels'].astype(np.float32)
        # keep vtx_instance_anno_id so that it can be used to match with caption data, etc
        # not instance labels
        instance_labels = pth_data['vtx_instance_anno_id'].astype(np.float32)
        
        file_len = len(coords)
        # add dummy information
        filebase["file_len"] = file_len
        filebase["scene_type"] = 'dummy'
        filebase["raw_description_filepath"] = 'dummy'
        filebase["raw_instance_filepath"] = 'dummy'
        filebase["raw_segmentation_filepath"] = 'dummy'
        
        # make segment IDs 0..N
        unique_segment_ids = np.unique(segment_ids, return_inverse=True)[1].astype(np.float32)
        # put everything in a single array
        # add an extra axis to segment IDs, sem labels and instance labels 
        points = np.hstack((coords, colors.astype(np.float32) * 255, normals, unique_segment_ids[..., None], semantic_labels[..., None], instance_labels[..., None]))
        # use np.int32 to avoid overflow?
        gt_data = points[:, -2] * 1000 + points[:, -1] + 1

        processed_filepath = self.save_dir / mode / f"{scene}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = self.save_dir / "instance_gt" / mode / f"{scene}.txt"
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        filebase["color_mean"] = [
            float((colors[:, 0] / 255).mean()),
            float((colors[:, 1] / 255).mean()),
            float((colors[:, 2] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((colors[:, 0] / 255) ** 2).mean()),
            float(((colors[:, 1] / 255) ** 2).mean()),
            float(((colors[:, 2] / 255) ** 2).mean()),
        ]
        return filebase

    def compute_color_mean_std(
            self, train_database_path: str = "./data/processed/scannet/train_database.yaml"
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean ** 2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

    @logger.catch
    def fix_bugs_in_labels(self):
        pass


if __name__ == "__main__":
    Fire(ScannetppPreprocessing)
