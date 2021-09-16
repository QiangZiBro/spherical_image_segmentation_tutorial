import sys
sys.path.append("..")
import numpy as np
from glob import glob
import os
import random
import io

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from gas_key import KEY
from tensorbay import GAS
from tensorbay.dataset import Segment

gas = GAS(KEY)
dataset_client = gas.get_dataset("spherical_segmentation")
segments = dataset_client.list_segment_names()
file_format = Segment("2d3ds_sphere", dataset_client)

nv_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
precomp_mean = [0.4974898, 0.47918808, 0.42809588, 1.0961773]
precomp_std = [0.23762763, 0.23354423, 0.23272438, 0.75536704]

def getter(a, seg):
    # 获取对应区域a的所有数据
    result = []
    for i in seg:
        if f"area_{a}" in i.path:
            result.append(i)
    return sorted(result, key=lambda x:x.path)

def read_gas(data):
    with data.open() as f:
        return np.load(io.BytesIO(f.read()), allow_pickle=True)
    
class S2D3DSegLoader(Dataset):
    """Data loader for 2D3DS dataset."""

    def __init__(self, partition, fold, sp_level, in_ch=4, normalize_mean=precomp_mean, normalize_std=precomp_std):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
            fold: 1, 2 or 3 (for 3-fold cross-validation)
            sp_level: sphere mesh level. integer between 0 and 7.
            
        """
        assert(partition in ["train", "test"])
        assert(fold in [1, 2, 3])
        self.in_ch = in_ch
        self.nv = nv_sphere[sp_level]
        self.partition = partition
        if fold == 1:
            train_areas = ['1', '2', '3', '4', '6']
            test_areas = ['5a', '5b']
        elif fold == 2:
            train_areas = ['1', '3', '5a', '5b', '6']
            test_areas = ['2', '4']
        elif fold == 3:
            train_areas = ['2', '4', '5a', '5b']
            test_areas = ['1', '3', '6']

        if partition == "train":
            self.areas = train_areas
        else:
            self.areas = test_areas

        self.flist = []
        for a in self.areas:
            self.flist += getter(a, file_format)

        self.mean = np.expand_dims(precomp_mean, -1).astype(np.float32)
        self.std = np.expand_dims(precomp_std, -1).astype(np.float32)

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        fname = self.flist[idx]
        data = (read_gas(fname)["data"].T[:self.in_ch, :self.nv] - self.mean) / self.std
        labels = read_gas(fname)["labels"].T[:self.nv].astype(np.int)
        return data, labels
