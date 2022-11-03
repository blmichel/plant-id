"""iNat Dataset Class."""

import argparse
import os
import glob
from pathlib import Path
from PIL import Image

from plant_id.data.base_data_module import BaseDataModule, load_and_print_info
import plant_id.metadata.inat as metadata
from plant_id.stems.image import iNatStem
from plant_id.data.util import BaseDataset
from plant_id.util.util import base_path_name

from torch.utils.util import load_label_mapping

from typing import Callable

PLANT_IDX_RANGE = metadata.PLANT_IDX_RANGE
OFFSET = metadata.OFFSET

### dataset class for iNat
# TODO: maybe this should inherit from BaseDataset. need refactor for compatibility?
# or should we just use what we've got already and add any features that we want?
class iNatDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable,
    ) -> None:

        self.root = root 
        self.transform = transform

        assert os.path.isdir(self.root), "Provided directory of images does not exist"

        paths = sorted(glob.glob(root + '/*Plantae*/*'))
        
        # can combine this
        class_ids = [int(base_path_name(paths[i]).split('_')[0]) - OFFSET \
            for i in range(len(paths))]
        self.index = {
            i : [paths[i], class_ids[i]] for i in range(len(paths))
            }

        # Load class mapping for classifiers
        mapping_file_path = metadata.MAPPING_FILE_PATH
        self.mapping = load_label_mapping(mapping_file_path)            
        
        #self.id_to_species = {int(base_path_name(path).split('_')[0] ) - OFFSET : ' '.join(base_path_name(path).split('_')[-2:]) for path in paths}
        # TODO: load this as a torch classmapping dict
                
        for idx in range(PLANT_IDX_RANGE[0], PLANT_IDX_RANGE[1] + 1):
            assert idx-OFFSET in self.index, "Index %d not present in root" %idx

    def __len__(self) -> None:
        return len(self.index)

    def __getitem__(self, idx: int):
        """
        Input: idx (int): Index; loader handles offset

        Returns: tuple: (image, target)
        """
        fname, target = self.index[idx]
        img = Image.open(fname)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class INAT_MINI(BaseDataModule):
    """iNat-mini DataModule."""

    def __init__(self, split: str, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.transform = iNatStem()
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        self.augment = self.args.get("augment_data", "true").lower() == "true"
        self.root = self.args.get("root", "/")
        
        self.setup('train')
        self.setup('val')
#        self.data_test = self.setup('test')

    def setup(self, split: str):
        if split == 'train':
            data_dir = metadata.MINI_DATA_DIRNAME
        elif split == 'val':
            data_dir = metadata.VAL_DATA_DIRNAME
        elif split == 'test':
            data_dir = metadata.TEST_DATA_DIRNAME
        else:
            ValueError("Split must be 'train', 'val' or 'test'")

        assert os.path.isdir(data_dir), f"Provided directory of images {data_dir} does not exist"

        paths = sorted(glob.glob(data_dir + '/*Plantae*/*'))
        self.hierarchy_map = {self.base_path_name(path).split('_')[-1]:int(self.base_path_name(path).split('_')[0]) - metadata.OFFSET for path in paths}
    
        indices = {
            i:[paths[i], self.hierarchy_map[self.base_path_name(paths[i]).split('_')[-1]]] for i in range(len(paths))
            }
        for idx in range(metadata.PLANT_IDX_RANGE[0], metadata.PLANT_IDX_RANGE[1] + 1):
            assert idx - metadata.OFFSET in self.index, "Index %d not present in root" %idx
        
        if split == 'train':
            self.data_train = indices
        elif split == 'val':
            self.data_val = indices
        elif split == 'test':
            pass
        else:
            ValueError("Split must be 'train', 'val' or 'test'")    
            
    def __len__(self) -> None:
        return len(self.train_index) + len(self.val_index)


    def __getitem__(self, idx: int):
        """
        Inputs: - idx (int): Index; loader handles offset
                - split: get item from train/val/test

        Returns: tuple: (image, target)
        """

        fname, target = self.index[idx]
        img = self.transform(Image.open(fname))

        return img, target
       
    @staticmethod
    def base_path_name(x):
        return os.path.basename(os.path.dirname(x))
    
    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

if __name__ == "__main__":
    load_and_print_info(INAT_MINI)
