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
# DONE?: inheriting from BaseDataset just gives us a few methods (split_dataset,
# resize_image) that could be useful
class iNatDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform: Callable,
    ) -> None:

        self.root = root 
        self.transform = transform

        assert os.path.isdir(self.root), "Provided directory of images does not exist"

        # get an array of all the paths to plant folders
        paths = sorted(glob.glob(root + '/*Plantae*/*'))
        
        # the class IDs are the first part of the pathname
        # minus the offset (i.e. the folder number of the first plant) 
        class_ids = [int(base_path_name(paths[i]).split('_')[0]) - OFFSET \
            for i in range(len(paths))]
        # each class gets associated with a path and an id
        self.index = {
            i : [paths[i], class_ids[i]] for i in range(len(paths))
            }

        # Load class mapping for classifiers
        mapping_file_path = metadata.MAPPING_FILE_PATH
        self.mapping = load_label_mapping(mapping_file_path)            
        
        #self.id_to_species = {int(base_path_name(path).split('_')[0] ) - OFFSET : ' '.join(base_path_name(path).split('_')[-2:]) for path in paths}
        # TODO: load this as a torch classmapping dict
        # DONE?
                
        for idx in range(PLANT_IDX_RANGE[0], PLANT_IDX_RANGE[1] + 1):
            assert idx-OFFSET in self.index, "Index %d not present in root" %idx

    def __len__(self) -> None:
        return len(self.index)

    def __getitem__(self, idx: int):
        """
        Input: idx (int): Index; loader handles offset

        Returns: tuple: (image, target)
        """
        # TODO: what's the advantage of doing this here instead
        # of in the stem?
        fname, target = self.index[idx]
        img = Image.open(fname)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

if __name__ == "__main__":
    load_and_print_info(iNatDataset)
