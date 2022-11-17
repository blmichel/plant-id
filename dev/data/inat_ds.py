"""Dataset class for iNaturalist datasets."""
import os
import glob
import json
from typing import Callable

from PIL import Image

import dev.metadata.inat as metadata
from dev.data.util import BaseDataset, load_and_print_info
from dev.util.util import base_path_name

PLANT_IDX_RANGE = metadata.PLANT_IDX_RANGE
OFFSET = metadata.OFFSET

### dataset class for iNat
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
        with open(metadata.MAPPING_FILE_PATH, "r") as f:
            self.mapping = json.load(f)
                
        for idx in range(PLANT_IDX_RANGE[0], PLANT_IDX_RANGE[1] + 1):
            assert idx-OFFSET in self.index, "Index %d not present in root" %idx

    def __len__(self) -> None:
        return len(self.index)

    def __getitem__(self, idx: int):
        """
        Input: idx (int): Index; loader handles offset

        Returns: tuple: (image, target)
        """
        # xTODO: what's the advantage of doing this here instead
        # of in the stem?
        # ANSWER: images are differently sized, need to resize before
        # running torch.collate. could preprocess the entire dataset,
        # not sure how much of a difference this would make at runtime
        # TODO: determine whether data transform is a bottleneck
        fname, target = self.index[idx]
        img = Image.open(fname)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

if __name__ == "__main__":
    load_and_print_info(iNatDataset)
