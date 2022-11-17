"""Stem classes for preprocessing images."""
## these classes are not currently used, see comments in data/inat_ds.py under __getitem__ 

import torch
from torchvision import transforms
import yaml

## load config
config_file = "training_config.yml"
with open(config_file, "rb") as file:
        config = yaml.load(file, Loader=yaml.Loader)

resolution = config['RESOLUTION']

class ImageStem:
    """A stem for models operating on images.

    Images are presumed to be provided as JPG images,
    which are converted to PIL via PIL.Image.open().

    Transforms are split into two categories:
    pil_transforms, which take in and return PIL images, and
    torch_transforms, which take in and return Torch tensors.

    By default, these two transforms are both identities.
    In between, the images are mapped to tensors.

    The torch_transforms are wrapped in a torch.nn.Sequential
    and so are compatible with torchscript if the underyling
    Modules are compatible.
    """

    def __init__(self):
        self.pil_transforms = transforms.Compose([])
        self.pil_to_tensor = transforms.ToTensor()
        self.torch_transforms = torch.nn.Sequential()

    def __call__(self, img):
        img = self.pil_transforms(img)
        img = self.pil_to_tensor(img)

        with torch.no_grad():
            img = self.torch_transforms(img)

        return img
    
    
class iNatStem(ImageStem):
    """A stem for handling images from the iNat datasets."""

    # TODO: what's the appropriate normalization for imagenet-pretrained models?
    def __init__(self):
        super().__init__()
        self.torch_transforms = torch.nn.Sequential([transforms.Normalize((0.1307,), (0.3081,)),
                                                     transforms.Resize((resolution, resolution)),
                                                     transforms.ToTensor()])
                                                    