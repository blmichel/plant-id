"""Metadata for iNat."""
import yaml

## load config
config_file = "training_config.yml"
with open(config_file, "rb") as file:
        config = yaml.load(file, Loader=yaml.Loader)
        
## determine data and mapping file locations
BASE_DATA_DIRNAME = config['BASE_DATA_DIRNAME']
MAPPING_FILE_DIRNAME = config['MAPPING_FILE_DIRNAME']

MINI_DATA_DIRNAME = BASE_DATA_DIRNAME + 'train_mini/'
TRAIN_DATA_DIRNAME = BASE_DATA_DIRNAME + 'train/'
VAL_DATA_DIRNAME = BASE_DATA_DIRNAME + 'val/'
TEST_DATA_DIRNAME = BASE_DATA_DIRNAME + 'test/'

MAPPING_FILE_NAME = 'class_mapping.json'
MAPPING_FILE_PATH = MAPPING_FILE_DIRNAME + MAPPING_FILE_NAME

## basic metadata for iNatDataset
PLANT_IDX_RANGE = [5729, 9999]
OFFSET = PLANT_IDX_RANGE[0]
NUM_PLANT_CLASSES = PLANT_IDX_RANGE[1] - PLANT_IDX_RANGE[0] + 1

MINI_TRAIN_SIZE = 500_000
MID_TRAIN_SIZE = 2_686_843
VAL_SIZE = 100_000
TEST_SIZE = 500_000
