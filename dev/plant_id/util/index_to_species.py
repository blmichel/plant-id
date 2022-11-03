import glob
import pdb
import os
import json

OFFSET = 5729
NUM_PLANTS = 9999 - 5729 + 1

root = '/home/team_050/data_2021_mini/2021_train_mini'
paths = sorted(glob.glob(root + '/*Plantae*/*'))

def base_path_name(x):
    return os.path.basename(os.path.dirname(x))

#hierarchy_map = {base_path_name(path).split('_')[-1]:int(base_path_name(path).split('_')[0]) - OFFSET for path in paths}
#index = {i:[paths[i], hierarchy_map[base_path_name(paths[i]).split('_')[-1]]] for i in range(len(paths))}

index_to_species = {int(base_path_name(path).split('_')[0]) - OFFSET : \
    ' '.join(base_path_name(path).split('_')[-2:]) for path in paths}

with open("class_mapping.json", "w") as outfile:
    json.dump(index_to_species, outfile)

#filepath_to_index = {i : [paths[i], int(base_path_name(paths[i]).split('_')[0]) - OFFSET] for i in range(len(paths))}
hierarchy_map = {
            '_'.join(base_path_name(path).split('_')[-2:]) : int(base_path_name(path).split('_')[0]) - OFFSET \
            for path in paths
            }       
index = {
        i : [paths[i], hierarchy_map['_'.join(base_path_name(paths[i]).split('_')[-2:])]] \
            for i in range(len(paths))
        }


#count = 0
#for i in range(4271):
#    if filepath_to_index[(i+1)*50 - 1][1] != i:
#        count += 1
##print('misalignments in filepath_to_index: ', count)

count = 0
for i in range(4271):
    if index[(i+1)*50 - 1][1] != i:
        count += 1
print('misalignments in index: ', count)
#index_to_species = {}
##for i, species_path in enumerate(paths):
#    index_to_species[i] = species_path
