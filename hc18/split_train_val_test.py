import os 
import random
import shutil
import sys

if (len(os.listdir("data/hc18/valid_set")) != 0 or len(os.listdir("data/hc18/test_set")) != 0): 
    print("Your valid set is not empty")
    sys.exit(1)
    

data_path = "data/hc18/training_set"

path_list = os.listdir(data_path)
path_list = [os.path.join(data_path, path) for path in path_list]

img_path_list = [path for path in path_list if not "Annotation" in path]
annot_path_list = [path for path in path_list if "Annotation" in path]

img_path_list.sort()
annot_path_list.sort()

pair_list = [(img, annot) for (img, annot) in zip(img_path_list, annot_path_list)]

def split(file_list, ratio): 
    return random.sample(file_list, int(len(file_list)*ratio))

def move_files(file_list, target): 
    os.makedirs(target, exist_ok=True)
    for pair in file_list: 
        img = pair[0]
        annot = pair[1]
        shutil.move(img, target)
        shutil.move(annot, target)

valid_set = split(pair_list, ratio=0.1)
move_files(valid_set, "data/hc18/valid_set")

train_set = [pair for pair in pair_list if pair not in valid_set]

test_set = split(train_set, ratio=0.1)
move_files(test_set, "data/hc18/test_set")
