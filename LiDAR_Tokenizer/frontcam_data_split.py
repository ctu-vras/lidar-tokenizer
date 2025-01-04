import numpy as np
import os
import json

data_root = "/data"
train_val_dirs = ["npz_2024-09-06-07-42-41","npz_2024-09-06-10-59-33","npz_2024-09-06-11-19-38","npz_2024-09-13-12-19-26",
                  "npz_2024-09-09-12-33-35","npz_2024-09-13-06-43-51","npz_2024-09-13-07-06-04","npz_2024-09-13-10-31-13"] # no. samples 13000 13000
test_dirs = ["npz_2024-09-06-11-42-34", "npz_2024-09-17-06-52-07"]   # no. samples 9000

assert len(set(train_val_dirs+test_dirs)) == len(train_val_dirs) + len(test_dirs)

train_val_files = []
test_files = []

for dir in train_val_dirs:
    dir = os.path.join(data_root,dir)
    train_val_files += [os.path.join(dir.split('/')[-1],f) for f in sorted(os.listdir(dir)) if os.path.isfile(os.path.join(dir,f))]

for dir in test_dirs:
    dir = os.path.join(data_root,dir)
    test_files += [os.path.join(dir.split('/')[-1],f) for f in sorted(os.listdir(dir)) if os.path.isfile(os.path.join(dir,f))]

#split = {'train': int(round(len(files)*0.7)), 'val': int(round(len(files)*0.1))}
split = {'train': int(round(len(train_val_files)*0.8)), 'val': int(round(len(train_val_files)*0.2))}

train_files = train_val_files[0:split['train']]
val_files   = train_val_files[split['train']:]
#test_files  = files[split['train']+split['val']:]

assert len(train_val_files) == len(train_files) + len(val_files)    # all used
assert len(train_val_files) == len(list(set(train_files+val_files)))          # no repeats

# DECIMATE
#train_files = train_files[0::5]
#val_files = val_files[0::5]
#test_files = test_files[0::5]

json_all_files = dict(train=train_files, val=val_files, test=test_files)
json_complete = {"v1.0-trainval": json_all_files}
json_object = json.dumps(json_complete, indent=4)

with open("LiDAR_Tokenizer/data_managment/frontcam.json", "w") as outfile:
    outfile.write(json_object)

print(f"train: {len(train_files)/(len(train_files)+len(val_files)+len(test_files))}\n val: {len(val_files)/(len(train_files)+len(val_files)+len(test_files))}\n test: {len(test_files)/(len(train_files)+len(val_files)+len(test_files))}")