import numpy as np
import os
import json

def main(data_root, path_out_range="Dataset/frontcam_tokens.json", path_out_voxel="Dataset/frontcam_tokens_voxel.json", val=False):
    train_val_dirs = [("img_tokens_2024-09-06-07-42-41",
                    "lidar_tokens_2024-09-06-07-42-41",
                    "lidar_tokens_voxel_2024-09-06-07-42-41",
                    "img_paths_2024-09-06-07-42-41.json",
                    "lidar_paths_2024-09-06-07-42-41.json",
                    "lidar_paths_voxel_2024-09-06-07-42-41.json",
                    ),

                        ("img_tokens_2024-09-06-10-59-33",
                        "lidar_tokens_2024-09-06-10-59-33",
                        "lidar_tokens_voxel_2024-09-06-10-59-33",
                        "img_paths_2024-09-06-10-59-33.json",
                        "lidar_paths_2024-09-06-10-59-33.json",
                        "lidar_paths_voxel_2024-09-06-10-59-33.json",
                        ),

                        ("img_tokens_2024-09-06-11-19-38",
                        "lidar_tokens_2024-09-06-11-19-38",
                        "lidar_tokens_voxel_2024-09-06-11-19-38",
                        "img_paths_2024-09-06-11-19-38.json",
                        "lidar_paths_2024-09-06-11-19-38.json",
                        "lidar_paths_voxel_2024-09-06-11-19-38.json",
                        ),
                        
                        ("img_tokens_2024-09-09-12-33-35",
                        "lidar_tokens_2024-09-09-12-33-35",
                        "lidar_tokens_voxel_2024-09-09-12-33-35",
                        "img_paths_2024-09-09-12-33-35.json",
                        "lidar_paths_2024-09-09-12-33-35.json",
                        "lidar_paths_voxel_2024-09-09-12-33-35.json",
                        ),

                        ("img_tokens_2024-09-13-06-43-51",
                        "lidar_tokens_2024-09-13-06-43-51",
                        "lidar_tokens_voxel_2024-09-13-06-43-51",
                        "img_paths_2024-09-13-06-43-51.json",
                        "lidar_paths_2024-09-13-06-43-51.json",
                        "lidar_paths_voxel_2024-09-13-06-43-51.json",
                        ),

                        ("img_tokens_2024-09-13-07-06-04",
                        "lidar_tokens_2024-09-13-07-06-04",
                        "lidar_tokens_voxel_2024-09-13-07-06-04",
                        "img_paths_2024-09-13-07-06-04.json",
                        "lidar_paths_2024-09-13-07-06-04.json",
                        "lidar_paths_voxel_2024-09-13-07-06-04.json",
                        ),

                        ("img_tokens_2024-09-13-10-31-13",
                        "lidar_tokens_2024-09-13-10-31-13",
                        "lidar_tokens_voxel_2024-09-13-10-31-13",
                        "img_paths_2024-09-13-10-31-13.json",
                        "lidar_paths_2024-09-13-10-31-13.json",
                        "lidar_paths_voxel_2024-09-13-10-31-13.json",
                        ),

                        ("img_tokens_2024-09-13-12-19-26",
                        "lidar_tokens_2024-09-13-12-19-26",
                        "lidar_tokens_voxel_2024-09-13-12-19-26",
                        "img_paths_2024-09-13-12-19-26.json",
                        "lidar_paths_2024-09-13-12-19-26.json",
                        "lidar_paths_voxel_2024-09-13-12-19-26.json",
                        )]
                        

    test_dirs = [

                ("img_tokens_2024-09-06-11-42-34",
                "lidar_tokens_2024-09-06-11-42-34",
                "lidar_tokens_voxel_2024-09-06-11-42-34",
                "img_paths_2024-09-06-11-42-34.json",
                "lidar_paths_2024-09-06-11-42-34.json",
                "lidar_paths_voxel_2024-09-06-11-42-34.json",
                ),

                ("img_tokens_2024-09-17-06-52-07",
                "lidar_tokens_2024-09-17-06-52-07",
                "lidar_tokens_voxel_2024-09-17-06-52-07",
                "img_paths_2024-09-17-06-52-07.json",
                "lidar_paths_2024-09-17-06-52-07.json",
                "lidar_paths_voxel_2024-09-17-06-52-07.json",
                )

    ]

    train_val_files = []
    train_val_files_voxel = []
    test_files = []
    test_files_voxel = []

    for img_tok_dir,lidar_tok_dir,lidar_voxel_tok_dir,img_json,lidar_json,lidar_json_voxel in train_val_dirs:
        img_tok_dir = os.path.join(data_root,img_tok_dir)
        lidar_tok_dir = os.path.join(data_root,lidar_tok_dir)
        lidar_voxel_tok_dir = os.path.join(data_root,lidar_voxel_tok_dir)
        img_json = os.path.join(data_root,img_json)
        lidar_json = os.path.join(data_root,lidar_json)
        lidar_json_voxel = os.path.join(data_root,lidar_json_voxel)

        img_tok_files = [os.path.join(img_tok_dir.split('/')[-1],f) for f in sorted(os.listdir(img_tok_dir)) if os.path.isfile(os.path.join(img_tok_dir,f))]
        lidar_tok_files = [os.path.join(lidar_tok_dir.split('/')[-1],f) for f in sorted(os.listdir(lidar_tok_dir)) if os.path.isfile(os.path.join(lidar_tok_dir,f))]
        lidar_voxel_tok_files = [os.path.join(lidar_voxel_tok_dir.split('/')[-1],f) for f in sorted(os.listdir(lidar_voxel_tok_dir)) if os.path.isfile(os.path.join(lidar_voxel_tok_dir,f))]
        
        img_paths_dict = dict()
        with open(img_json, "r") as f:
            img_paths_dict = json.load(f)
        img_files = [img_paths_dict[img_tok_file].replace(data_root+"/","") for img_tok_file in img_tok_files]
        
        lidar_paths_dict = dict()
        with open(lidar_json, "r") as f:
            lidar_paths_dict = json.load(f)
        lidar_files = [lidar_paths_dict[lidar_tok_file].replace(data_root+"/","") for lidar_tok_file in lidar_tok_files]

        lidar_paths_dict_voxel = dict()
        with open(lidar_json_voxel, "r") as f:
            lidar_paths_dict_voxel = json.load(f)
        lidar_voxel_files = [lidar_paths_dict_voxel[lidar_tok_file].replace(data_root+"/","") for lidar_tok_file in lidar_voxel_tok_files]

        for i in range(len(img_files)):
            train_val_files.append((img_tok_files[i],lidar_tok_files[i],img_files[i],lidar_files[i]))
            train_val_files_voxel.append((img_tok_files[i],lidar_voxel_tok_files[i],img_files[i],lidar_voxel_files[i]))


    for img_tok_dir,lidar_tok_dir,lidar_voxel_tok_dir,img_json,lidar_json,lidar_json_voxel in test_dirs:
        img_tok_dir = os.path.join(data_root,img_tok_dir)
        lidar_tok_dir = os.path.join(data_root,lidar_tok_dir)
        lidar_voxel_tok_dir = os.path.join(data_root,lidar_voxel_tok_dir)
        img_json = os.path.join(data_root,img_json)
        lidar_json = os.path.join(data_root,lidar_json)
        lidar_json_voxel = os.path.join(data_root,lidar_json_voxel)

        img_tok_files = [os.path.join(img_tok_dir.split('/')[-1],f) for f in sorted(os.listdir(img_tok_dir)) if os.path.isfile(os.path.join(img_tok_dir,f))]
        lidar_tok_files = [os.path.join(lidar_tok_dir.split('/')[-1],f) for f in sorted(os.listdir(lidar_tok_dir)) if os.path.isfile(os.path.join(lidar_tok_dir,f))]
        lidar_voxel_tok_files = [os.path.join(lidar_voxel_tok_dir.split('/')[-1],f) for f in sorted(os.listdir(lidar_voxel_tok_dir)) if os.path.isfile(os.path.join(lidar_voxel_tok_dir,f))]

        img_paths_dict = dict()
        with open(img_json, "r") as f:
            img_paths_dict = json.load(f)
        img_files = [img_paths_dict[img_tok_file].replace(data_root+"/","") for img_tok_file in img_tok_files]

        lidar_paths_dict = dict()
        with open(lidar_json, "r") as f:
            lidar_paths_dict = json.load(f)
        lidar_files = [lidar_paths_dict[lidar_tok_file].replace(data_root+"/","") for lidar_tok_file in lidar_tok_files]

        lidar_paths_dict_voxel = dict()
        with open(lidar_json_voxel, "r") as f:
            lidar_paths_dict_voxel = json.load(f)
        lidar_voxel_files = [lidar_paths_dict_voxel[lidar_tok_file].replace(data_root+"/","") for lidar_tok_file in lidar_voxel_tok_files]

        for i in range(len(img_files)):
            test_files.append((img_tok_files[i],lidar_tok_files[i],img_files[i],lidar_files[i]))
            test_files_voxel.append((img_tok_files[i],lidar_voxel_files[i],img_files[i],lidar_files[i]))

    if val:
        split = {'train': int(round(len(train_val_files)*0.8)), 'val': int(round(len(train_val_files)*0.2))}
    else:
        split = {'train': int(round(len(train_val_files)*1.0)), 'val': int(round(len(train_val_files)*0.0))}

    train_files = train_val_files[0:split['train']]
    val_files   = train_val_files[split['train']:]
    train_files_voxel = train_val_files_voxel[0:split['train']]
    val_files_voxel   = train_val_files_voxel[split['train']:]

    assert len(train_val_files) == len(train_files) + len(val_files)    # all used
    assert len(train_val_files) == len(list(set(train_files+val_files)))          # no repeats
    assert len(train_val_files_voxel) == len(train_files_voxel) + len(val_files_voxel)    # all used
    assert len(train_val_files_voxel) == len(list(set(train_files_voxel+val_files)))          # no repeats

    # DECIMATE
    #train_files = train_files[0::5]
    #val_files = val_files[0::5]
    #test_files = test_files[0::5]
    #train_files_voxel = train_files_voxel[0::5]
    #val_files_voxel = val_files_voxel[0::5]
    #test_files_voxel = test_files_voxel[0::5]

    json_all_files = dict(train=train_files, val=val_files, test=test_files)
    json_complete = {"v1.0-trainval": json_all_files}
    json_object = json.dumps(json_complete, indent=4)

    with open(path_out_range, "w") as outfile:
        outfile.write(json_object)

    json_all_files_voxel = dict(train=train_files_voxel, val=val_files_voxel, test=test_files_voxel)
    json_complete_voxel = {"v1.0-trainval": json_all_files_voxel}
    json_object_voxel = json.dumps(json_complete_voxel, indent=4)

    with open(path_out_voxel, "w") as outfile:
        outfile.write(json_object_voxel)

    print(f"train: {len(train_files)}\n val: {len(val_files)}\n test: {len(test_files)}")
    print(f"train: {len(train_files)/(len(train_files)+len(val_files)+len(test_files))}\n val: {len(val_files)/(len(train_files)+len(val_files)+len(test_files))}\n test: {len(test_files)/(len(train_files)+len(val_files)+len(test_files))}")

if __name__ == "__main__":
    main(data_root="/data",
         path_out_range="Dataset/frontcam_tokens.json",
         path_out_voxel="Dataset/frontcam_tokens_voxel.json",
         val=False)