import os, sys
import subprocess
from tqdm import tqdm

src = "/tmp2/tsunghan/scannetv2_data/scans"

for scene_name in tqdm(os.listdir(src)):
    depth_dir = os.path.join(src, scene_name, "color")
    # write image name list files
    img_name_files = f'{scene_name}/image_name_list.txt'
    with open (img_name_files, "w") as f:
        for fname in os.listdir(depth_dir):
            if ".jpg" in fname:
                print (fname[:-4], file=f)

    # copy intrinsic, pose dir               
    #cmd1 = f'cp -r {intrinsic_dir}/ {dst}/{scene_name}/'
    #os.system(cmd1)
    #cmd2 = f'cp -r {pose_dir}/ {dst}/{scene_name}/'
    #os.system(cmd2)


    


    