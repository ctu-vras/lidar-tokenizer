import numpy as np
import os
import re
import datetime
from bisect import bisect_left
from operator import itemgetter

def extract_timestamp(text):
    # extracts string between second underscore and first dot
    # AI generated function
    match = re.search(r'^[^_]*_[^_]*_([^.]*)\.', text)
    if match:
        return match.group(1)
    return 0

def match_timestamps(pcd_timestamps, img_dir):
    img_fns = sorted(os.listdir(img_dir))
    img_fns = [f for f in img_fns if ".jpg" in f]
    img_timestamps = [int(extract_timestamp(f))/10e8 for f in img_fns]
    indices, img_timestamps_sorted = zip(*sorted(enumerate(img_timestamps), key=itemgetter(1)))

    match_ids = [bisect_left(img_timestamps_sorted, ts) for ts in pcd_timestamps]
    match_ids = [i if i<len(img_fns) else i-1 for i in match_ids]
    match_ids_original_list = [indices[i] for i in match_ids]

    match_time_diff = np.abs(np.array(img_timestamps_sorted)[match_ids] - np.array(pcd_timestamps))

    return [os.path.join(img_dir,img_fns[id]) for id in match_ids_original_list], match_time_diff

if __name__ == "__main__":
    img_dir = 'data/camera1'
    pcd_fn = 'data/npz/time_1725621648.5456.npz'
    pcd = np.load(pcd_fn)
    ts = float(pcd['timestamp'])
    match_timestamps([ts,ts,ts], img_dir)