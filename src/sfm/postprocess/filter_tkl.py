import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from src.utils.colmap.read_write_model import write_model


def get_points_count(points3D, show=False):
    """ Count track length for each point """
    points_count_list = [] # track length for each point
    for points_id, points_item in points3D.items():
        points_count = len(points_item.point2D_idxs)
        points_count_list.append(points_count)
    
    count_dict = dict() # {track length: num of 3d points}
    for count in points_count_list:
        if count not in count_dict.keys():
            count_dict[count] = 0
        count_dict[count] += 1
    counts = list(count_dict.keys())
    counts.sort()
    
    count_list_ordered = []
    for count in counts:
        count_list_ordered.append(count_dict[count])
    
    if show:
        plt.plot(counts, count_list_ordered)
        plt.show()

    return count_dict, points_count_list


def get_tkl(model_path, thres, show=False):
    """ Get the track length value which can limit the number of 3d points below thres"""
    from src.utils.colmap.read_write_model import read_model

    cameras, images, points3D = read_model(model_path, ext='.bin')
    count_dict, points_count_list = get_points_count(points3D, show)
    
    ret_points = len(points3D.keys())
    count_keys = np.array(list(count_dict.keys())) 
    count_keys.sort()
    
    for key in count_keys:
        ret_points -= count_dict[key]
        if ret_points <= thres:
            track_length = key
            break

    return track_length, points_count_list


def vis_tkl_filtered_pcds(model_path, points_count_list, track_length, output_path):
    """ 
    Given a track length value, filter 3d points.
    Output filtered pcds for visualization.
    """ 
    from src.utils.colmap.read_write_model import read_model

    cameras, images, points3D = read_model(model_path, ext='.bin')
    
    invalid_points = np.where(np.array(points_count_list) < track_length)[0]
    point_ids = []
    for k, v in points3D.items():
        point_ids.append(k)
    
    invalid_points_ids = []
    for invalid_count in invalid_points:
        points3D.pop(point_ids[invalid_count])
        invalid_points_ids.append(point_ids[invalid_count])
    
    output_path = osp.join(output_path, 'tkl_model')
    output_file_path = osp.join(output_path, 'tl-{}.ply'.format(track_length))
    if not osp.exists(output_path):
        os.makedirs(output_path)
    
    write_model(cameras, images, points3D, output_path, '.bin')
    os.system(f'colmap model_converter --input_path {output_path} --output_path {output_file_path} --output_type PLY')
    return output_file_path