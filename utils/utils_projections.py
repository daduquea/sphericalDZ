import numpy as np
from projection import Projection
from utils_challenge import *


def compute_local_projection(proj, some_points, feature_to_project, aggregate_func, global_parameters):
    height_glob, width_glob, min_val_glob = global_parameters
    bev_proj = proj.project_points_values(some_points, feature_to_project, aggregate_func = aggregate_func)
    height_local, width_local = bev_proj.shape
    min_val_local = some_points.min(0)    
    return bev_proj, [height_local, width_local, min_val_local]

def move_local_proj_to_global(proj, bev_proj, global_parameters, local_parameters, mapping = None):
    height_glob, width_glob, min_val_glob = global_parameters
    height_local, width_local, min_val_local = local_parameters
    diffs = min_val_local - min_val_glob
    disp_h = np.floor(diffs[0] * proj.projector.res_x).astype(int)
    disp_w = np.floor(diffs[1] * proj.projector.res_y).astype(int)
        
    # fill image
    #print("global_parameters", global_parameters)
    #print("local_parameters", local_parameters)
    placed_img = np.zeros([height_glob, width_glob])
    placed_img [disp_h:disp_h + height_local, disp_w:disp_w+width_local] = bev_proj 
    
    if mapping:
        return placed_img, [disp_h, disp_w]
    
    else:
        return placed_img

def compute_all_projections(proj, full_xyz_points, full_labels, global_parameters, z_min = -100, z_max = 100,  acc_clip = 255):
    # some points
    idx_some_points = np.where( (full_xyz_points[:,2] < z_max) & (full_xyz_points[:,2] > z_min) )[0]
    some_points = full_xyz_points[idx_some_points, :]
    some_labels = full_labels[idx_some_points]    
    
    #features
    # h_max
    feature_to_proj = some_points[:,2]
    aggregate_func = 'max'
    bev_hmax, local_parameters = compute_local_projection(proj, some_points, feature_to_proj , aggregate_func, global_parameters)
    bev_hmax_centered = move_local_proj_to_global(proj, bev_hmax, global_parameters, local_parameters)    
    
    # accum
    feature_to_proj = np.ones_like(some_points[:,2])
    aggregate_func = 'sum'
    bev_acc, local_parameters = compute_local_projection(proj, some_points, feature_to_proj , aggregate_func, global_parameters)
    bev_acc_centered = move_local_proj_to_global(proj, bev_acc, global_parameters, local_parameters)
    bev_acc_clipped = np.clip(bev_acc_centered, 0, acc_clip)    
    
    # GT
    gt_projection, local_parameters, mapping = compute_gt_image(proj, some_points, some_labels, 'max')
    gt_projection_centered, move = move_local_proj_to_global(proj, gt_projection, global_parameters, local_parameters, mapping)
    return bev_hmax_centered, bev_acc_clipped, bev_acc_centered, gt_projection_centered, mapping, move

# Method to find "closest" index in spherical projection
def min_aggregation_proj(height, width, proj, points):
    depth = np.linalg.norm(points, axis = 1)
    lidx, i_img_mapping, j_img_mapping = proj.projector.project_point(points)
    aux_depth_proj = np.full([height, width], np.inf)
    aux_idx_proj = np.full([height, width], -1)
    for k in range(len(lidx)):
        h_coor = i_img_mapping[k]
        w_coor = j_img_mapping[k]    
        #print(h_coor)
        #print(w_coor)
        #print("*")
        if depth[k] < aux_depth_proj[h_coor, w_coor]:
            aux_idx_proj[h_coor, w_coor] = k
            aux_depth_proj[h_coor, w_coor] = depth[k]
    return aux_idx_proj, [lidx, i_img_mapping, j_img_mapping]


def max_aggregation_proj(proj, points, labels, aggreg):
    h, w = proj.projector.get_image_size(points=points)
    lidx, i_img_mapping, j_img_mapping = proj.projector.project_point(points)
    max_projection = np.full([h, w], -1)
    aux_height_proj = np.full([h, w], -np.inf)
    aux_idx_proj = np.full([h, w], -np.inf)
    min_val_local = points.min(0)    
    for k in range(len(lidx)):
        h_coor = i_img_mapping[k]
        w_coor = j_img_mapping[k]

        #if h_coor >= h: h_coor = h - 1
        #if w_coor >= w: w_coor = w - 1
            
        if points[k, 2] > aux_height_proj[h_coor, w_coor]:
            max_projection[h_coor, w_coor] = labels[k]
            aux_height_proj[h_coor, w_coor] = points[k, 2]
    return max_projection, aux_idx_proj, [h,w,min_val_local], [lidx, i_img_mapping, j_img_mapping]

#def min_aggregation_proj()

def compute_gt_image(proj, points, labels, aggreg):
    if aggreg == 'max' : 
        gt_projection, _, local_parameters, mapping = max_aggregation_proj(proj, points, labels, aggreg)
    else:
        print("agg not supported")
        1 / 0
        
    return gt_projection, local_parameters, mapping


def map_gt_to_colored(gt_array, color_map):
    colored_array = np.zeros([gt_array.shape[0], gt_array.shape[1], 3], dtype=np.uint8)
    uu = np.unique(gt_array)
    for u in uu:
        idx = np.where(gt_array == u)
        colored_array[idx[0], idx[1], :] = color_map[u]
        
    return colored_array