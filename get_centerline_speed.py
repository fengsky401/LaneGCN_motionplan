import argoverse
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import numpy as np
from pykalman import KalmanFilter
from argoverse.map_representation.map_api import ArgoverseMap
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import os
from tqdm import tqdm
import time
import argparse
from utils.others import process_input, process_xy, sample_speed, process_xy_back, get_direction, prediction, \
    generate_offset, convert_to_frenet, close_list_index, cal_dist, cal_paral, cal_speed_list, inc_length
from utils.frenet_optimal_trajectory import generate_target_course, frenet_optimal_planning, calc_frenet_paths
import torch

ACCELERATE_SEARCH = 1
SEARCH_SPEED = 10.0

def centerline_speed(avm, data, start):
    city_name_, data_ = process_input(data)

    ## filter the data

    x, y = process_xy(data_)
    measurements = data_
    initial_state_mean = [measurements[0, 0],
                          0,
                          measurements[0, 1],
                          0]

    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]

    kf1 = KalmanFilter(transition_matrices=transition_matrix,
                       observation_matrices=observation_matrix,
                       initial_state_mean=initial_state_mean)

    kf1 = kf1.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

    kf2 = KalmanFilter(transition_matrices=transition_matrix,
                       observation_matrices=observation_matrix,
                       initial_state_mean=initial_state_mean,
                       observation_covariance=10 * kf1.observation_covariance,
                       em_vars=['transition_covariance', 'initial_state_covariance'])

    kf2 = kf2.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf2.smooth(measurements)
    data_k = process_xy_back(smoothed_state_means[:, 0], smoothed_state_means[:, 2])

    ## dfs search
    # avm = ArgoverseMap()
    candidate_centerlines = avm.get_candidate_centerlines_for_traj(data_k[0:19], city_name_, viz=False)
    dfs = candidate_centerlines

    ## dfs pruning
    dfs_save_p = []
    for i in range(len(candidate_centerlines)):
        x, y = process_xy(candidate_centerlines[i])
        x_ = []
        y_ = []
        for i in range(0, len(x), 2):
            x_.append(x[i])
            y_.append(y[i])

        # ？为啥先取了一半的点，又用0.1的间隔算出这么多插值
        tx, ty, tyaw, tc, csp = generate_target_course(x_, y_)
        ## prune the line by finding the minimum distance
        tgt_dfs = process_xy_back(tx, ty)
        ind = close_list_index(tgt_dfs, data_[start:])  # ?找到距离最近的distance的index，两个问题：1.data_不能包括未来的轨迹吧 2.
        prune_dfs = tgt_dfs[ind:]  # ？tgt_dfs没有按顺序排序，不能用最近的index

        dfs_save_p.append(prune_dfs)
    ##remove overlap line
    overlap = []
    for i in range(len(dfs_save_p)):
        for j in range(i, len(dfs_save_p)):
            if i != j:
                dist = cal_dist(dfs_save_p[i][1], dfs_save_p[j][1]) + cal_dist(dfs_save_p[i][-1], dfs_save_p[j][
                    -1])  # ？为啥算两个centerline插值出来的第一个点+最后一个点的距离
                # print(dist)
                if dist < 0.5:
                    overlap.append(j)
                    break
            else:
                pass
    # print(len(overlap))
    # # print(len(dfs_save_p))
    save_line = []
    for i in range(len(dfs_save_p)):
        if i not in overlap:
            save_line.append(dfs_save_p[i])

    return save_line,data_k,data_


if __name__ == "__main__":
    ##set root_dir to the correct path to your dataset folder
    def parse_args():

        parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
        parser.add_argument('--root_dir', type=str, default='/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/train/data')
        parser.add_argument('--save_dir', type=str, default='/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/centerline_speed/train' )
        parser.add_argument('--start_num', type=int, default=0)
        parser.add_argument('--end_num', type=int, default=60)

        args = parser.parse_args()

        return args

    args = parse_args()
    root_dir = args.root_dir#'/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/train/data'
    save_dir = args.save_dir#'/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/save_train_frenet/test'
    start_num = args.start_num
    end_num = args.end_num
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # name=os.listdir(root_dir)
    Afl = ArgoverseForecastingLoader
    avm = ArgoverseMap()

    afl = Afl(root_dir)
    centerline_speed_data = {}
    for path_name_ext in tqdm(afl.seq_list):
        single_num = path_name_ext.parts[-1].split(".")[0]
        if (int(single_num) > start_num) and (int(single_num) < end_num):
            afl_ = afl.get(path_name_ext)
            path, name_ext = os.path.split(path_name_ext)
            name, ext = os.path.splitext(name_ext)
            time_begin = time.time()
            try:
                save_centerline, speed,raw_speed = centerline_speed(avm, afl_.seq_df, 19)
            except:
                continue
            centerline_speed_data[name] = {"centerline":save_centerline,
                                           "speed":speed,
                                           "raw_speed":raw_speed}
    torch.save(centerline_speed_data,os.path.join(args.save_dir,"centerline_speed_%d_%d.pt" %(args.start_num,args.end_num)))
