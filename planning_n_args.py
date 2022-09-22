from logging import root
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
#from utils.map_features_utils import MapFeaturesUtils
#from utils.visualize_sequences import interpolate_polyline, viz_sequence
#import pickle as p
#import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap
import numpy as np
#import torch
from utils.others import process_input, process_xy, process_xy_back
#from pykalman import KalmanFilter
#from scipy import poly1d
#from pykalman import KalmanFilter
from argoverse.map_representation.map_api import ArgoverseMap
#from utils.others import process_input, process_xy, sample_speed, process_xy_back, get_direction, prediction, \
#    generate_offset, convert_to_frenet, close_list_index, cal_dist, cal_paral, cal_speed_list, inc_length
#from utils.frenet_optimal_trajectory import generate_target_course, frenet_optimal_planning, calc_frenet_paths
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
#import warnings

import argparse

#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#from multiprocessing import Pool
#from itertools import product
from tqdm import tqdm
import time

ACCELERATE_SEARCH = 1
SEARCH_SPEED = 10.0


def model_generator(avm, data, start):
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

    '''
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
    '''
    ## dfs search
    # avm = ArgoverseMap()
    #candidate_centerlines = avm.get_candidate_centerlines_for_traj(data_k[0:19], city_name_, viz=False)
    candidate_centerlines = avm.get_candidate_centerlines_for_traj(data_[0:19], city_name_, viz=False)
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

    # np.save(f"{save_dir}/pdfs_{data_name}.npy",np.asarray(dfs_save_p))
    ## search for traj
    input = save_line
    generate_trj = []
    goal_thre = 99
    num_sample = 35  # ?num_sample代表啥,为啥给35
    up_to = 30  ## how many points for our prediction points
    track_length = 3000  # track length代表啥,为啥给3000
    speed = cal_speed_list(data_[0:19])
    ob_ = np.array([[0, 0]])  # ？假设没有障碍？
    if speed[-1] > 33:  ##this case we dont need planning just keep going!
        generate_trj = []
        for i in range(35):
            ratio = np.arange(0.5, 1, 0.0005)
            m = inc_length(data_[-3:-1], up_to=30, ratio=np.random.choice(
                ratio))  ##becase we are not sure about the acc, so i sample the gradient
            generate_trj.append(m)
    else:
        pass
    '''
        print("num of centerline:", len(input))
        for j in range(len(input)):
            all_frenet_begin = time.time()
            count_planning_num = 0
            if len(input[j]) > 30:

                for i in range(num_sample):
                    # print(j)
                    offset = np.round(np.random.choice(np.linspace(-2, 2, 9)),
                                      2)  # ？为啥随机取 ## sample the offset from (-lane/2,lane/2)
                    # print(offset)
                    # offset=2
                    prune_dfs_ = generate_offset(input[j][0:track_length], offset=offset)  # 返回centerline的左右两个边界

                    c_speed = speed[-1]
                    target_speed = sample_speed(speed[-1])  # ?用最后一个时刻的速度随便猜一个target速度

                    x1 = prune_dfs_[1][0]
                    x0 = prune_dfs_[0][0]
                    y1 = prune_dfs_[1][1]
                    y0 = prune_dfs_[0][1]
                    x2 = data_[start][0]
                    y2 = data_[start][1]
                    # ?为什么只看prune_dfs的第0个点和第1个点
                    if (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0) < 0:
                        c_d = -cal_dist(data_[start], prune_dfs_[
                            0])  # current lateral position [m] ## need to get from the current agent's postion offset has problem
                    else:
                        c_d = cal_dist(data_[start], prune_dfs_[0])
                    # 计算
                    if abs(c_d) > 4:
                        prune_dfs2 = generate_offset(input[j][0:track_length], offset=-offset)
                        prune_dfs_ = cal_paral(prune_dfs_=input[j][0:track_length], prune_dfs_2=prune_dfs2)

                    x1 = prune_dfs_[1][0]
                    x0 = prune_dfs_[0][0]
                    y1 = prune_dfs_[1][1]
                    y0 = prune_dfs_[0][1]
                    x2 = data_[start][0]
                    y2 = data_[start][1]

                    if (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0) < 0:
                        c_d = -cal_dist(data_[start], prune_dfs_[
                            0])  # current lateral position [m] ## need to get from the current agent's postion offset has problem
                    else:
                        c_d = cal_dist(data_[start], prune_dfs_[0])

                    if abs(c_d) > 4:  # ？为什么c_d>4的时候不做motion planning
                        pass

                    else:
                        pass
                        
                        #                       #TODO:Line187-191
                        c_d = c_d
                        c_d_d = 0.0  # current lateral speed [m/s]
                        c_d_dd = 0.0  # current lateral acceleration [m/s]
                        s0 = 0  # current course position, it means where to start track on the target lane
                        sim_loop = track_length
                        xs = []
                        ys = []

                        tx, ty, tyaw, tc, csp = convert_to_frenet(prune_dfs_)  # 把车道线转化到frenet坐标系下
                        
                        for k in range(sim_loop):  # ?为什么要循环sim_loop次
                            count_planning_num += 1
                            one_loop_begin = time.time()
                            try:
                                path = frenet_optimal_planning(
                                    csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob=ob_, target_speed=target_speed)
                                # if k%10 == 0 and (k>0):
                                #     print("k:%d,s0:%f,c_speed:%f,c_d:%f,c_d_d:%f,c_d_dd:%f,target_speed:%f" %(k,s0,c_speed,c_d,c_d_d,c_d_dd,target_speed))
                            except:
                                break
                            if path is None:
                                break

                            once_time_cost = time.time() - one_loop_begin
                            # print("once time cost:",once_time_cost)
                            if path == None:
                                # print('no path!')
                                break

                            if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= goal_thre:
                                xs = xs + path.x
                                ys = ys + path.y
                                # print("Goal")
                                break

                            if ACCELERATE_SEARCH == 1:
                                s0 = s0 + (path.s[1] - s0) * SEARCH_SPEED
                                c_d = c_d + (path.d[1] - c_d) * SEARCH_SPEED
                                c_speed = c_speed + (path.s_d[1] - c_speed) * SEARCH_SPEED
                                c_d_d = c_d_d + (path.d_d[1] - c_d_d) * SEARCH_SPEED
                                c_d_dd = c_d_dd + (path.d_dd[1] - c_d_dd) * SEARCH_SPEED

                            else:
                                s0 = path.s[1]  # s方向位移
                                c_d = path.d[1]  # d方向位移
                                c_speed = path.s_d[1]  # s方向速度
                                c_d_d = path.d_d[1]  # d方向速度
                                c_d_dd = path.d_dd[1]  # d方向加速度
                                xs = xs + path.x  # 笛卡尔坐标系下x坐标
                                ys = ys + path.y  # 笛卡尔坐标系下y坐标
                        # pre_x,pre_y=get_direction(process_xy_back(xs,ys))
                        if len(ys) > 2:  ## if we have find a path that is not none,
                            pre_x, pre_y = np.gradient(xs), np.gradient(ys)
                            # print(pre_x[0],pre_y[0])
                            # d_x,d_y=get_direction(data_)
                            x, y = process_xy(data_)
                            d_x, d_y = np.gradient(x), np.gradient(y)
                            # print(d_x[0],d_y[0])
                            if cal_dist(data_[start], process_xy_back(xs, ys)[
                                0]) > 10:  ## we dont want the wrong planned traj or (pre_x[0]*d_x[0]<0 and pre_y[0]*d_y[0]<0)(not sure whether we need this)
                                # generate_trj.append(process_xy_back(xs,ys))
                                pass
                            else:
                                generate_trj.append(inc_length(process_xy_back(xs, ys), up_to=up_to, ratio=0.2))
                
            all_frenet_cost = time.time() - all_frenet_begin
            if count_planning_num == 0:
                print("%d planning iteration,frenet cost in one centerline:%f,average cost is %f"
                      % (count_planning_num, all_frenet_cost, 0.0))
            else:
                print("%d planning iteration,frenet cost in one centerline:%f,average cost is %f"
                      % (count_planning_num, all_frenet_cost, all_frenet_cost / count_planning_num))
    ## save data process here: cut line to the minimum predicted cl:
    '''
    len_line = []
    for i in range(len(save_line)):
        len_line.append(len(save_line[i]))
    len_line.sort()
    save_line_ = []
    for i in range(len(save_line)):
        save_line_.append(inc_length(save_line[i], len_line[0]))
    # np.save("")
    # np.save(f"{save_dir}/trj_{data_name}.npy",np.asarray(generate_trj))
    # print('data',np.asarray(save_line[0],np.float32),np.asarray(generate_trj[0],np.float32))
    return np.asarray(save_line_, np.float32), np.asarray(generate_trj,
                                                          np.float32)  ## change the type as np.float 32? length problem, still remaining need to determine how to prune


def multi_run_wrapper(args):
    return model_generator(*args)


if __name__ == "__main__":
    ##set root_dir to the correct path to your dataset folder
    def parse_args():

        parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
        parser.add_argument('--root_dir', type=str, default=None)
        parser.add_argument('--save_dir', type=str, default=None )
        parser.add_argument('--start_num', type=int, default=0)
        parser.add_argument('--end_num', type=int, default=0)

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
    print('Total number of sequences:', len(afl))
    info_dict = []
    check_list = ["52", "39", "60"]
    for path_name_ext in tqdm(afl.seq_list):
        single_num = path_name_ext.parts[-1].split(".")[0]
        if (int(single_num) > start_num) and (int(single_num) < end_num):
            afl_ = afl.get(path_name_ext)
            path, name_ext = os.path.split(path_name_ext)
            name, ext = os.path.splitext(name_ext)
            time_begin = time.time()
            #save_centerline, generate_traj = model_generator(avm, afl_.seq_df, 20)
            save_centerline, generate_traj  = [],[]
            data_dict = {"save_centerline": save_centerline, "generate_traj": generate_traj}
            torch.save(data_dict, os.path.join(save_dir, name + ".path"))
            time_cost = time.time() - time_begin
            print("time cost per scene:", time_cost)

    # model_generator()
    # model_generator()

    # pool = Pool(processes=1)
    # pool.map(multi_run_wrapper,[(save_dir,root_dir,name)])
    # pool.close()
    # pool.join()
