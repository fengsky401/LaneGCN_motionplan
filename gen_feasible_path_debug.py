#import torch
import argparse
import os
cpu_num = 1 # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
#torch.set_num_threads(cpu_num)
import numpy as np
from tqdm import tqdm
#import torch
import time
from utils.others import process_input, process_xy, sample_speed, process_xy_back, get_direction, prediction, \
    generate_offset, convert_to_frenet, close_list_index, cal_dist, cal_paral, cal_speed_list, inc_length
from utils.frenet_optimal_trajectory import generate_target_course, frenet_optimal_planning, calc_frenet_paths
'''
cpu_num = 1 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
#torch.set_num_threads(cpu_num)
'''
ACCELERATE_SEARCH = 1
SEARCH_SPEED = 20.0

def gen_feasible_traj(save_line,data_k,data_,start):
    input = save_line
    generate_trj = []
    goal_thre = 99
    num_sample = 35  # ?num_sample代表啥,为啥给35
    up_to = 30  ## how many points for our prediction points
    track_length = 3000  # track length代表啥,为啥给3000
    speed = cal_speed_list(data_k[0:19])
    ob_ = np.array([[0, 0]])  # ？假设没有障碍？
    if speed[-1] > 33:  ##this case we dont need planning just keep going!
        generate_trj = []
        for i in range(35):
            ratio = np.arange(0.5, 1, 0.0005)
            m = inc_length(data_k[-3:-1], up_to=30, ratio=np.random.choice(
                ratio))  ##becase we are not sure about the acc, so i sample the gradient
            generate_trj.append(m)
    else:
        print("num of centerline:", len(input))
        
        for j in range(len(input)):
            all_frenet_begin = time.time()
            count_planning_num = 0
            if len(input[j]) > 30:

                for i in range(num_sample):
                    # print(j)
                    begin_sample = time.time()
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
                    #print("each sample cost:",time.time()-begin_sample)
                    if abs(c_d) > 4:  # ？为什么c_d>4的时候不做motion planning
                        pass
                    #print("each sample cost:",time.time()-begin_sample)
                    
                    else:
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
                            #print("once time cost:",once_time_cost)
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


if  __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
        parser.add_argument('--pre_file', type=str, default="./data_av1/centerline_speed/test_150000_220000.pkl")
        parser.add_argument('--save_dir', type=str, default="./data_av1/feasible_traj" )
        parser.add_argument('--start_num', type=int, default=150000)
        parser.add_argument('--end_num', type=int, default=220000)
        args = parser.parse_args()
        return args

    import pickle
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    ct = None
    with open(args.pre_file,'rb') as f:
        ct = pickle.load(f)
    print("after load")
    pre_file = args.pre_file
    save_dir = args.save_dir
    start_num = args.start_num
    end_num = args.end_num
    #torch.set_num_threads(1)
    #centerline_speed_dict = torch.load(pre_file)
    for key,value in tqdm(ct.items()):
        if (int(key) > start_num) and (int(key) < end_num):
            begin=time.time()
            argo_id = key
            save_centerline =value["centerline"]
            speed = value["speed"]
            raw_speed = value["raw_speed"]
            #save_centerline, generate_traj = [],[]
            save_centerline, generate_traj = gen_feasible_traj(save_centerline, speed, raw_speed,20)
            data_dict = {"save_centerline": save_centerline, "generate_traj": generate_traj}
            save_begin = time.time()
            with open(os.path.join(save_dir, argo_id + ".path"),'wb') as fw:
                pickle.dump(data_dict,fw)
            
            #torch.save(data_dict, os.path.join(save_dir, argo_id + ".path"))
            print("save cost:",time.time()-save_begin)
            print("time cost per scene:",time.time()-begin)

