import os
import argparse
import pickle
from tqdm import tqdm
import shutil
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('--root_dir', type=str, default="./data_av1/centerline_speed/test_150000_220000.pkl")
    parser.add_argument('--save_dir', type=str, default="./data_av1/feasible_traj" )
    parser.add_argument('--start_num', type=int, default=150000)
    parser.add_argument('--end_num', type=int, default=220000)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir
    save_dir = args.save_dir
    start_num = args.start_num
    end_num = args.end_num

    root_file_list = [fn.split(".")[0] for fn in os.listdir(root_dir)]
    save_file_list = [fn.split(".")[0] for fn in os.listdir(save_dir)]
    
    fail_file_list = []
    for fn in tqdm(root_file_list):
        if (int(fn) >= start_num) and (int(fn)<=end_num):
            if fn in save_file_list:
                pass
            else:
                fail_file_list.append(fn)
    fail_file_dict = {"fail_list":fail_file_list}
    if not os.path.exists("./data_av1/fail_val_1007/"):
        os.makedirs("./data_av1/fail_val_1007/")
    #root_dir = "/data2/queenie/av1_data/train/data" 
    for fn in fail_file_list:
        shutil.copy(os.path.join(root_dir,fn+".csv"),os.path.join("./data_av1/fail_val_1007/",fn+".csv"))
    with open("./data_av1/fail_val_1007/fail_file_%d_%d.pkl" %(start_num,end_num),'wb') as f:
        pickle.dump(fail_file_dict,f)

