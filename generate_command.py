import os
process_num = 4
process_dir = "/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/train/data"
save_dir = "/Users/queenie/Documents/LaneGCN_Tianyu/data_av1/save_train_frenet/test_limit"
command_file_name = "train_command.sh"
keyword = "train"
filenum_list = [int(i.split(".")[0]) for i in os.listdir(process_dir)]
file_num = len(filenum_list)
max_filenum = max(filenum_list)
min_filenum = min(filenum_list)
each_increment = max_filenum//process_num
with open(command_file_name,"w") as f:

    for i in range(process_num):
        start_num = each_increment*i
        end_num = each_increment*(i+1)
        command_line = "nohup python planning_n_args.py --root_dir %s --save_dir %s  --start_num %d  --end_num  %d >a_%s_%d.out 2>&1 & \n" \
                       %(process_dir,save_dir,start_num,end_num,keyword,i)
        f.writelines(command_line)




