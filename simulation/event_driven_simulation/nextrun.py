import os
from datetime import date


def NextRun(dir):
    path_to_file = dir+"/counter.txt"
    file_exists = os.path.exists(path_to_file)

    if file_exists:
        with open(dir+"/counter.txt", 'r+') as f:
            dir_index = f.readline()
            dir_path = dir + "/" + dir_index
            f.seek(0)  ## find the beginning of the file for overwrite
            f.write(str(int(dir_index) + 1))
            f.truncate() 
            f.close()
    else:
        os.mkdir(dir)
        with open(dir+"/counter.txt", 'w') as f:
            dir_path = dir + "/1"
            f.write("2")
            f.close()

    os.mkdir(dir_path)
    os.mkdir(dir_path + "/data")
    # print("Experiment results stored in {}".format(dir_path))

    return dir_path

if __name__ == "__main__":
    today = date.today()
    datestring = today.strftime("exp-%y-%m-%d/")
    data_dir = "/home/zhizhenz/lightning-master/data/scheduling_results/{}/".format(datestring)
    dir = NextRun(data_dir)
    print(dir)