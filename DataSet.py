import os

DataSet_Path = 'F:\\Dropbox\\Code_DataSet'

def data_dir():
    dir_list = []
    for root, dirs, _ in os.walk(DataSet_Path):
        for d in dirs:
            dir_list.append(os.path.join(root, d))
    return dir_list
