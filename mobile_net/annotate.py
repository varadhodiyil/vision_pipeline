import scipy.io as sio
import os
import sys
path= "/home/madhan/Downloads/car_devkit/devkit/cars_meta.mat"

mat_contents = sio.loadmat(path,squeeze_me=True,struct_as_record=False)
_conts = mat_contents['class_names']
files_names = {"sedan": [] , "hatchback":[]}
for i , lbl in enumerate(_conts):
    if 'Sedan' in lbl:
        files_names['sedan'].append(i+1)
    elif 'Hatchback' in lbl:
        files_names['hatchback'].append(i+1)
# idx = [i[0] for i in files_names]
# names = [i[1] for i in files_names]
# print(files_names)
_files = list()
sedan_idx = files_names['sedan']
hatchback_idx = files_names['hatchback']

sedan_files = []
hatchback_files = []
# print(sedan_idx)
path= "/home/madhan/Downloads/car_devkit/devkit/cars_train_annos.mat"
mat_contents = sio.loadmat(path,squeeze_me=True,struct_as_record=False)
for m in mat_contents['annotations']:
    m = m.__dict__
    if m['class'] in sedan_idx:
        sedan_files.append(m['fname'].lower())

    elif m['class'] in hatchback_idx:
        hatchback_files.append(m['fname'].lower())
#     if m['class'] in idx:
#         _class = m['class']
#         for _f in files_names:
#             if m['class'] == _f[0]:
#                 _files.append((_f[1], m['fname']))
# print(len(_files))
# idx = [i[1] for i in _files]
# print(hatchback_files)
# print('00004.jpg' in sedan_files)
for r,_,t_img in os.walk("cars_train"):
    for f in t_img:
        exec_f = None
# #         break
        # print(f in sedan_files)
#         # print(f in sedan_files)
        if  f in sedan_files:
            exec_f = "cp {0}/{1} data/{2}/{1}".format(r,f , "sedan")
        elif f in hatchback_files:
            exec_f = "cp {0}/{1} data/{2}/{1}".format(r,f , "hatchback")
        # print(exec_f)
        
        if exec_f:
            os.popen(exec_f)
#             for __f in _files:
#                 if __f[1] == f:
#                     
#                     # print(exec_f)
#                     os.popen(exec_f)

            