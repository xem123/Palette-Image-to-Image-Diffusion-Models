# def check_utf8_encoding(file_path):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             file.read()
#         print('true')
#         return True
#     except UnicodeDecodeError:
#         print('False')
#         return False

import glob
import os

# 数据父目录
filepath = 'H:/Palette-Image-to-Image-Diffusion-Models/data/'
# 拿到所有路径
filelist = glob.glob(filepath + '*.png')

# 打开或创建 train.flist 文件并写入路径
with open("D:/Diffusion_model/data/test.flist", "w") as fo:
    for path in filelist:
        basename = os.path.basename(path)
        temp_path = filepath + basename
        fo.write(temp_path + '\n')
