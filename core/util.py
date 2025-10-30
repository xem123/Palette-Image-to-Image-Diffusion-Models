import random
import numpy as np
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid

# def tensor2img(tensor, out_type=np.uint16, min_max=(-1, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     将一个 PyTorch 张量转换为图像的 NumPy 数组。
#     输入：4D(B,(3/1),H,W)，3D(C,H,W)，或 2D(H,W)，任意范围，RGB 通道顺序。
#     输出：3D(H,W,C)或 2D(H,W)，[0,255]，np.uint8（默认）。
#     '''
#     tensor = tensor.clamp_(*min_max)  # clamp # 对张量进行截断，确保其值在指定的范围（min_max）内。
#     n_dim = tensor.dim() # 获取输入张量的维度。
#     if n_dim == 4:
#         n_img = len(tensor) # 如果是 4 维张量，获取张量中的图像数量。
# 		# 使用 make_grid 函数将多个图像张量组合成一个网格，nrow 指定每行显示的图像数量，normalize=False 表示不进行归一化，然后将结果转换为NumPy 数组。
#         img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
# 		# 将数组的维度顺序从（H，W，C）转换为（H，W，C），确保是 HWC（高、宽、通道）顺序，RGB 通道顺序。
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 3:
#         img_np = tensor.numpy()# 如果是 3 维张量，直接将其转换为 NumPy 数组
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB # 同样将维度顺序调整为 HWC，RGB。
#     elif n_dim == 2:
#         img_np = tensor.numpy() # 如果是 2 维张量，转换为 NumPy 数组。
#     else:
#         raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
# 	# 如果输出类型是 np.uint8，将图像数组的值从[-1,1]范围映射到[0,255]范围，并进行四舍五入。
# 	# 如果输出类型是 np.uint16，将图像数组的值从[-1,1]范围映射到[0,65535]范围，并进行四舍五入。
#     if out_type == np.uint16:
#         img_np = ((img_np+1) * 32767.5).round()
#         # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
#     return img_np.astype(out_type).squeeze()

def tensor2img(tensor, out_type=np.uint16, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    将一个 PyTorch 张量转换为图像的 NumPy 数组。
    输入：4D(B,(3/1),H,W)，3D(C,H,W)，或 2D(H,W)，任意范围，RGB 通道顺序。
    输出：3D(H,W,C)或 2D(H,W)，[0,255]，np.uint8（默认）。
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint16: # 如果输出类型是 np.uint16，将图像数组的值从[-1,1]范围映射到[0,65535]范围，并进行四舍五入。
        img_np = ((img_np+1) * 32767.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()

def postprocess(images):
	# print('images:',images)
	# print('---------------------')
	return [tensor2img(image) for image in images]


def set_seed(seed, gl_seed=0):
	"""  set random seed, gl_seed used in worker_init_fn function """
	if seed >=0 and gl_seed>=0:
		seed += gl_seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
		speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
	if seed >=0 and gl_seed>=0:  # slower, more reproducible
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:  # faster, less reproducible
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True

def set_gpu(args, distributed=False, rank=0):
	""" set parameter to gpu or ddp """
	if args is None:
		return None
	if distributed and isinstance(args, torch.nn.Module):
		return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
	else:
		return args.cuda()
		
def set_device(args, distributed=False, rank=0):
	""" set parameter to gpu or cpu """
	if torch.cuda.is_available():
		if isinstance(args, list):
			return (set_gpu(item, distributed, rank) for item in args)
		elif isinstance(args, dict):
			return {key:set_gpu(args[key], distributed, rank) for key in args}
		else:
			args = set_gpu(args, distributed, rank)
	return args



