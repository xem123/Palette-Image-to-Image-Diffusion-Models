import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import cv2
import imageio
import torch.nn.functional as F

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox,head_pad_crop)

# 定义了支持的图像文件扩展名列表
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# 定义一个函数，用于判断给定的文件名是否为图像文件，通过检查文件名是否以支持的扩展名结尾
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# 4.2.2.1、定义一个函数来创建数据集，根据给定的目录路径，返回其中所有图像文件的路径列表。如果给定的是文件路径，则从该文件中读取图像路径；如果是目录路径，则遍历目录及其子目录，收集所有图像文件的路径
def make_dataset(dir):#dir：'data/train_head.flist'
    if os.path.isfile(dir):
        # images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]# images=['data/train_head_data_png16/1024_1024_622_10_0.png', 'data/train_head_data_png16/1024_1024_622_10_1.png', ...]
        images = [i for i in np.genfromtxt(dir, dtype=np.dtype('str'), encoding='utf-8')]# images=['data/train_head_data_png16/1024_1024_622_10_0.png', 'data/train_head_data_png16/1024_1024_622_10_1.png', ...]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images

# 定义一个函数，用于使用PIL库读取图像并将其转换为RGB模式
def pil_loader(path):
    # print('path:',path)
    img_ori = Image.open(path)
    # img_ori.save(os.path.join('./data/tmp_data/', 'img_ori.png'))
    # RGB_img = img_ori.convert('RGB')
    # # RGB_img.save(os.path.join('data/tmp_data/', 'RGB_img.png'))
    # # print('RGB_img[0,1,:]:',RGB_img[0,1,:])
    # return RGB_img # 所有进入网络的图像都会转成三通道的8位RGB图像
    # # return Image.open(path).convert('RGB')

    # 保持16位数据，不转换为8位
    # print("img_ori.mode = ",img_ori.mode )
    if img_ori.mode in ['I;16', 'I;16N', 'I;16B', 'I;16L', 'I']:
        # 直接返回16位图像，不进行转换
        # 如果需要转换为RGB，但保持16位
        if img_ori.mode != 'RGB':
            # 将16位灰度转换为16位RGB
            img_array = np.array(img_ori).astype('uint16')
            # cv2.imwrite('./data/tmp_data/img_array_16bit.png', img_array)
            # print('img_array.shape:',img_array.shape)
            # print('img_array[:,200]:',img_array[:,200])

            # 进行最大最小值归一化，训练时使用
            # min_val = img_array.min()  # 图像全局最小值
            # max_val = img_array.max()  # 图像全局最大值
            min_val = 0  # 所有训练图像的全局最小值
            max_val = 21000  # 所有训练图像的全局最大值
            # 处理纯色图像（避免除零错误）
            if max_val > min_val:
                # 应用归一化公式，四舍五入后转为uint16（保持16位精度）
                img_array_nor = ((img_array - min_val) / (max_val - min_val) * 65535).round().astype('uint16')

            # # ------------ 像素值扩散60倍 ------------
            # # 1. 转换为uint32避免乘法溢出
            # img_array = img_array.astype('uint32')
            # # 2. 所有像素值乘以60
            # img_array_nor = img_array * 60
            # # 3. 截断到16位图像的有效范围（0~65535）
            # img_array_nor = np.clip(img_array_nor, 0, 65535)
            # # 4. 转回uint16类型
            # img_array_nor = img_array_nor.astype('uint16')

            # 堆叠为三通道（RGB格式，HWC）
            RGB_img = np.stack([img_array_nor, img_array_nor, img_array_nor], axis=2).astype(np.uint16)  # 形状为 (H, W, 3)，数据类型uint16
            # RGB_img = np.stack([img_array, img_array, img_array], axis=2).astype(np.uint16)  # 形状为 (H, W, 3)，数据类型uint16
            # print('RGB_img.shape:',RGB_img.shape)
            # print('RGB_img[:,200,0]:',RGB_img[:,200,0])
            # print("转换后的数据类型：", RGB_img.dtype)
            # print("------")
            RGB_img_PIL = cv2.cvtColor(RGB_img.astype(np.uint16), cv2.COLOR_RGB2BGR).astype(np.uint16) # 关键：OpenCV默认使用BGR通道顺序，需将RGB转换为BGR
            # print('RGB_img_PIL.shape:', RGB_img_PIL.shape)
            # print('RGB_img_PIL[:,200,0]:', RGB_img_PIL[:, 200, 0])
            # print("转换后的数据类型：", RGB_img_PIL.dtype)  # 必须输出uint16
            # print("-------------------")
            # # 用OpenCV保存三通道16位图像
            # cv2.imwrite('./data/tmp_data/img_array_rgb_16bit_cv2.png', RGB_img_PIL[:, :, 0])
            # img_dtype = RGB_img_PIL.dtype  # 数据类型（应为uint16）
            # img_min = np.min(RGB_img_PIL)  # 计算全局最小值
            # img_max = np.max(RGB_img_PIL)  # 计算全局最大值
            # print(f"[16位图像] 路径：{path} | 数据类型：{img_dtype} | 最小值：{img_min} | 最大值：{img_max}")# 数据类型：uint16 | 最小值：8 | 最大值：1023

    else:
        # img_ori.save(os.path.join('data/tmp_data/', 'img_ori.png'))
        RGB_img_PIL = img_ori.convert('RGB')
        # RGB_img.save(os.path.join('data/tmp_data/', 'RGB_img.png'))
        # print('RGB_img[0,1,:]:',RGB_img[0,1,:])
        # return RGB_img # 所有进入网络的图像都会转成三通道的8位RGB图像
        # return Image.open(path).convert('RGB')

    return RGB_img_PIL

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[96, 128], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'head_pad':
            mask = head_pad_crop(self.image_size) # 头部padding掩码
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

# 定义一个名为UncroppingDataset的数据集类，继承自PyTorch的data.Dataset类
class UncroppingDataset(data.Dataset):
    # 4.2.2、数据集类的初始化函数，接收数据根目录、掩码配置、数据长度、图像大小和图像加载器等参数
    # def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[768, 1024], loader=pil_loader):
    # 全局函数：将 numpy 数组转换为 PIL 图像（16位RGB）

    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[192, 256], loader=pil_loader):#将768*1024按比例缩放到384*512、192*256
    # def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[952, 1280],loader=pil_loader):
    # def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[238, 320], loader=pil_loader):
        imgs = make_dataset(data_root) # 4.2.2.1、调用make_dataset函数获取数据根目录下的图像路径列表 imgs =['data/train_head_data_png16/1024_1024_622_10_0.png', 'data/train_head_data_png16/1024_1024_622_10_1.png', ...]
        if data_len > 0: # 根据数据长度参数截取图像路径列表
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        # 定义图像变换序列，包括调整大小、转换为张量和归一化
        self.tfs = transforms.Compose([
                # 先将numpy数组转为PIL图像（8位或16位），再Resize
                # lambda x: Image.fromarray(x.astype(np.uint16), mode='RGB') if isinstance(x, np.ndarray) else x,
                # numpy_to_pil,  # 使用全局函数替换 lambda
                # 将图像调整为指定的大小（image_size[0]和image_size[1]）
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),#将图像转换为张量格式，并缩放到0-1之间
                transforms.ConvertImageDtype(torch.float32),
                # #对8位图像进行归一化处理，将像素值的范围从[0, 255]转换为[-1, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
                # transforms.Normalize(mean=[32767.5, 32767.5, 32767.5],std=[32767.5, 32767.5, 32767.5])
        ])
        self.loader = loader        # 保存图像加载器
        self.mask_config = mask_config # 保存掩码配置
        self.mask_mode = self.mask_config['mask_mode'] # 获取掩码模式
        self.image_size = image_size # 保存图像大小

    def __getitem__(self, index): # 实现获取数据集中单个样本的功能
        ret = {}# 创建一个空的字典用于存储样本数据
        path = self.imgs[index]# 获取当前样本的图像路径

        img_tensor = self.tfs(self.loader(path))

        # # 打印单张图像的最值（维度：[C, H, W]，如[3, 192, 256]）
        # print(f"图像 {path} 数值范围：")
        # print(f"  最小值：{img_tensor.min().item():.6f}")
        # print(f"  最大值：{img_tensor.max().item():.6f}")
        # print(f"  数据类型：{img_tensor.dtype}")  # 数据类型：torch.float32

        # 对img进行resize尺寸
        img = F.interpolate(img_tensor.unsqueeze(0),
                                       size=(192, 256),
                                       mode='bilinear',
                                       align_corners=False).squeeze(0)
        # print(img.shape)  # 输出: torch.Size([3, 192, 256])
        # print('img[0,:, 200]:', img[0,:, 200])
        # # 1. 将 Tensor 转换为 NumPy 数组（先移到 CPU，再转 numpy）
        # img_np = img.cpu().numpy()  # 转为 NumPy 数组 [C, H, W]
        # print('img_np.shape:',img_np.shape) # (3, 192, 256)
        # img_np_r = np.transpose(img_np, (1, 2, 0))  # 转置为 [H, W, C]
        # print('img_np_r.shape:', img_np_r.shape) # (192, 256, 3)
        # img_np_aa = ((img_np_r + 1) * 32767.5).astype('uint16')
        # cv2.imwrite('./data/tmp_data/img_np_aa_BGR_uint16.png', img_np_aa[:, :, 0])
        # print(f'img.shape: {img.shape}, img.dtype: {img.dtype}') # img.shape: torch.Size([3, 192, 256]), img.dtype: torch.float32

        mask = self.get_mask()# 获取掩码
        # mask_np = mask.cpu().numpy()  # 转为 NumPy 数组 [1, H, W]
        # mask_np = np.transpose(mask_np, (1, 2, 0))  # 转置为 [H, W, 1]
        # mask_np_aa = ((mask_np + 1) * 32767.5).astype('uint16')  # 注意：掩码可能不需要 +1，根据实际范围调整
        # cv2.imwrite('./data/tmp_data/mask_uint16.png', cv2.cvtColor(mask_np_aa, cv2.COLOR_RGB2BGR)[:, :, 0])
        # print(f'mask.shape: {mask.shape}, mask.dtype: {mask.dtype}') # mask.shape: torch.Size([1, 192, 256]), mask.dtype: torch.uint8

        cond_image = img*(1. - mask) + mask*torch.randn_like(img)# 生成条件图像
        # cond_image_np = cond_image.cpu().numpy()  # 转为 NumPy 数组
        # cond_image_np = np.transpose(cond_image_np, (1, 2, 0))
        # cond_image_np_aa = ((cond_image_np + 1) * 32767.5).astype('uint16')
        # cv2.imwrite('./data/tmp_data/cond_image_uint16.png', cv2.cvtColor(cond_image_np_aa, cv2.COLOR_RGB2BGR)[:, :, 0])
        # print(f'cond_image.shape: {cond_image.shape}, cond_image.dtype: {cond_image.dtype}') # cond_image.shape: torch.Size([3, 192, 256]), cond_image.dtype: torch.float32

        mask_img = img*(1. - mask) + mask# 生成掩码图像
        # mask_img_np = mask_img.cpu().numpy()  # 转为 NumPy 数组
        # mask_img_np = np.transpose(mask_img_np, (1, 2, 0))
        # mask_img_np_aa = ((mask_img_np + 1) * 32767.5).astype('uint16')
        # cv2.imwrite('./data/tmp_data/mask_img_uint16.png', cv2.cvtColor(mask_img_np_aa, cv2.COLOR_RGB2BGR)[:, :, 0])
        # print(f'mask_img.shape: {mask_img.shape}, mask_img.dtype: {mask_img.dtype}') # mask_img.shape: torch.Size([3, 192, 256]), mask_img.dtype: torch.float32

        # 将图像、条件图像、掩码图像、掩码和路径等数据存储在字典中
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]

        # # 保存 img
        # img_np = np.array(ret['gt_image'].permute(1, 2, 0))
        # img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        # # img_pil = Image.fromarray((img * 255).astype(np.uint8))
        # img_pil.save(os.path.join('data/tmp_data/', 'img.png'))
        # # 保存 cond_image
        # cond_image_np = np.array(ret['cond_image'].permute(1, 2, 0))
        # cond_image_pil = Image.fromarray((cond_image_np * 255).astype(np.uint8))
        # # cond_image_pil = Image.fromarray((cond_image * 255).astype(np.uint8))
        # cond_image_pil.save(os.path.join('data/tmp_data/', 'cond_image.png'))
        # # 保存 mask_img
        # mask_img_np = np.array(ret['mask_image'].permute(1, 2, 0))
        # mask_img_pil = Image.fromarray((mask_img_np * 255).astype(np.uint8))
        # # mask_img_pil = Image.fromarray((mask_img * 255).astype(np.uint8))
        # mask_img_pil.save(os.path.join('data/tmp_data/', 'mask_img.png'))

        return ret

    def __len__(self):
        return len(self.imgs)        # 返回数据集的长度，即图像数量

    def get_mask(self):  # 根据设置的掩码模式，定义获取掩码的函数
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'head_pad':
            mask = head_pad_crop(self.image_size) # 头部padding掩码
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')

        #保存mask掩码图像
        # np.save('data/tmp_data/mask.png', mask)

        return torch.from_numpy(mask).permute(2,0,1)  # 将NumPy数组转换为PyTorch张量并进行维度调整


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


