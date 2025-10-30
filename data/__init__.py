from functools import partial
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

import core.util as Util
from core.praser import init_obj


def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0
        创建训练/测试数据加载器和验证数据加载器，当阶段为测试或不是 GPU 0 时，验证数据加载器为 None """
    '''4.1、create dataset and set random seed 创建数据集并设置随机种子'''
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args'] # 获取数据加载器的参数
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])# 定义工作进程初始化函数

    phase_dataset, val_dataset = define_dataset(logger, opt)# 4.2、创建数据集

    '''create datasampler 创建数据采样器'''
    data_sampler = None# 初始化数据采样器为 None
    if opt['distributed']: # 如果是分布式训练
        # 创建分布式数据采样器
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle # 因为使用了采样器，所以将数据加载器的 shuffle 参数设置为 False
    
    ''' create dataloader and validation dataloader 创建数据加载器和验证数据加载器'''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)# 创建数据加载器
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! 验证数据加载器只在 GPU 0 上运行，所以不使用 DistributedSampler！'''
    # if opt['global_rank']==2 and val_dataset is not None:# 如果是 GPU 0 且有验证数据集
    if val_dataset is not None:  # 如果是 GPU 0 且有验证数据集
        dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args',{}))# 更新数据加载器参数
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args) # 创建验证数据加载器
    else:
        val_dataloader = None # 否则验证数据加载器为 None
    return dataloader, val_dataloader# 返回数据加载器和验证数据加载器


def define_dataset(logger, opt):
    ''' 4.2、loading Dataset() class from given file's name '''
    ''' 4.2、从给定文件的名称加载 Dataset() 类 '''
    # 获取数据集配置选项 {'name': ['data.dataset', 'UncroppingDataset'], 'args': {'data_root': 'data/train_head.flist', 'data_len': -1, 'mask_config': {'mask_mode': 'head_pad'}}}
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']

    #phase_dataset = <data.dataset.UncroppingDataset object at 0x7f3c3aeda490>
    phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')# 4.2.1、初始化阶段数据集
    val_dataset = None# 初始化验证数据集为空

    valid_len = 0 # 验证集长度初始化为 0
    data_len = len(phase_dataset)# data_len = 9330，获取数据集中数据的数量 运行dataset.py函数class UncroppingDataset(data.Dataset)中的def __len__(self)函数
    if 'debug' in opt['name']:# 如果优化选项的名称中包含 'debug'
        debug_split = opt['debug'].get('debug_split', 1.0)# 获取调试分割比例
        if isinstance(debug_split, int):# 如果调试分割比例是整数
            data_len = debug_split# 数据长度设置为调试分割比例
        else:
            data_len *= debug_split# 数据长度乘以调试分割比例

    #dataloder_opt：{'validation_split': 2, 'args': {'batch_size': 3, 'num_workers': 4, 'shuffle': True, 'pin_memory': True, 'drop_last': True}, 'val_args': {'batch_size': 1, 'num_workers': 4, 'shuffle': False, 'pin_memory': True, 'drop_last': False}}
    dataloder_opt = opt['datasets'][opt['phase']]['dataloader']# 获取数据加载器配置选项
    #valid_split = 2
    valid_split = dataloder_opt.get('validation_split', 0) # 获取验证集分割比例
    
    ''' divide validation dataset, valid_split==0 when phase is test or validation_split is 0. '''
    ''' 划分验证数据集，当阶段为测试或验证集分割比例为 0 时，valid_split 为 0。 '''
    if valid_split > 0.0 or 'debug' in opt['name']: # 如果验证集分割比例大于 0 或名称中包含 'debug'
        if isinstance(valid_split, int):# 如果验证集分割比例是整数
            assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset." # 断言验证集大小小于数据集大小
            valid_len = valid_split# 验证集长度设置为验证集分割比例
        else:
            valid_len = int(data_len * valid_split)# 验证集长度为数据长度乘以验证集分割比例
        data_len -= valid_len # 数据长度减去验证集长度
        # 4.2.2、划分数据集为阶段数据集和验证数据集
        phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[data_len, valid_len], generator=Generator().manual_seed(opt['seed']))
    
    logger.info('Dataset for {} have {} samples.'.format(opt['phase'], data_len)) # 记录阶段数据集的样本数量信息
    if opt['phase'] == 'train':# 如果阶段为训练
        logger.info('Dataset for {} have {} samples.'.format('val', valid_len)) # 记录验证数据集的样本数量信息
    return phase_dataset, val_dataset # 返回阶段数据集和验证数据集

def subset_split(dataset, lengths, generator):
    """
    # 4.2.2、split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    # 4.2.2、将一个数据集分割成给定长度的不重叠的新数据集。主要代码来自 PyTorch 中的 random_split 函数"""
    indices = randperm(sum(lengths), generator=generator).tolist()  # 生成随机排列的索引列表
    Subsets = []  # 初始化子集列表
    for offset, length in zip(np.add.accumulate(lengths), lengths):  # 遍历累加长度和每个长度
        if length == 0:  # 如果长度为 0
            Subsets.append(None)  # 向子集列表添加 None
        else:
            Subsets.append(Subset(dataset, indices[offset - length: offset]))  # 创建并添加子集
    return Subsets  # 返回子集列表
