import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric

def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """

    """  1、在每个 GPU 上运行的线程 """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    '''2、设置种子和 cuDNN 环境 '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    ''' 3、设置日志记录器 '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    '''4、设置网络和数据集'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    ''' 设置指标、损失、优化器和调度器 '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train() #5、开始训练 跳进base_model.py中的def train(self)
        else:
            model.test()
    finally:
        phase_writer.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()# 创建一个 argparse 解析器对象
    # 添加一个命令行参数 -c 或 --config，用于指定配置文件的路径，类型为字符串，默认值为 'config/colorization_mirflickr25k.json'，帮助信息为 'JSON file for configuration'
    parser.add_argument('-c', '--config', type=str, default='config/uncropping_places2.json', help='JSON file for configuration')
    # 添加一个命令行参数 -p 或 --phase，用于指定运行的阶段（训练或测试），类型为字符串，可选值为 ['train', 'test']，帮助信息为 'Run train or test'，默认值为 'train'
    # parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='test')
    # 添加一个命令行参数 -b6
    # +或 --batch，用于指定每个 GPU 上的批大小，类型为整数，默认值为 None，帮助信息为 'Batch size in every gpu'
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    # 添加一个命令行参数 -gpu 或 --gpu_ids，用于指定 GPU 的 ID，类型为字符串，默认值为 None
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    # 添加一个命令行参数 -d 或 --debug，如果指定了该参数，则将 debug 标志设置为 True
    parser.add_argument('-d', '--debug', action='store_true')
    # 添加一个命令行参数 -P 或 --port，用于指定端口号，类型为字符串，默认值为 '21012'
    parser.add_argument('-P', '--port', default='21023', type=str)

    ''' parser configs '''
    # 解析命令行参数，并将结果存储在 args 变量中
    args = parser.parse_args()
    # 使用 Praser 类解析 args，将结果存储在 opt 变量中
    opt = Praser.parse(args)
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])# 将 opt['gpu_ids'] 中的 GPU ID 转换为字符串，并使用逗号连接起来
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str# 设置环境变量 CUDA_VISIBLE_DEVICES，指定可见的 GPU
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))# 打印出设置的 CUDA_VISIBLE_DEVICES 环境变量

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''# 使用 DistributedDataParallel（DDP）和多进程进行多 GPU 训练
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:# 如果 opt['distributed'] 为 True，表示使用分布式训练
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()# 计算每个节点上的 GPU 数量
        opt['world_size'] = ngpus_per_node# 设置 opt['world_size'] 为 GPU 数量
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port # 设置初始化方法为 'tcp://127.0.0.1:' + args.port
        # 使用 multiprocessing 的 spawn 函数启动多个进程，每个进程执行 main_worker 函数，nprocs 为进程数量，args 为传递给 main_worker 函数的参数
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:# 否则，表示不使用分布式训练
        opt['world_size'] = 1 # 设置 opt['world_size'] 为 1
        main_worker(2, 1, opt)# 直接调用 main_worker 函数，传递参数 0、1 和 opt