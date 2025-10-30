import os
from abc import abstractmethod
from functools import partial
import collections

import torch
import torch.nn as nn


import core.util as Util
CustomResult = collections.namedtuple('CustomResult', 'name result')

class BaseModel():
    def __init__(self, opt, phase_loader, val_loader, metrics, logger, writer):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        """ 用基本输入初始化模型，这些输入来自继承类的 __init__(**kwargs) 函数 """
        """ 
                用以下基本输入初始化模型，这些输入来自继承类的 __init__(**kwargs) 函数：
                - opt：模型的配置选项。
                - phase_loader：当前阶段（如训练阶段）的数据加载器。
                - val_loader：验证阶段的数据加载器。
                - metrics：用于评估模型性能的指标。
                - logger：用于记录日志的对象。
                - writer：用于将数据写入到 TensorBoard 和结果文件的对象。
        """
        self.opt = opt# 保存配置选项
        self.phase = opt['phase']# 获取当前阶段
        self.set_device = partial(Util.set_device, rank=opt['global_rank']) # 设置设备的部分函数

        ''' optimizers and schedulers '''
        ''' 优化器和调度器 '''
        self.schedulers = [] # 初始化调度器列表为空
        self.optimizers = [] # 初始化优化器列表为空

        ''' process record '''
        ''' 进程记录 '''
        self.batch_size = self.opt['datasets'][self.phase]['dataloader']['args']['batch_size']# 获取当前阶段数据加载器的批大小
        self.epoch = 0 # 初始化 epoch 为 0
        self.iter = 0 # 初始化迭代次数为 0

        self.phase_loader = phase_loader # 保存阶段数据加载器
        self.val_loader = val_loader # 保存验证数据加载器
        self.metrics = metrics # 保存指标

        ''' logger to log file, which only work on GPU 0. writer to tensorboard and result file '''
        ''' 记录到日志文件的日志记录器，仅在 GPU 0 上工作（可在json文件中设置gpu_ids）。记录到 TensorBoard 和结果文件的写入器 '''
        self.logger = logger# 保存日志记录器
        self.writer = writer# 保存写入器
        self.results_dict = CustomResult([],[]) # {"name":[], "result":[]} # 创建一个 CustomResult 对象，用于存储结果 {"name":[], "result":[]}

    def train(self):
        """ 5.1、训练函数 """
        # 当 epoch 未超过训练的最大 epoch 且迭代次数未超过最大迭代次数时
        while self.epoch <= self.opt['train']['n_epoch'] and self.iter <= self.opt['train']['n_iter']:
            self.epoch += 1 # epoch 加 1
            if self.opt['distributed']:# 如果是分布式训练
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                ''' 为这个采样器设置 epoch。当 :attr:`shuffle=True` 时，这确保所有副本在每个epoch 使用不同的随机顺序 '''
                self.phase_loader.sampler.set_epoch(self.epoch) # 为阶段数据加载器的采样器设置 epoch

            train_log = self.train_step()# 5.1.2：执行训练步骤并获取训练日志，跳进model.py中def train_step(self)

            ''' save logged informations into log dict '''
            ''' 将记录的信息保存到日志字典中 '''
            train_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard '''
            ''' 将记录的信息打印到屏幕和 TensorBoard 上 '''
            for key, value in train_log.items():# 遍历训练日志中的每个键值对
                self.logger.info('{:5s}: {}\t'.format(str(key), value)) # 记录信息到日志
            
            if self.epoch % self.opt['train']['save_checkpoint_epoch'] == 0:# 如果当前 epoch 是保存检查点的 epoch
                self.logger.info('Saving the self at the end of epoch {:.0f}'.format(self.epoch))# 记录保存模型的信息
                self.save_everything() # 保存所有相关信息

            if self.epoch % self.opt['train']['val_epoch'] == 0:# 如果当前 epoch 是验证的 epoch
                self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_loader is None:# 如果验证数据加载器为空
                    self.logger.warning('Validation stop where dataloader is None, Skip it.')# 记录警告信息
                else:
                    val_log = self.val_step()# 执行验证步骤并获取验证日志
                    for key, value in val_log.items():# 遍历验证日志中的每个键值对
                        self.logger.info('{:5s}: {}\t'.format(str(key), value))
                self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        self.logger.info('Number of Epochs has reached the limit, End.') # 当 epoch 达到限制时，记录结束信息

    def test(self):
        pass

    # 定义了一个抽象方法train_step，该方法在当前类中没有具体的实现
    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your networks.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError('You must specify how to do validation on your networks.')

    def test_step(self):
        pass
    
    def print_network(self, network):
        """ print network structure, only work on GPU 0 """
        # if self.opt['global_rank'] !=0:
        #     return
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        
        s, n = str(network), sum(map(lambda x: x.numel(), network.parameters()))
        net_struc_str = '{}'.format(network.__class__.__name__)
        self.logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, network, network_label):
        """ save network structure, only work on GPU 0 保存网络结构，仅在 GPU 0 上工作"""
        # if self.opt['global_rank'] !=2:#如果不是 GPU 0
        #     return
        save_filename = '{}_{}.pth'.format(self.epoch, network_label)# 确定保存的文件名，格式为“epoch_网络标签.pth”
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, strict=True):
        if self.opt['path']['resume_state'] is None:
            return 
        self.logger.info('Beign loading pretrained model [{:s}] ...'.format(network_label))

        model_path = "{}_{}.pth".format(self. opt['path']['resume_state'], network_label)
        
        if not os.path.exists(model_path):
            self.logger.warning('Pretrained model in [{:s}] is not existed, Skip it'.format(model_path))
            return

        self.logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        network.load_state_dict(torch.load(model_path, map_location = lambda storage, loc: Util.set_device(storage)), strict=strict)

        total_params = sum(p.numel() for p in network.parameters())  # 统计所有参数
        print(f"Total parameters-----: {total_params:.2f}")  # 62641347
        print(f"Total parameters-----: {total_params / 1e9:.2f}B (十亿)")  # 0.06B

    def save_training_state(self):
        """ saves training state during training, only work on GPU 0 """
        """ 保存训练状态的函数，仅在 GPU 0 上工作"""
        # if self.opt['global_rank'] !=0:
        #     return
        # 检查 self.optimizers 和 self.schedulers 是否为列表类型
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        # 创建一个状态字典，包含当前的 epoch、iter 以及 schedulers 和 optimizers 的状态字典
        state = {'epoch': self.epoch, 'iter': self.iter, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:  # 将每个 scheduler 的状态字典添加到 state 中的'schedulers'列表中
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers: # 将每个 optimizer 的状态字典添加到 state 中的'optimizers'列表中
            state['optimizers'].append(o.state_dict())
        # 确定保存文件名，格式为'{当前 epoch}.state'
        save_filename = '{}.state'.format(self.epoch)
        # 构建保存路径，位于 self.opt['path']['checkpoint'] 目录下，文件名是 save_filename
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        # 使用 torch.save 函数将状态保存到指定路径
        torch.save(state, save_path)

    def resume_training(self):
        """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
        if self.phase!='train' or self. opt['path']['resume_state'] is None:
            return
        self.logger.info('Beign loading training states'.format())
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        
        state_path = "{}.state".format(self. opt['path']['resume_state'])
        
        if not os.path.exists(state_path):
            self.logger.warning('Training state in [{:s}] is not existed, Skip it'.format(state_path))
            return

        self.logger.info('Loading training state for [{:s}] ...'.format(state_path))
        resume_state = torch.load(state_path, map_location = lambda storage, loc: self.set_device(storage))
        
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers {} != {}'.format(len(resume_optimizers), len(self.optimizers))
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers {} != {}'.format(len(resume_schedulers), len(self.schedulers))
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        self.epoch = resume_state['epoch']
        self.iter = resume_state['iter']

    def load_everything(self):
        pass 
    
    @abstractmethod
    def save_everything(self):
        raise NotImplementedError('You must specify how to save your networks, optimizers and schedulers.')

    def setup_visualization(self):
        """设置可视化基础功能"""
        if not hasattr(self, 'viz_initialized'):
            self.viz_initialized = True
            self.visualization_config = self.opt.get('visualization', {})

            # 只在主进程进行可视化
            if self.opt['global_rank'] == 0:
                self.init_visualization_tools()