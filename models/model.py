import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        # 初始化方法，接收网络模型、损失函数、采样数量、任务类型、优化器和EMA调度器等参数
        ''' must to init BaseModel with kwargs '''
        # 必须使用kwargs初始化基类
        super(Palette, self).__init__(**kwargs)
        # 调用父类的初始化方法

        ''' networks, dataloder, optimizers, losses, etc. '''
        # 设置网络模型、数据加载器、优化器、损失函数等
        self.loss_fn = losses[0]
        # 设置第一个损失函数
        self.netG = networks[0]
        # 设置第一个网络模型为生成器

        if ema_scheduler is not None:
            # 如果提供了EMA调度器
            self.ema_scheduler = ema_scheduler
            # 设置EMA调度器
            self.netG_EMA = copy.deepcopy(self.netG)
            # 创建生成器网络的深度副本，用于EMA模型
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
            # 初始化EMA类，设置衰减系数
        else:
            self.ema_scheduler = None
            # 未提供EMA调度器时设为None

        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        # 网络可以是列表，如果使用多GPU，必须通过set_device函数转换
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        # 将生成器网络移至指定设备，支持分布式训练
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
            # 将EMA版本的生成器网络也移至指定设备

        self.load_networks()
        # 加载预训练的网络权重

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        # 为生成器网络创建Adam优化器，只优化需要梯度的参数
        self.optimizers.append(self.optG)
        # 将优化器添加到优化器列表中
        self.resume_training()
        # 恢复训练状态（如果有检查点）

        if self.opt['distributed']:
            # 如果使用分布式训练
            self.netG.module.set_loss(self.loss_fn)
            # 在分布式包装器内的实际模型上设置损失函数
            self.netG.module.set_new_noise_schedule(phase=self.phase)
            # 设置新的噪声调度
        else:
            self.netG.set_loss(self.loss_fn)
            # 直接在模型上设置损失函数
            self.netG.set_new_noise_schedule(phase=self.phase)
            # 设置新的噪声调度

        ''' can rewrite in inherited class for more informations logging '''
        # 可以在继承类中重写以记录更多信息
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        # 初始化训练指标跟踪器，使用损失函数名称
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        # 初始化验证指标跟踪器，使用评估指标名称
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')
        # 初始化测试指标跟踪器，使用评估指标名称

        self.sample_num = sample_num
        # 设置样本数量
        self.task = task
        # 设置任务类型

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])

    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }
        if self.task in ['inpainting','uncropping']:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': (self.mask_image+1)/2,
            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            # ret_path.append('GT_{}'.format(self.path[idx]))
            # ret_result.append(self.gt_image[idx].detach().float().cpu())
            #
            # ret_path.append('Process_{}'.format(self.path[idx]))
            # ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())

            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())

        if self.task in ['inpainting','uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        # 将生成器网络设置为训练模式，启用dropout和batch normalization等训练时特性
        self.train_metrics.reset()
        # 重置训练指标计算器，准备记录新的训练轮次的指标
        for train_data in tqdm.tqdm(self.phase_loader):
            # 使用tqdm显示进度条，遍历训练数据加载器中的每个批次数据
            self.set_input(train_data)
            # 将训练数据加载到模型的输入变量中
            self.optG.zero_grad()
            # 清除生成器优化器中的梯度信息，准备计算新的梯度
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            # 前向传播：将真实图像、条件图像和掩码输入到生成器网络，计算损失
            loss.backward()
            # 反向传播：计算损失相对于网络参数的梯度
            self.optG.step()
            # 更新参数：使用优化器根据计算出的梯度更新生成器网络的参数
            self.iter += self.batch_size
            # 更新全局迭代计数，增加一个批次的样本数量
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            # 设置TensorBoard写入器的当前轮次和迭代，标记为训练阶段
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            # 更新训练指标，记录当前批次的损失值
            if self.iter % self.opt['train']['log_iter'] == 0:
                # 如果当前迭代次数达到记录日志的间隔
                for key, value in self.train_metrics.result().items():
                    # 遍历当前累积的所有训练指标
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    # 将指标记录到日志文件
                    self.writer.add_scalar(key, value)
                    # 将指标写入TensorBoard，用于可视化
                for key, value in self.get_current_visuals().items():
                    # 遍历当前生成的可视化结果（如图像）
                    self.writer.add_images(key, value)
                    # 将可视化结果写入TensorBoard
            if self.ema_scheduler is not None:
                # 如果配置了指数移动平均(EMA)调度器
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    # 如果当前迭代次数超过EMA开始的迭代次数，并且达到更新EMA的间隔
                    self.EMA.update_model_average(self.netG_EMA, self.netG)
                    # 更新EMA模型，保持一个平滑版本的生成器网络
        for scheduler in self.schedulers:
            # 遍历所有学习率调度器
            scheduler.step()
            # 更新学习率，通常基于训练轮次
        return self.train_metrics.result()
        # 返回当前轮次的训练指标结果

    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image,
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image,
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image,
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image,
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

        test_log = self.test_metrics.result()
        ''' save logged informations into log dict '''
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard '''
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        # 加载预训练模型和训练状态的方法，仅在GPU 0上执行
        """ save pretrained model and training state, which only do on GPU 0. """
        # 保存预训练模型和训练状态，仅在GPU 0上执行
        print('load_networks--------------')
        # 打印日志，标记加载网络开始
        if self.opt['distributed']:
            # 如果使用分布式训练
            netG_label = self.netG.module.__class__.__name__
            # 获取分布式包装器内实际模型的类名
        else:
            netG_label = self.netG.__class__.__name__
            # 直接获取模型的类名
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        # 加载生成器网络的预训练权重，不严格匹配参数
        if self.ema_scheduler is not None:
            # 如果启用了EMA（指数移动平均）
            self.load_network(network=self.netG_EMA, network_label=netG_label + '_ema', strict=False)
            # 加载EMA版本生成器网络的预训练权重，不严格匹配参数

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()