import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        # 1、module_name:'guided_diffusion'
        # 2、unet:{'in_channel': 6, 'out_channel': 3, 'inner_channel': 64, 'channel_mults': [1, 2, 4, 8], 'attn_res': [16],
        #          'num_head_channels': 32, 'res_blocks': 2, 'dropout': 0.2, 'image_size': 256}
        # 3、kwargs={'init_type': 'kaiming'}
        # 4、beta_schedule={'train': {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-06, 'linear_end': 0.01},
        #                   'test': {'schedule': 'linear', 'n_timestep': 1000, 'linear_start': 0.0001, 'linear_end': 0.09}}
        super(Network, self).__init__(**kwargs) # 调用父类的构造函数
        if module_name == 'sr3':# 如果模块名为'sr3'
            from .sr3_modules.unet import UNet # 从当前包的.sr3_modules.unet模块中导入UNet类
        elif module_name == 'guided_diffusion': # 如果模块名为'guided_diffusion'
            from .guided_diffusion_modules.unet import UNet # 从当前包的.guided_diffusion_modules.unet模块中导入UNet类

        # 创建去噪函数，传入unet参数
        self.denoise_fn = UNet(**unet)
        # 设置beta_schedule属性
        self.beta_schedule = beta_schedule#{'train': {'schedule': 'linear', 'n_timestep': 1000, 'linear_start': 1e-06, 'linear_end': 0.01}, 'test': {'schedule': 'linear', 'n_timestep': 500, 'linear_start': 0.0001, 'linear_end': 0.09}}

    # 定义set_loss方法，用于设置损失函数
    def set_loss(self, loss_fn):
        # 将传入的损失函数赋值给self.loss_fn属性
        self.loss_fn = loss_fn

    # 定义set_new_noise_schedule方法，用于设置新的噪声调度
    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        # 创建一个偏函数，用于将数据转换为指定类型和设备的PyTorch张量
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        # 根据beta_schedule的phase阶段调用make_beta_schedule函数生成beta值
        betas = make_beta_schedule(**self.beta_schedule[phase])
        # 如果betas是PyTorch张量，则将其分离并转换为Numpy数组
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        # 计算alphas，即1减去betas
        alphas = 1. - betas

        # 获取beta值的形状，即时间步数
        timesteps, = betas.shape
        # 将时间步数转换为整数并赋值给self.num_timesteps属性
        self.num_timesteps = int(timesteps)

        # 计算gammas，即alphas的累积乘积
        gammas = np.cumprod(alphas, axis=0)
        # 计算gammas_prev，即在gammas前面添加1
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 为扩散q(x_t | x_{t-1})和其他计算注册缓冲区
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 为后验q(x_{t-1} | x_t, x_0)进行计算
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # 由于扩散链开始时后验方差为0，对对数计算进行裁剪
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    # 定义predict_start_from_noise方法，用于从噪声预测起始值
    def predict_start_from_noise(self, y_t, t, noise):
        # 根据给定的公式，使用extract函数提取相关系数并计算返回值
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        # 根据给定的公式，使用extract函数提取相关系数并计算后验均值和对数方差
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        # 提取噪声水平
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        # 根据噪声水平预测起始值
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        # 计算模型均值和对数方差
        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    # 定义q_sample方法，用于对给定的初始值进行采样
    def q_sample(self, y_0, sample_gammas, noise=None):
        # 如果噪声未给定，则使用默认的随机噪声
        noise = default(noise, lambda: torch.randn_like(y_0))
        # 根据给定的公式进行采样并返回结果
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    # 定义p_sample方法，用于无梯度地进行采样
    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        # 计算模型均值和对数方差
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        # 根据条件生成噪声
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        # 根据给定的公式进行采样并返回结果
        return model_mean + noise * (0.5 * model_log_variance).exp()

    # 定义restoration方法，用于恢复图像
    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        b, *_ = y_cond.shape        # 获取条件图像的批次大小等维度信息

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'# 确保时间步数大于采样数

        sample_inter = (self.num_timesteps//sample_num)# 计算采样间隔
        # print('sample_num:', sample_num) # 8
        # print('self.num_timesteps:', self.num_timesteps) # 1000
        # print('sample_inter:',sample_inter) # 125
        y_t = default(y_t, lambda: torch.randn_like(y_cond))# 如果y_t未给定，则使用默认的随机噪声
        ret_arr = y_t# 初始化返回数组为y_t
        # 从最后一个时间步开始，反向迭代时间步
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)# 创建一个全为当前时间步的张量
            y_t = self.p_sample(y_t, t, y_cond=y_cond)# 进行采样操作
            if mask is not None:# 如果有掩码，根据掩码更新y_t
                y_t = y_0*(1.-mask) + mask*y_t
            #if i % sample_inter == 0:# 如果当前时间步是采样间隔的倍数，将y_t连接到返回数组
            # if i in [999,998,997,996,995,994,993,992,991,990,989,9,8,7,6,5,4,3,2,1,0]: #一共1000次去噪，将最后10次去噪的结果保存出来
            #     print('i={}'.format(i))
            #     ret_arr = torch.cat([ret_arr, y_t], dim=0)
            ret_arr = torch.cat([ret_arr, y_t], dim=0)
        print('ret_arr.shape:',ret_arr.shape) # torch.Size([1001, 3, 192, 256])
        return y_t, ret_arr

    # 定义forward方法，用于前向传播
    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas) 从p(gammas)中采样
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()  # 在1到时间步数之间随机选择时间步
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))# 提取相关的gamma值
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        # 根据公式计算采样的gamma值
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)
        # 如果噪声未给定，则使用默认的随机噪声
        noise = default(noise, lambda: torch.randn_like(y_0))
        # 对初始值进行采样得到噪声图像
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:# 如果有掩码，计算噪声预测值并计算损失
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)

            loss = self.loss_fn(mask*noise, mask*noise_hat)
        else: # 如果没有掩码，计算噪声预测值并计算损失
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
        # print("loss ======",loss)
        return loss


# gaussian diffusion trainer class
def exists(x):# 定义exists函数，用于判断对象是否存在
    return x is not None

def default(val, d):# 定义default函数，用于提供默认值
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):# 定义extract函数，用于从张量中提取指定索引的值
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function 定义_warmup_beta函数，用于生成带有热身阶段的beta值
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

# 定义make_beta_schedule函数，用于根据不同的调度方式生成beta值
def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


