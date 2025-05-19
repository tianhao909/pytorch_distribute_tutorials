# 导入必要的模块
import os, sys  # 提供操作系统路径和系统特定功能的接口
import torch  # PyTorch 深度学习框架
import torch.nn.functional as F  # 包含常用的神经网络函数，例如损失函数等
from torch.utils.data import Dataset, DataLoader  # 数据加载工具

# 导入分布式训练相关模块
import torch.multiprocessing as mp  # 支持多进程并行处理
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器，用于在多个 GPU 上均匀分配数据
from torch.nn.parallel import DistributedDataParallel as DDP  # 用于分布式训练的模型包装器
from torch.distributed import init_process_group, destroy_process_group  # 初始化和销毁分布式进程组

# 定义分布式训练的初始化函数
def ddp_setup(rank, world_size):
    """
    Args:
        rank: 每个进程的唯一标识符（从0开始）
        world_size: 总进程数（通常等于 GPU 数量）
    """
    # 设置主节点地址和端口，用于进程间通信
    os.environ["MASTER_ADDR"] = "localhost"  # 主节点 IP 地址（本机）
    os.environ["MASTER_PORT"] = "12355"  # 主节点通信端口
    
    # 初始化进程组，使用 NCCL 后端进行 GPU 之间的高效通信
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    # 设置当前进程使用的 GPU 设备
    torch.cuda.set_device(rank)

# 定义 Trainer 类，用于封装训练流程
class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 train_dataloader: DataLoader, 
                 optimizer: torch.optim.Optimizer, 
                 gpu_id: int) -> None:
        # 保存当前 GPU ID（即 rank）
        self.gpu_id = gpu_id
        
        # 将模型移动到当前 GPU 上
        self.model = model.to(gpu_id)
        
        # 保存训练用的数据加载器
        self.train_dataloader = train_dataloader
        
        # 保存优化器
        self.optimizer = optimizer
        
        # 使用 DistributedDataParallel 包装模型，启用分布式训练
        # device_ids 指定当前进程使用的 GPU 设备
        self.model = DDP(model, device_ids=[gpu_id])
    
    # 执行单个 batch 的训练步骤
    def _run_batch(self, xs, ys):
        # 清空上一次的梯度
        self.optimizer.zero_grad()
        
        # 前向传播：得到模型输出
        output = self.model(xs)
        
        # 计算损失函数（交叉熵损失）
        loss = F.cross_entropy(output, ys)
        
        # 反向传播：计算梯度
        loss.backward()
        
        # 更新参数
        self.optimizer.step()
    
    # 执行一个完整的 epoch（遍历整个训练集一次）
    def _run_epoch(self, epoch):
        # 获取 batch 大小（用于打印信息）
        batch_size = len(next(iter(self.train_dataloader))[0])
        
        # 打印当前 epoch、batch 大小和总步数（每个 epoch 的 batch 数量）
        print(f'[GPU: {self.gpu_id}] Epoch: {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_dataloader)}')
        
        # 在每个 epoch 开始时设置 sampler 的 epoch，确保数据打乱方式不同
        self.train_dataloader.sampler.set_epoch(epoch)
        
        # 遍历 dataloader 中的每一个 batch
        for xs, ys in self.train_dataloader:
            # 将输入数据移动到当前进程的 GPU 上
            xs = xs.to(self.gpu_id)
            
            # 将标签数据移动到当前进程的 GPU 上
            ys = ys.to(self.gpu_id)
            
            # 执行单个 batch 的训练步骤
            self._run_batch(xs, ys)
    
    # 对每个 epoch 进行训练
    def train(self, max_epoch: int):
        # 对每个 epoch 进行训练
        for epoch in range(max_epoch):
            # 执行一个完整的训练轮次（一个 epoch）
            self._run_epoch(epoch)

# 自定义数据集类，继承自 Dataset
class MyTrainDataset(Dataset):
    def __init__(self, size):
        # 保存数据集大小
        self.size = size
        
        # 生成随机输入和输出的数据对，作为数据集内容
        # 每个样本是一个包含输入（20维）和目标（1维）的元组
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    # 返回数据集的大小（样本数量）
    def __len__(self):
        return self.size
    
    # 根据索引获取单个数据样本
    def __getitem__(self, index):
        return self.data[index]

# 主函数，执行分布式训练流程
def main(rank: int, world_size: int, max_epochs: int, batch_size: int):
    # 初始化分布式训练环境
    ddp_setup(rank, world_size)
    
    # 创建自定义数据集实例，大小为 2048
    train_dataset = MyTrainDataset(2048)
    
    # 创建 DataLoader 加载数据，使用 DistributedSampler 确保不同 GPU 之间数据不重叠
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                pin_memory=True,  # 启用内存固定，提高 GPU 数据传输效率
                                shuffle=False,  # 不打乱数据，由 DistributedSampler 控制
                                sampler=DistributedSampler(train_dataset))  # 使用分布式采样器
    
    # 定义一个简单的线性模型：发生异常，可以输入更多信息再让我来回答或重试