# CNN_cifar10

使用 pytorch, cnn 训练 cifar10 数据集

配置信息写在 config.py

请在 config 中选择网络

models 用于存放各种 model

checkpoint 用于持久化各模型的训练结果，

- 可通过调整 config 里的 save_flag 决定是否持久化
- 通过调整 resume 可决定是否读取 checkpoint 继续训练
- 第一次训练某个模型请将 resume 设置为 False

config 中可以也更改持久化目录名与统计画图存放路径
