# 复现说明

  

本实验基于 Python 3.9.21 环境，在配备 NVIDIA GPU的机器上完成训练与推理。以下是复现本报告所有结果所需的完整说明。

  

**依赖库与环境配置**  

主要依赖库包括：  

- PyTorch ≥ 1.12（需支持 CUDA）  

- NLTK（用于 BLEU 计算，首次运行需执行 `nltk.download('punkt')`）  

- tqdm（进度条）  

- matplotlib（绘图）  

- 标准库：xml.etree.ElementTree（解析 IWSLT XML 文件）、collections、math、os 等  

  

可通过以下命令创建并激活环境（以 Conda 为例）：

```bash

conda create -n transformer python=3.9.21

conda activate transformer

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install nltk tqdm matplotlib

```

  

**代码仓库结构**  

项目根目录结构如下（已省略 .git 内部细节和 __pycache__）：

```

D:.

├── datasets/                 # 数据集目录

│   └── en-zh/               # IWSLT2017 英中数据（含 XML 和 .tags 文件）

├── module/                   # 模型权重保存目录（如 BaseLine_model_Picture_BLEU.pth）

├── result_picture/           # 训练曲线等结果图像保存目录

├── src/

│   ├── model/                # Transformer 模型定义（Transformer.py）

│   └── component                 # 组件相关代码

├── Train_Function/           # 训练函数（其他版本的训练函数，如果要用移动到当前工作目录下）

└── train_with_BLEU.py      # 主训练脚本（最终使用的训练版本，主要是包含BLEU评估）
└── inference.py             # 推理版本
└── README.md

```

  

**数据准备**  

请将 IWSLT2017 英中数据（包括 `IWSLT17.TED.train.en-zh.en.xml`、`IWSLT17.TED.dev2010.en-zh.zh.xml` 等）放置于 `datasets/en-zh/` 目录下。若使用 .tags 文件，需确保文件名符合代码中 `read_texts_from_local_dir` 的命名规则。
(https://huggingface.co/datasets/IWSLT/iwslt2017/tree/main/data/2017-01-trnted/texts/en/zh)
  

**运行命令**  

直接运行主脚本即可启动训练与评估流程：

```bash

python train_with_BLEU.py

```

程序将自动：

- 读取数据并构建字符级（中文）/空格分词（英文）词表；

- 初始化 4 层 Transformer 模型（D_MODEL=256, N_HEADS=8, D_FF=512）；

- 在 GPU 上训练 30 轮；

- 每轮在验证集上计算 Loss、ACC、PPL 和 BLEU-4（字符级）；

- 保存最佳模型至 `module/BaseLine_model_Picture_BLEU.pth`；

- 训练结束后生成并保存四指标曲线图至 `result_picture/`。

  

**预期运行时长与硬件**  

在笔记本的  NVIDIA RTX 3060 （6 G）上，每轮训练约需 8–10 分钟，30 轮总计约 4–5 小时。BLEU 评估阶段（每轮采样 1000 句）会额外增加约 4–5 分钟/轮。若使用 CPU，训练时间将显著延长，不建议。

  

**注意事项**  

- 若 `dev2010` 验证集不存在，程序会自动从训练集中划分 10% 作为验证集。  

- 首次运行需取消 `nltk.download('punkt')` 注释以下载分词资源。  

- 模型、优化器状态及词表均保存在 checkpoint 中，支持中断后恢复训练。

  
