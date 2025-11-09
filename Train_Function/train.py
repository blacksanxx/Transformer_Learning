import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from collections import Counter, defaultdict
from tqdm.auto import tqdm
import math
import os
import xml.etree.ElementTree as ET  # 用于解析 XML 文件
import matplotlib.pyplot as plt

from src.model.Transformer import Transformer 

"""
2.0版本
主要改动是：
    模型保存与加载功能
    训练曲线绘制
"""


# ========== 超参数配置区（集中管理） ==========
DATA_DIR = "./datasets/en-zh"

model_save_path = "./module/BaseLine_model.pth"  # --- 添加模型保存路径 ---
os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # 确保目录存在

MAX_LENGTH = 128
BATCH_SIZE = 32
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
D_FF = 512
DROPOUT = 0.1
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4 
SRC_MIN_FREQ = 2
TGT_MIN_FREQ = 2
VAL_SPLIT_RATIO = 0.1  # 仅在无法加载验证集时使用
# ==========================================

# 1. 读取本地文件
def read_texts_from_local_dir(data_dir, split='train'):
    """
    从本地目录读取源语言和目标语言的文本。
    假设文件名格式为:
    - 纯文本: {split}.tags.en-zh.en, {split}.tags.en-zh.zh
    - XML: IWSLT17.TED.{split}.en-zh.en.xml, IWSLT17.TED.{split}.en-zh.zh.xml
    """
    src_texts = []
    tgt_texts = []

    # 尝试 XML 格式文件
    src_xml_file = os.path.join(data_dir, f"IWSLT17.TED.{split}.en-zh.en.xml")
    tgt_xml_file = os.path.join(data_dir, f"IWSLT17.TED.{split}.en-zh.zh.xml")
    
    if os.path.exists(src_xml_file) and os.path.exists(tgt_xml_file):
        print(f"Reading XML files for {split} split: {src_xml_file}, {tgt_xml_file}")
        try:
            src_tree = ET.parse(src_xml_file)
            tgt_tree = ET.parse(tgt_xml_file)
            src_root = src_tree.getroot()
            tgt_root = tgt_tree.getroot()

            # 查找所有 <seg> 标签
            src_segs = src_root.findall('.//seg')
            tgt_segs = tgt_root.findall('.//seg')

            if len(src_segs) != len(tgt_segs):
                print(f"Warning: Mismatched segment counts in {src_xml_file} and {tgt_xml_file}")
                return [], []

            for s, t in zip(src_segs, tgt_segs):
                src_texts.append(s.text.strip() if s.text else "")
                tgt_texts.append(t.text.strip() if t.text else "")
        except ET.ParseError as e:
            print(f"XML parsing failed for {src_xml_file} or {tgt_xml_file}: {e}")
            return [], []

    # 如果 XML 文件不存在，尝试纯文本文件
    elif not src_texts:  # 如果 XML 读取未成功
        src_txt_file = os.path.join(data_dir, f"{split}.tags.en-zh.en")
        tgt_txt_file = os.path.join(data_dir, f"{split}.tags.en-zh.zh")
        
        if os.path.exists(src_txt_file) and os.path.exists(tgt_txt_file):
            print(f"Reading text files for {split} split: {src_txt_file}, {tgt_txt_file}")
            with open(src_txt_file, 'r', encoding='utf-8') as f_src, \
                 open(tgt_txt_file, 'r', encoding='utf-8') as f_tgt:
                src_lines = f_src.readlines()
                tgt_lines = f_tgt.readlines()

                if len(src_lines) != len(tgt_lines):
                    print(f"Warning: Mismatched line counts in {src_txt_file} and {tgt_txt_file}")
                    return [], []

                src_texts = [line.strip() for line in src_lines]
                tgt_texts = [line.strip() for line in tgt_lines]
        else:
            print(f"Warning: Could not find XML or text files for {split} split in {data_dir}")
            return [], []

    return src_texts, tgt_texts

# 2. 构建词汇表的辅助函数
def build_vocab(texts, min_freq=2, is_tgt=False):
    """
    从原始文本列表中构建词汇表
    is_tgt: True 表示目标语言（如中文），按字符处理；False 表示源语言（如英文），按空格分词。
    返回: vocab_dict (普通字典), unk_idx (未知词索引)
    """
    counter = Counter()
    for text in texts:
        if is_tgt:  # 假设目标语言是中文，按字符分割
            tokens = list(text)
        else:       # 假设源语言是英文，按空格分割
            tokens = text.lower().split()
        counter.update(tokens)

    # 创建普通字典，先添加特殊token
    vocab = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3
    }
    idx = 4  # 从 4 开始分配新词索引

    # 添加高频词
    for token, freq in counter.items():
        if freq >= min_freq:
            if token not in vocab:  # 确保特殊token ID 不被覆盖
                vocab[token] = idx
                idx += 1

    # 确保 unk_idx 是正确的
    unk_idx = vocab['<unk>']
    return vocab, unk_idx  # 返回 vocab 字典和 unk 索引

# 3. 自定义数据集类 (处理原始文本 -> token IDs)
class IWSLT2017Dataset(Dataset):
    """
    IWSLT 2017 数据集的 PyTorch Dataset 包装器 (本地文件版)
    接收原始文本列表和词汇表，返回 token IDs
    """
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_length=128):
        if len(src_texts) != len(tgt_texts):
            raise ValueError("Source and target text lists must have the same length.")
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        # 使用普通字典和 unk 索引
        self.src_vocab = src_vocab[0]  # vocab_dict
        self.tgt_vocab = tgt_vocab[0]  # vocab_dict
        self.src_unk_idx = src_vocab[1]  # unk_idx
        self.tgt_unk_idx = tgt_vocab[1]  # unk_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # 简单分词 (英文按空格，中文按字符)
        src_tokens = src_text.lower().split()[:self.max_length - 2]  # 预留 <sos> 和 <eos>
        tgt_tokens = list(tgt_text)[:self.max_length - 2]  # 中文按字符分割

        # 转换为ID (未知词用<unk>的索引)
        src_ids = [self.src_vocab.get(token, self.src_unk_idx) for token in src_tokens]
        tgt_ids = [self.tgt_vocab.get(token, self.tgt_unk_idx) for token in tgt_tokens]

        # 添加特殊token
        src_ids = [self.src_vocab['<sos>']] + src_ids + [self.src_vocab['<eos>']]
        tgt_ids = [self.tgt_vocab['<sos>']] + tgt_ids + [self.tgt_vocab['<eos>']]

        # 填充到固定长度
        src_ids += [self.src_vocab['<pad>']] * (self.max_length - len(src_ids))
        tgt_ids += [self.tgt_vocab['<pad>']] * (self.max_length - len(tgt_ids))

        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)

        # 分离目标输入和标签
        tgt_input_ids = tgt_ids[:-1]  # <sos> w1 w2 ... wN (移除最后一个 <eos>)
        tgt_label_ids = tgt_ids[1:]   # w1 w2 ... wN <eos> (移除第一个 <sos>)

        # 创建掩码 (1 for real tokens, 0 for padding)-------------
        src_mask = (src_ids != self.src_vocab['<pad>']).long()
        tgt_input_mask = (tgt_input_ids != self.tgt_vocab['<pad>']).long()

        # 将填充部分的标签设置为 -100
        tgt_label_ids = tgt_label_ids.clone()
        tgt_label_ids[tgt_input_mask == 0] = -100  # 使用 tgt_input_mask 判断填充位置

        return {
            'src_ids': src_ids,
            'src_mask': src_mask,
            'tgt_input_ids': tgt_input_ids,
            'tgt_input_mask': tgt_input_mask,
            'tgt_label_ids': tgt_label_ids
        }

# 4. 训练函数
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        src_ids = batch['src_ids'].to(device)
        tgt_input_ids = batch['tgt_input_ids'].to(device)
        tgt_label_ids = batch['tgt_label_ids'].to(device)
        src_mask = batch['src_mask'].to(device)

        tgt_seq_len = tgt_input_ids.size(1)
        tgt_mask = _generate_future_mask(tgt_seq_len).to(device)

        optimizer.zero_grad()
        outputs = model(src_ids, tgt_input_ids, src_mask=src_mask, tgt_mask=tgt_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_label_ids.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# 5. 评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src_ids = batch['src_ids'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device)
            tgt_label_ids = batch['tgt_label_ids'].to(device)
            src_mask = batch['src_mask'].to(device)

            tgt_seq_len = tgt_input_ids.size(1)
            tgt_mask = _generate_future_mask(tgt_seq_len).to(device)

            outputs = model(src_ids, tgt_input_ids, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_label_ids.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def _generate_future_mask(sz):
    """生成 future mask（下三角为 True）"""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask  # [sz, sz]

# 6. 主函数
def main():
    # --- 使用集中定义的超参数 ---
    data_dir = DATA_DIR
    max_length = MAX_LENGTH
    batch_size = BATCH_SIZE
    d_model = D_MODEL
    n_layers = N_LAYERS
    n_heads = N_HEADS
    d_ff = D_FF
    dropout = DROPOUT
    num_epochs = NUM_EPOCHS
    learning_rate = LEARNING_RATE
    src_min_freq = SRC_MIN_FREQ
    tgt_min_freq = TGT_MIN_FREQ
    val_split_ratio = VAL_SPLIT_RATIO

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 读取本地文件 ---
    print("Reading training data...")
    train_src_texts, train_tgt_texts = read_texts_from_local_dir(data_dir, split='train')
    print(f"Loaded {len(train_src_texts)} training sentence pairs.")

    print("Reading validation data...")
    val_src_texts, val_tgt_texts = read_texts_from_local_dir(data_dir, split='dev2010')
    if not val_src_texts:
        print("Warning: Could not load validation data from dev2010. Using a split from training data instead.")
        split_idx = int((1 - val_split_ratio) * len(train_src_texts))
        val_src_texts = train_src_texts[split_idx:]
        val_tgt_texts = train_tgt_texts[split_idx:]
        train_src_texts = train_src_texts[:split_idx]
        train_tgt_texts = train_tgt_texts[:split_idx]

    print(f"Loaded {len(val_src_texts)} validation sentence pairs.")

    # --- 构建词汇表 ---
    print("Building source vocabulary...")
    src_vocab_tuple = build_vocab(train_src_texts, min_freq=src_min_freq, is_tgt=False)
    print("Building target vocabulary...")
    tgt_vocab_tuple = build_vocab(train_tgt_texts, min_freq=tgt_min_freq, is_tgt=True)
    src_vocab_size = len(src_vocab_tuple[0])
    tgt_vocab_size = len(tgt_vocab_tuple[0])
    print(f"Source vocab size: {src_vocab_size}, Target vocab size: {tgt_vocab_size}")

    # --- 创建数据集实例 ---
    train_dataset = IWSLT2017Dataset(train_src_texts, train_tgt_texts, src_vocab_tuple, tgt_vocab_tuple, max_length)
    val_dataset = IWSLT2017Dataset(val_src_texts, val_tgt_texts, src_vocab_tuple, tgt_vocab_tuple, max_length)

    # --- 创建 DataLoader ---
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 创建模型、优化器、损失函数 ---
    print("Initializing model...")
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout).to(device)
    print("Model initialized successfully.")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)


    """# --- 添加模型加载逻辑 ---"""
    if os.path.exists(model_save_path):
        print(f"Loading model from {model_save_path}...")
        checkpoint = torch.load(model_save_path, map_location=device) # Load on specified device
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) # Resume from next epoch, default 0 if key not found
        best_val_loss = checkpoint.get('val_loss', float('inf')) # Get best val loss, default inf if key not found
        print(f"Loaded checkpoint from epoch {start_epoch - 1}. Resuming training.")
    else:
        print("No checkpoint found, starting training from scratch.")
        start_epoch = 0
        best_val_loss = float('inf')
        

    
     # --- 初始化记录损失的列表 ---
    train_losses = []
    val_losses = []
    epochs_list = []
    # --- 训练循环 ---
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # --- 记录损失 ---
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs_list.append(epoch + 1) # 记录当前epoch编号

        # --- 添加模型保存逻辑 ---
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model state, optimizer state, epoch, and best val loss
            torch.save({
                'epoch': epoch + 1, # Save the next epoch number
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                # Optionally save vocabularies if needed for inference
                'src_vocab': src_vocab_tuple[0],
                'tgt_vocab': tgt_vocab_tuple[0],
            }, model_save_path)
            print(f"New best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")
        else:
            print(f"Model not saved this epoch. Best Val Loss: {best_val_loss:.4f}")

    print("Training finished.")

    # --- 绘制并保存训练曲线 ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs_list, val_losses, label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 保存图片到本地
    plot_save_path = "./training_curves.png"
    plt.savefig(plot_save_path)
    plt.show()
    print(f"Training curves saved to {plot_save_path}")

if __name__ == "__main__":
    main()