import torch
import os
import random
from collections import defaultdict
import xml.etree.ElementTree as ET

from src.model.Transformer import Transformer
from Train_Function.train import read_texts_from_local_dir, build_vocab, IWSLT2017Dataset  # 假设训练代码保存为 train_code.py

# ========== 配置 ==========
DATA_DIR = "./datasets/en-zh"
#model_save_path = "./module/All_transformer_model.pth"
#model_save_path = "./module/BaseLine_model_Picture.pth"
model_save_path = "./module/BaseLine_model_Picture_BLEU.pth"
MAX_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 辅助函数：ID 转文本 ==========
def ids_to_text(ids, vocab, idx_to_token_map):
    """将 ID 序列转为可读文本"""
    tokens = []
    for idx in ids:
        if idx in [0, 1, 2, 3]:  # <pad>, <sos>, <eos>, <unk>
            continue
        tokens.append(idx_to_token_map.get(idx, '<unk>'))
    return ''.join(tokens)  # 中文无需空格

# ========== 自回归推理（不依赖 model.generate）==========
def greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx, pad_idx, tgt_vocab_size):
    model.eval()
    with torch.no_grad():
        # 初始化目标序列
        ys = torch.full((1, 1), sos_idx, dtype=torch.long, device=DEVICE)

        for _ in range(max_len - 1):
            # 构建 target mask (future mask)
            tgt_len = ys.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(DEVICE)

            # 前向传播
            logits = model(src, ys, src_mask=src_mask, tgt_mask=tgt_mask)  # [1, tgt_len, tgt_vocab]
            next_token_logits = logits[:, -1, :]  # [1, tgt_vocab]
            next_token = next_token_logits.argmax(dim=-1)  # [1]

            ys = torch.cat([ys, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == eos_idx:
                break

        return ys.squeeze(0)

# ========== 主推理逻辑 ==========
def main():
    # 1. 加载训练数据（用于展示例子）
    print("Loading training data for inference examples...")
    train_src_texts, train_tgt_texts = read_texts_from_local_dir(DATA_DIR, split='train')
    print(f"Loaded {len(train_src_texts)} training pairs.")

    # 2. 随机选5个例子
    indices = random.sample(range(len(train_src_texts)), 5)
    selected_src = [train_src_texts[i] for i in indices]
    selected_tgt = [train_tgt_texts[i] for i in indices]

    # 3. 从 checkpoint 加载模型和词汇表
    print("Loading model and vocabularies...")
    checkpoint = torch.load(model_save_path, map_location=DEVICE)
    
    src_vocab = checkpoint['src_vocab']  # dict: token -> id
    tgt_vocab = checkpoint['tgt_vocab']  # dict: token -> id

    # 构建反向映射: id -> token
    tgt_idx_to_token = {idx: token for token, idx in tgt_vocab.items()}

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    model = Transformer(
        len(src_vocab),      # src_vocab_size
        len(tgt_vocab),      # tgt_vocab_size
        256,                 # d_model
        4,                   # n_layers
        8,                   # n_heads
        512,                 # d_ff
        0.1                  # dropout
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # 4. 推理并展示结果
    print("\n" + "="*80)
    print("INFERENCE RESULTS (English -> Chinese)")
    print("="*80)

    for i, (src_text, tgt_text) in enumerate(zip(selected_src, selected_tgt)):
        print(f"\nExample {i+1}:")
        print(f"Source (EN): {src_text}")
        print(f"Reference (ZH): {tgt_text}")

        # 预处理源句子
        src_tokens = src_text.lower().split()[:MAX_LENGTH - 2]
        src_ids = [src_vocab.get(token, src_vocab['<unk>']) for token in src_tokens]
        src_ids = [src_vocab['<sos>']] + src_ids + [src_vocab['<eos>']]
        src_ids += [src_vocab['<pad>']] * (MAX_LENGTH - len(src_ids))
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)  # [1, L]
        src_mask = (src_tensor != src_vocab['<pad>']).long().to(DEVICE)

        # 推理生成
        decoded_ids = greedy_decode(
            model=model,
            src=src_tensor,
            src_mask=src_mask,
            max_len=MAX_LENGTH,
            sos_idx=tgt_vocab['<sos>'],
            eos_idx=tgt_vocab['<eos>'],
            pad_idx=tgt_vocab['<pad>'],
            tgt_vocab_size=tgt_vocab_size
        )

        # 转为文本
        pred_text = ids_to_text(decoded_ids.cpu().tolist(), tgt_vocab, tgt_idx_to_token)
        print(f"Prediction (ZH): {pred_text}")
        print("-" * 80)

if __name__ == "__main__":
    main()