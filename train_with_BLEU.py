import torch
import torch.nn.functional as F
import os
import xml.etree.ElementTree as ET
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# 首次运行请取消注释（只需一次）
# nltk.download('punkt')

from src.model.Transformer import Transformer

# ========== 配置 ==========
DATA_DIR = "./datasets/en-zh"
MODEL_PATH = "./module/BaseLine_model_Picture_BLEU.pth"
MAX_LENGTH = 128
BLEU_SAMPLE_LIMIT = 1000  # None 表示全量，整数表示最多评估多少句
BEAM_SIZE = 5  # 设为 1 相当于贪心；>1 为 beam search

# ========== 辅助函数：数据读取与词表 ==========
def read_texts_from_local_dir(data_dir, split='dev2010'):
    src_texts = []
    tgt_texts = []

    src_xml_file = os.path.join(data_dir, f"IWSLT17.TED.{split}.en-zh.en.xml")
    tgt_xml_file = os.path.join(data_dir, f"IWSLT17.TED.{split}.en-zh.zh.xml")
    
    if os.path.exists(src_xml_file) and os.path.exists(tgt_xml_file):
        try:
            src_tree = ET.parse(src_xml_file)
            tgt_tree = ET.parse(tgt_xml_file)
            src_root = src_tree.getroot()
            tgt_root = tgt_tree.getroot()
            src_segs = src_root.findall('.//seg')
            tgt_segs = tgt_root.findall('.//seg')
            for s, t in zip(src_segs, tgt_segs):
                src_texts.append(s.text.strip() if s.text else "")
                tgt_texts.append(t.text.strip() if t.text else "")
        except Exception as e:
            print(f"XML parsing error: {e}")
            return [], []
    else:
        src_txt_file = os.path.join(data_dir, f"{split}.tags.en-zh.en")
        tgt_txt_file = os.path.join(data_dir, f"{split}.tags.en-zh.zh")
        if os.path.exists(src_txt_file) and os.path.exists(tgt_txt_file):
            with open(src_txt_file, 'r', encoding='utf-8') as f_src, \
                 open(tgt_txt_file, 'r', encoding='utf-8') as f_tgt:
                src_texts = [line.strip() for line in f_src]
                tgt_texts = [line.strip() for line in f_tgt]
        else:
            print(f"Could not find validation files for {split}")
            return [], []
    return src_texts, tgt_texts

def char_tokenize(text):
    return list(text.strip())

def ids_to_text(ids, idx_to_token):
    tokens = []
    for idx in ids:
        if idx in [0, 1, 2, 3]:  # <pad>, <sos>, <eos>, <unk>
            continue
        tokens.append(idx_to_token.get(idx, '<unk>'))
    return ''.join(tokens)

# ========== 解码函数 ==========
def greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx, pad_idx, tgt_vocab_size, device):
    model.eval()
    with torch.no_grad():
        ys = torch.full((1, 1), sos_idx, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_len = ys.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)
            logits = model(src, ys, src_mask=src_mask, tgt_mask=tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)
            ys = torch.cat([ys, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == eos_idx:
                break
        return ys.squeeze(0)

def beam_decode(model, src, src_mask, max_len, sos_idx, eos_idx, pad_idx, tgt_vocab_size, device, beam_size=5):
    model.eval()
    with torch.no_grad():
        batch_size = src.size(0)
        assert batch_size == 1, "Beam search here assumes batch_size=1."

        candidates = [(torch.tensor([sos_idx], device=device), 0.0)]

        for step in range(max_len - 1):
            all_candidates = []
            for seq, score in candidates:
                if seq[-1].item() == eos_idx:
                    all_candidates.append((seq, score))
                    continue

                ys = seq.unsqueeze(0)
                tgt_len = ys.size(1)
                tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)

                logits = model(src, ys, src_mask=src_mask, tgt_mask=tgt_mask)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                topk_vals, topk_idxs = log_probs.topk(beam_size, dim=-1)

                for i in range(beam_size):
                    token = topk_idxs[0, i].item()
                    token_score = topk_vals[0, i].item()
                    new_seq = torch.cat([seq, torch.tensor([token], device=device)], dim=0)
                    new_score = score + token_score
                    all_candidates.append((new_seq, new_score))

            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            candidates = ordered[:beam_size]

            if all(cand[0][-1].item() == eos_idx for cand in candidates):
                break

        return candidates[0][0]

# ========== BLEU 评估主函数 ==========
def evaluate_bleu_only():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 加载 checkpoint ---
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    
    tgt_idx_to_token = {idx: token for token, idx in tgt_vocab.items()}
    
    sos_idx = tgt_vocab['<sos>']
    eos_idx = tgt_vocab['<eos>']
    pad_idx = tgt_vocab['<pad>']

    # --- 初始化模型 ---
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=512,
        dropout=0.1
    ).to(device)"""

    model = Transformer(
        len(src_vocab),      # src_vocab_size
        len(tgt_vocab),      # tgt_vocab_size
        256,                 # d_model
        4,                   # n_layers
        8,                   # n_heads
        512,                 # d_ff
        0.1                  # dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully.")

    # --- 加载验证数据 ---
    print("Loading validation data...")
    val_src_texts, val_tgt_texts = read_texts_from_local_dir(DATA_DIR, split='dev2010')
    if not val_src_texts:
        print("Fallback to using 'dev' or 'valid' split...")
        val_src_texts, val_tgt_texts = read_texts_from_local_dir(DATA_DIR, split='dev')
    if not val_src_texts:
        raise FileNotFoundError("No validation data found!")

    print(f"Loaded {len(val_src_texts)} validation sentences.")

    # --- 构建简化 Dataset（仅用于推理）---
    class SimpleValDataset:
        def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len):
            self.src_texts = src_texts
            self.tgt_texts = tgt_texts
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            self.max_len = max_len
            self.src_unk = src_vocab['<unk>']
            self.tgt_unk = tgt_vocab['<unk>']
            self.pad = src_vocab['<pad>']

        def __len__(self):
            return len(self.src_texts)

        def __getitem__(self, idx):
            src_text = self.src_texts[idx]
            tokens = src_text.lower().split()[:self.max_len - 2]
            src_ids = [self.src_vocab.get(t, self.src_unk) for t in tokens]
            src_ids = [self.src_vocab['<sos>']] + src_ids + [self.src_vocab['<eos>']]
            src_ids += [self.pad] * (self.max_len - len(src_ids))
            src_ids = torch.tensor(src_ids[:self.max_len], dtype=torch.long)
            src_mask = (src_ids != self.pad).long()
            return src_ids, src_mask, self.tgt_texts[idx]

    val_dataset = SimpleValDataset(val_src_texts, val_tgt_texts, src_vocab, tgt_vocab, MAX_LENGTH)

    # --- 开始评估 ---
    model.eval()
    smoothie = SmoothingFunction().method4
    num_samples = min(BLEU_SAMPLE_LIMIT, len(val_dataset)) if BLEU_SAMPLE_LIMIT else len(val_dataset)
    total_bleu = 0.0

    with torch.no_grad():
        for i in range(num_samples):
            src_ids, src_mask, ref_text = val_dataset[i]
            src_ids = src_ids.unsqueeze(0).to(device)
            src_mask = src_mask.unsqueeze(0).to(device)

            # 选择解码方式
            if BEAM_SIZE == 1:
                decoded_ids = greedy_decode(
                    model, src_ids, src_mask, MAX_LENGTH, sos_idx, eos_idx, pad_idx, tgt_vocab_size, device
                )
            else:
                decoded_ids = beam_decode(
                    model, src_ids, src_mask, MAX_LENGTH, sos_idx, eos_idx, pad_idx, tgt_vocab_size, device, beam_size=BEAM_SIZE
                )

            pred_text = ids_to_text(decoded_ids.cpu().tolist(), tgt_idx_to_token)
            reference = char_tokenize(ref_text)
            hypothesis = char_tokenize(pred_text)

            if len(hypothesis) == 0:
                bleu_score = 0.0
            else:
                bleu_score = sentence_bleu(
                    [reference],
                    hypothesis,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothie
                )
            total_bleu += bleu_score

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples}, Current BLEU: {total_bleu / (i + 1):.4f}")

    final_bleu = total_bleu / num_samples
    print(f"\n✅ Final BLEU-4 (char-level) on {num_samples} samples: {final_bleu:.4f}")
    return final_bleu

if __name__ == "__main__":
    evaluate_bleu_only()