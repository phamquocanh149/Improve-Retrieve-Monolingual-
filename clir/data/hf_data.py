import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from transformers import AutoTokenizer

class LaBSEHuggingFaceDataset(Dataset):
    def __init__(self, vi_file, km_file, indices_file, lev_file, tokenizer_name="sentence-transformers/LaBSE", max_length=128):
        # Tải danh sách văn bản
        with open(vi_file, "r", encoding="utf-8") as f: self.vi_texts = [l.strip() for l in f.readlines()]
        with open(km_file, "r", encoding="utf-8") as f: self.km_texts = [l.strip() for l in f.readlines()]
        
        # Tải ma trận Top-K (K=5)
        self.indices = np.load(indices_file) # Shape: (N, K)
        self.lev_scores = np.load(lev_file)  # Shape: (N, K)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.vi_texts)

    def __getitem__(self, idx):
        vi_text = self.vi_texts[idx]
        km_texts_k = [self.km_texts[i] for i in self.indices[idx]] # Lấy K câu Khmer
        levs = self.lev_scores[idx]
        
        # Tokenize Tiếng Việt (1 câu)
        vi_inputs = self.tokenizer(vi_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        
        # Tokenize K câu Tiếng Khmer (Gom thành 1 batch con)
        km_inputs = self.tokenizer(km_texts_k, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")

        # Cấu trúc Dict CHÍNH XÁC như kiến trúc BiEncoder của tác giả yêu cầu
        return {
            "src": {
                "input_ids": vi_inputs["input_ids"].squeeze(0),
                "attention_mask": vi_inputs["attention_mask"].squeeze(0)
            },
            "tgt": {
                "input_ids": km_inputs["input_ids"],
                "attention_mask": km_inputs["attention_mask"]
            },
            "levs": torch.tensor(levs, dtype=torch.float32)
        }

# Hàm batch (Collate) để nối các khối lại với nhau
def collate_fn(batch):
    src_input_ids = torch.stack([item["src"]["input_ids"] for item in batch])
    src_attention_mask = torch.stack([item["src"]["attention_mask"] for item in batch])
    
    # Do tgt đã có K chiều, ta dùng cat để dàn phẳng nó ra thành (Batch * K, Seq_Len)
    # Đây là định dạng mà hàm tgt_model(**input["tgt"]) của tác giả có thể xử lý mượt mà
    tgt_input_ids = torch.cat([item["tgt"]["input_ids"] for item in batch], dim=0)
    tgt_attention_mask = torch.cat([item["tgt"]["attention_mask"] for item in batch], dim=0)
    
    levs = torch.stack([item["levs"] for item in batch])
    
    return {
        "src": {"input_ids": src_input_ids, "attention_mask": src_attention_mask},
        "tgt": {"input_ids": tgt_input_ids, "attention_mask": tgt_attention_mask},
        "levs": levs
    }