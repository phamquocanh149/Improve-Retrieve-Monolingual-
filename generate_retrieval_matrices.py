import os
import gc
import numpy as np
import faiss
import torch
import editdistance  # <--- Vũ khí mới C++
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from khmernltk import word_tokenize

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN 
# ==========================================
DIR_RAW = "/kaggle/working/Improve-Retrieve-Monolingual--qap_fix_khmer/raw"
DIR_OUT = "/kaggle/working/Improve-Retrieve-Monolingual--qap_fix_khmer/retrieval"
K_RETRIEVE = 5
BATCH_SIZE = 512

os.makedirs(DIR_OUT, exist_ok=True)

def tokenize_khmer_safe(text):
    try:
        tokens = word_tokenize(text, return_tokens=True)
        if isinstance(tokens, list): return tokens
        if '\u200b' in tokens: return tokens.split('\u200b')
        return tokens.split()
    except:
        return str(text).split()

def calc_lev_score(true_tokens, retrieved_tokens):
    # Sử dụng lõi C++ để tính toán chớp nhoáng
    dist = editdistance.eval(true_tokens, retrieved_tokens)
    max_len = max(len(true_tokens), len(retrieved_tokens))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0

def process_split(split_name, model, tokenizer):
    print(f"\n{'='*50}\n🚀 ĐANG XỬ LÝ TẬP: {split_name.upper()}\n{'='*50}")
    
    path_vi = os.path.join(DIR_RAW, f"{split_name}.vi")
    path_km = os.path.join(DIR_RAW, f"{split_name}.km")
    
    with open(path_vi, "r", encoding="utf-8") as f: vi_texts = [l.strip() for l in f.readlines()]
    with open(path_km, "r", encoding="utf-8") as f: km_texts = [l.strip() for l in f.readlines()]
    
    def get_embeddings(texts):
        embeds = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Mã hóa LaBSE"):
            batch = texts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
                emb = torch.nn.functional.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)
                embeds.append(emb.cpu().numpy())
        return np.vstack(embeds)

    print("⏳ Đang sinh Vector cho tiếng Việt...")
    vi_emb = get_embeddings(vi_texts)
    print("⏳ Đang sinh Vector cho tiếng Khmer...")
    km_emb = get_embeddings(km_texts)

    print("⏳ Đang chạy thuật toán tìm kiếm Cosine FAISS (CPU)...")
    index = faiss.IndexFlatIP(km_emb.shape[1])
    index.add(km_emb)
    D, I = index.search(vi_emb, K_RETRIEVE)
    
    del vi_emb, km_emb, index
    gc.collect()
    torch.cuda.empty_cache()

    print("⏳ Đang tiêm Ground Truth vào danh sách Top-K...")
    for i in range(len(vi_texts)):
        if i not in I[i]:
            I[i, -1] = i

    print("⏳ Đang tách từ kho Khmer (Chuẩn bị tính Levenshtein)...")
    tokenized_km_corpus = [tokenize_khmer_safe(text) for text in tqdm(km_texts, desc="Tokenize")]

    print("⏳ Đang tính ma trận Levenshtein (Lõi C++ Siêu Tốc)...")
    lev_matrix = np.zeros((len(vi_texts), K_RETRIEVE), dtype=np.float32)

    # Chạy vòng lặp đơn thuần (Đã bỏ joblib vì C++ quá nhanh, không cần làm phức tạp)
    for i in tqdm(range(len(vi_texts)), desc="Tính Levenshtein"):
        true_tokens = tokenized_km_corpus[i]
        for j in range(K_RETRIEVE):
            retrieved_idx = I[i, j]
            retrieved_tokens = tokenized_km_corpus[retrieved_idx]
            lev_matrix[i, j] = calc_lev_score(true_tokens, retrieved_tokens)

    path_indices = os.path.join(DIR_OUT, f"indices-cat-{split_name}-full-k={K_RETRIEVE}.npy")
    path_lev = os.path.join(DIR_OUT, f"lev-cat-{split_name}-full-k={K_RETRIEVE}.npy")
    
    np.save(path_indices, I)
    np.save(path_lev, lev_matrix)
    print(f"✅ HOÀN TẤT tập {split_name.upper()}! Đã lưu file tại:\n - {path_indices}\n - {path_lev}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    print("Khởi tạo mô hình LaBSE...")
    tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/datasets/pqanhhl149/sentence-transformerslabse/LaBSE')
    model = AutoModel.from_pretrained('/kaggle/input/datasets/pqanhhl149/sentence-transformerslabse/LaBSE').to("cuda")
    
    process_split("train", model, tokenizer)
    process_split("valid", model, tokenizer)
    
    print("\n🎉🎉🎉 TOÀN BỘ QUÁ TRÌNH TIỀN TÍNH TOÁN ĐÃ THÀNH CÔNG! SẴN SÀNG HUẤN LUYỆN!")