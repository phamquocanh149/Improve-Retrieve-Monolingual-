import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import sys
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks import Callback

# Tự chế một Callback để moi điểm Loss ra in
class LossPrinterCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Cứ mỗi 100 bước thì in ra 1 lần cho đỡ rác màn hình
        if batch_idx % 100 == 0:
            # Rút trích giá trị Loss trực tiếp từ kết quả tính toán
            loss_val = outputs['loss'].item() if isinstance(outputs, dict) else outputs.item()
            print(f"👉 [Epoch {trainer.current_epoch} | Step {batch_idx}] Train Loss: {loss_val:.4f}")

# Gọi nó ra
loss_printer = LossPrinterCallback()

# Ép Python phải nhìn vào thư mục gốc của project
PROJECT_ROOT = "/kaggle/working/Improve-Retrieve-Monolingual--qap_fix_khmer_v2"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from clir.data.hf_data import LaBSEHuggingFaceDataset, collate_fn
from clir.train.trainee import BiEncoder

# ==========================================
# 0. HÀM TẠO MA TRẬN 1-1 (CHỐNG LỖI TỐI THƯỢNG)
# ==========================================
def create_dummy_matrices(text_file, indices_out, lev_out, k=1):
    if os.path.exists(indices_out) and os.path.exists(lev_out):
        return
    
    print(f"🛠️ Đang tạo ma trận giả (K={k}) cho: {os.path.basename(text_file)}...")
    os.makedirs(os.path.dirname(indices_out), exist_ok=True)
    
    with open(text_file, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
        
    # Tạo mảng index: [0, 1, 2, 3, ..., num_lines-1]
    # Ép nó thành cột dọc (N x 1) để khớp với chuẩn K=1
    dummy_indices = np.arange(num_lines).reshape(-1, 1).astype(np.int64)
    
    # Điểm Levenshtein cho câu chuẩn luôn là 1.0
    dummy_levs = np.ones((num_lines, 1), dtype=np.float32)
    
    np.save(indices_out, dummy_indices)
    np.save(lev_out, dummy_levs)

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU
# ==========================================
DIR_RAW = "/kaggle/working/Improve-Retrieve-Monolingual--qap_fix_khmer_v2/raw"
DIR_RET = "/kaggle/working/Improve-Retrieve-Monolingual--qap_fix_khmer_v2/retrieval"

# K=1: Mỗi câu Việt ép DataLoader nạp đúng 1 câu Khmer tương ứng
K_RETRIEVE = 1 
BATCH_SIZE = 64 

OFFLINE_MODEL_PATH = "/kaggle/input/datasets/pqanhhl149/sentence-transformerslabse/LaBSE"

# Sinh ma trận k=1 cho tập Train và Valid
create_dummy_matrices(f"{DIR_RAW}/train.vi", f"{DIR_RET}/indices-cat-train-full-k={K_RETRIEVE}.npy", f"{DIR_RET}/lev-cat-train-full-k={K_RETRIEVE}.npy", k=K_RETRIEVE)
create_dummy_matrices(f"{DIR_RAW}/valid.vi", f"{DIR_RET}/indices-cat-valid-full-k={K_RETRIEVE}.npy", f"{DIR_RET}/lev-cat-valid-full-k={K_RETRIEVE}.npy", k=K_RETRIEVE)

print("⏳ Đang nạp dữ liệu Huấn luyện (Train) cho STAGE 1...")
train_dataset = LaBSEHuggingFaceDataset(
    vi_file=f"{DIR_RAW}/train.vi",
    km_file=f"{DIR_RAW}/train.km",
    indices_file=f"{DIR_RET}/indices-cat-train-full-k={K_RETRIEVE}.npy",
    lev_file=f"{DIR_RET}/lev-cat-train-full-k={K_RETRIEVE}.npy",
    max_length=256,
    tokenizer_name=OFFLINE_MODEL_PATH
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)

print("⏳ Đang nạp dữ liệu Đánh giá (Valid) cho STAGE 1...")
valid_dataset = LaBSEHuggingFaceDataset(
    vi_file=f"{DIR_RAW}/valid.vi",
    km_file=f"{DIR_RAW}/valid.km",
    indices_file=f"{DIR_RET}/indices-cat-valid-full-k={K_RETRIEVE}.npy",
    lev_file=f"{DIR_RET}/lev-cat-valid-full-k={K_RETRIEVE}.npy",
    max_length=256,
    tokenizer_name=OFFLINE_MODEL_PATH
)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH (BiEncoder) - STAGE 1
# ==========================================
print("🧠 Đang khởi tạo mô hình LaBSE cho STAGE 1...")
model = BiEncoder(
    model_name_or_path=OFFLINE_MODEL_PATH,
    vocab_size=501153, 
    pad_token_id=0,
    lr=2e-5,               
    warmup_steps=5000,     

    lev_train=False,       
    bow_loss=True,         
    bow_lr=1e-2,           
    bow_loss_factor=0.1,   
    bow_multiplicator=50.0,
    label_smoothing=0.1,   
    label_smoothing_bow=0.1,

    normalize=True,           
    divide_in_k=False,        
    lr_scheduler="isqrt"
)

# ==========================================
# 3. CẤU HÌNH PYTORCH LIGHTNING TRAINER
# ==========================================
checkpoint_callback = ModelCheckpoint(
    dirpath="/kaggle/working/my_best_model_checkpoints",
    filename="labse-khmer-stage1-best", 
    monitor="eval/loss",          # Theo dõi điểm Loss của tập Valid
    mode="min",                   # Ưu tiên Loss nhỏ nhất
    save_top_k=1,                 # CHỈ lưu 1 cái tốt nhất
    save_last=False,              # TẮT lưu file cuối cùng để tiết kiệm dung lượng
    save_weights_only=True        # RẤT QUAN TRỌNG: Ép giảm 50% dung lượng file (Bỏ qua Optimizer)
)

# THÊM VŨ KHÍ MỚI: Tùy chỉnh thanh tiến trình để không bị spam màn hình
# refresh_rate=100: Cứ 100 bước mới cập nhật màn hình 1 lần (Thay vì mỗi bước 1 lần)
progress_bar = TQDMProgressBar(refresh_rate=100) 

trainer = pl.Trainer(
    max_epochs=7, 
    accelerator="gpu",
    devices=1,
    precision="16-mixed", 
    # Nhớ thêm progress_bar vào danh sách callbacks!
    callbacks=[checkpoint_callback, progress_bar, loss_printer], # <--- Thêm vào đây
    log_every_n_steps=100 # Giảm cả tần suất ghi log nội bộ cho nhẹ máy
)

# ==========================================
# 4. KÍCH HOẠT HUẤN LUYỆN
# ==========================================
print("🔥 BẮT ĐẦU HUẤN LUYỆN STAGE 1...")
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

print("\n🎉 XONG STAGE 1! Mô hình nền tảng đã được lưu tại:")
print("👉 /kaggle/working/my_best_model_checkpoints/labse-khmer-stage1-last.ckpt")