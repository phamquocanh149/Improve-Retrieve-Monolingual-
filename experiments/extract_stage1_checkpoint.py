import os
import torch
from transformers import AutoModel, AutoTokenizer

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
# Đường dẫn tới file Checkpoint vừa train xong (Thay đổi tên file nếu cần)
CKPT_PATH = "/kaggle/working/my_best_model_checkpoints/labse-khmer-stage1-best.ckpt" 

# Nơi chứa mô hình LaBSE gốc (để lấy cấu trúc mạng lưới)
OFFLINE_MODEL_PATH = "/kaggle/input/datasets/pqanhhl149/sentence-transformerslabse/LaBSE"

# Nơi lưu kết quả sau khi "bóc vỏ"
OUTPUT_DIR = "/kaggle/working/labse_stage1_hf"

def extract_model():
    print(f"📦 1. Đang mở hộp Checkpoint từ: {CKPT_PATH}...")
    if not os.path.exists(CKPT_PATH):
        # PyTorch Lightning đôi khi tự thêm đuôi '-v1', '-v2'. Hàm này giúp tìm file tự động nếu sai tên
        ckpt_dir = os.path.dirname(CKPT_PATH)
        files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if not files:
            raise FileNotFoundError("Không tìm thấy file .ckpt nào trong thư mục!")
        real_ckpt_path = os.path.join(ckpt_dir, files[0])
        print(f"⚠️ Tên file có chút khác biệt, đã tự động tìm thấy: {real_ckpt_path}")
    else:
        real_ckpt_path = CKPT_PATH

    ckpt = torch.load(real_ckpt_path, map_location="cpu")
    
    # Lấy ra bộ từ điển chứa trọng số (state_dict)
    state_dict = ckpt.get("state_dict", ckpt)

    print("✂️ 2. Đang cắt bỏ các vỏ bọc của PyTorch Lightning...")
    hf_state_dict = {}
    for key, value in state_dict.items():
        # Trong BiEncoder của bạn, mô hình cốt lõi thường nằm dưới biến 'src_model' hoặc 'tgt_model'
        if key.startswith("src_model."):
            new_key = key.replace("src_model.", "")
            hf_state_dict[new_key] = value
        elif key.startswith("model."):
            new_key = key.replace("model.", "")
            hf_state_dict[new_key] = value

    print("🧠 3. Đang tải cấu trúc LaBSE gốc để làm khuôn...")
    hf_model = AutoModel.from_pretrained(OFFLINE_MODEL_PATH)

    print("💉 4. Đang bơm trí tuệ (trọng số) mới học được vào khuôn LaBSE...")
    # strict=False giúp mô hình phớt lờ các lớp phụ (như BoW Loss) không thuộc về HF
    hf_model.load_state_dict(hf_state_dict, strict=False)

    print(f"💾 5. Đang lưu mô hình chuẩn Hugging Face ra thư mục: {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hf_model.save_pretrained(OUTPUT_DIR)

    print("📝 6. Đang lưu kèm Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(OFFLINE_MODEL_PATH)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n🎉 HOÀN TẤT! Mô hình của bạn đã sẵn sàng cho bước sinh ma trận FAISS!")

if __name__ == "__main__":
    extract_model()