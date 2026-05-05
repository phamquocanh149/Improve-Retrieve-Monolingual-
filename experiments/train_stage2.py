from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# 1. Khởi tạo Dataset & DataLoader
# Đảm bảo đường dẫn file .npy khớp với script sinh dữ liệu lúc nãy
train_dataset = LaBSEHuggingFaceDataset(
    vi_file="raw/train.vi", km_file="raw/train.km",
    indices_file="retrieval/indices-cat-train-full-k=5.npy",
    lev_file="retrieval/lev-cat-train-full-k=5.npy"
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)

valid_dataset = LaBSEHuggingFaceDataset(
    vi_file="raw/valid.vi", km_file="raw/valid.km",
    indices_file="retrieval/indices-cat-valid-full-k=5.npy",
    lev_file="retrieval/lev-cat-valid-full-k=5.npy"
)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

# 2. Khởi tạo Model (BiEncoder của tác giả)
# Nhớ trả lại alpha=8.0, beta=-7.0 trong file optim.py trước khi chạy nhé!
model = BiEncoder(
    model_name_or_path="sentence-transformers/LaBSE",
    vocab_size=501153,
    pad_token_id=0,
    lr=2e-5, # TỐC ĐỘ HỌC CHUẨN CỦA HUGGING FACE
    temp_lr=1e-3, # Tốc độ học hãm phanh cho alpha/beta
    lev_train=True,
    lev_loss_type="mae",
    divide_in_k=True # Bật tính năng này để xử lý tgt (Batch * K)
)

# 3. Cấu hình PyTorch Lightning Trainer
checkpoint_callback = ModelCheckpoint(
    dirpath="my_hf_checkpoints",
    monitor="eval/loss",
    mode="min",
    save_top_k=1
)
early_stop = EarlyStopping(monitor="eval/loss", patience=3, mode="min")

trainer = Trainer(
    max_epochs=7,
    accelerator="gpu",
    devices=1,
    precision=16, # Bật AMP để train nhanh, tiết kiệm VRAM
    callbacks=[checkpoint_callback, early_stop],
    log_every_n_steps=10
)

# 4. KÍCH HOẠT HUẤN LUYỆN 🔥
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)