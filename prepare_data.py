import os
from transformers import AutoTokenizer

# 1. Tải Tokenizer của LaBSE
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")

# 2. Tạo file dict.txt cho Fairseq (Ép Fairseq học từ điển của LaBSE)
vocab = tokenizer.get_vocab()
# Sắp xếp vocab theo ID để đảm bảo đúng thứ tự
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

os.makedirs("data-bin", exist_ok=True)
with open("data-bin/dict.txt", "w", encoding="utf-8") as f:
    # Fairseq mặc định bỏ qua 4 ID đầu (<s/bos>, <pad>, </s>, <unk>)
    # Nên ta ghi từ ID thứ 4 trở đi của LaBSE, kèm theo tần suất ảo (ví dụ: 100)
    for word, idx in sorted_vocab[4:]:
        f.write(f"{word} 100\n")

print(f"Đã tạo xong dict.txt với {len(vocab)} từ vựng!")

# 3. Hàm Tokenize dữ liệu text thô của bạn bằng LaBSE
def tokenize_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            # Tokenize và ghép lại bằng dấu cách (Đúng chuẩn đầu vào fairseq-preprocess)
            tokens = tokenizer.tokenize(line.strip())
            fout.write(" ".join(tokens) + "\n")

print(" Đang Tokenize dữ liệu Train/Valid...")
tokenize_file("raw/train.vi", "tok/train.vi")
tokenize_file("raw/train.km", "tok/train.km")
tokenize_file("raw/valid.vi", "tok/valid.vi")
tokenize_file("raw/valid.km", "tok/valid.km")