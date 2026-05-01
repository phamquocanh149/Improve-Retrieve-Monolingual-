# coding: utf-8

from pathlib import Path
import json
import os
import warnings

import re
import string

import transformers
import torch
from transformers import BertModel, XLMModel, XLMRobertaModel
from ..models import LABSEModule

ClassModel = BertModel

def get_pretrained(pretrained_model_name_or_path, **kwargs):
    if pretrained_model_name_or_path is None:
        config = ClassModel.config_class(**kwargs)
        model = ClassModel(config)
        print(f"Randomly initialized model:\n{model}")
    elif pretrained_model_name_or_path == "labse":
        model = LABSEModule()
        config = None
    else:
        # --- BẢN VÁ LỖI: PHÂN BIỆT FILE VÀ THƯ MỤC ---
        if os.path.isfile(pretrained_model_name_or_path):
            # Nếu là file .pt (Cách cũ của tác giả)
            model = torch.load(pretrained_model_name_or_path)
            config = None
        else:
            # Nếu là Thư mục chứa mô hình Offline (Chuẩn Hugging Face)
            print(f"Đang nạp mô hình HuggingFace từ thư mục nội bộ: {pretrained_model_name_or_path}")
            model = ClassModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
            config = model.config
        # ---------------------------------------------
        
    return model, config


def load_pretrained_in_kwargs(kwargs):
    """Recursively loads pre-trained models/tokenizer in kwargs using get_pretrained"""
    # base case: load pre-trained model
    if 'class_name' in kwargs:
        return get_pretrained(**kwargs)
    # recursively look in the child arguments
    for k, v in kwargs.items():
        if isinstance(v, dict):
            kwargs[k] = load_pretrained_in_kwargs(v)
        # else keep as is
    return kwargs


if __name__ == '__main__':
    pass