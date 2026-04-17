import os
import sys
import torch
import numpy as np
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import PrependTokenDataset
from fairseq.data import iterators, data_utils
from fairseq.data import Dictionary, FairseqDataset
from torch.utils.data import DataLoader, Dataset
from functools import partial
from typing import *
import time
import random


def load_data_iter_and_lev_from_path(data_path, dict_path, split, src, tgt, retrieval_path, lev_path, max_positions=1024, max_tokens=2048, max_sentences=100, max_lev=2, random_select=False, bos="<s>", eos="</s>", unk="<unk>", pad="<pad>"):
    mono_dict = Dictionary.load(dict_path, add_special_symbols=(bos is None))
    if bos is not None and bos[:3] == "<<<":
        bos = f"[{bos[3:-3]}]"
        eos = f"[{eos[3:-3]}]"
        unk = f"[{unk[3:-3]}]"
        pad = f"[{pad[3:-3]}]"

        mono_dict.bos_word, mono_dict.unk_word, mono_dict.pad_word, mono_dict.eos_word = bos, unk, pad, eos
        # print(bos, unk, pad, eos)
        # print(mono_dict.indices['!'])
        mono_dict.bos_index = mono_dict.add_symbol(bos)
        mono_dict.pad_index = mono_dict.add_symbol(pad)
        mono_dict.eos_index = mono_dict.add_symbol(eos)
        mono_dict.unk_index = mono_dict.add_symbol(unk)
        remove_offset = True
    else:
        # bos = "<s>"
        # eos = "</s>"
        # unk = "<unk>"
        # pad = "<pad>"
        remove_offset = False


    # print(mono_dict.bos(), mono_dict.eos(), mono_dict.unk(), mono_dict.pad())
    # sys.exit(8)
    # # print(data_path, f"{split}.{src}-{tgt}.{src}")
    print(">> src dataset path:", os.path.join(data_path, f"{split}.{src}-{tgt}.{src}"))
    src_dataset = data_utils.load_indexed_dataset(
        os.path.join(data_path, f"{split}.{src}-{tgt}.{src}"),
        mono_dict,
        "mmap",
        combine=True,
    )
    assert src_dataset is not None
    src_dataset = PrependTokenDataset(src_dataset, mono_dict.bos())
    tgt_dataset = data_utils.load_indexed_dataset(
        os.path.join(data_path, f"train.{src}-{tgt}.{tgt}"),
        mono_dict,
        "mmap",
        combine=True,
    )
    tgt_dataset = PrependTokenDataset(tgt_dataset, mono_dict.bos())
    
    retrieval_ids = np.load(retrieval_path)[:, :max_lev].copy()
    retrieval_ids = torch.from_numpy(retrieval_ids)
    levs = np.load(lev_path)[:, :max_lev].copy()
    levs = torch.from_numpy(levs).float()
    # make composite dataset
    dataset = ContrastiveDataset(src_dataset, tgt_dataset, retrieval_ids, levs, max_length=max_positions, eos=mono_dict.eos(), pad=mono_dict.pad(), no_train=(split != "train"), random_select=random_select, remove_offset=remove_offset)

    return load_epoch_iter(
        dataset,
        seed=0,
        max_positions=max_positions,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=1,
        ignore_invalid_inputs=True
    ), mono_dict


def load_data_iter_from_path(data_path, split, src, tgt, max_positions=1024, max_tokens=2048, max_sentences=100):
    dataset, src_dict = load_data_from_path(data_path, split, src, tgt)
    return load_epoch_iter(
        dataset,
        seed=0,
        max_positions=max_positions,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=1,
        ignore_invalid_inputs=True
    ), src_dict


def load_data_from_path(data_path, split, src, tgt):
    src_dict = Dictionary.load(os.path.join(data_path, f"dict.{src}.txt"))
    tgt_dict = src_dict
    return load_data(data_path, split, src_dict, tgt_dict, src, tgt), src_dict


def load_data(
    data_path,
    split,
    src_dict,
    tgt_dict,
    src,
    tgt
):
    return load_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        True,
        "mmap",
        1,
        False,
        False,
        1024,
        1024,
        prepend_bos=True
    )


def load_epoch_iter(
    dataset,
    seed=0,
    max_positions=1024,
    max_tokens=3000,
    max_sentences=100,
    required_batch_size_multiple=2,
    ignore_invalid_inputs=True
):
    # get indices ordered by example size
    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()

    # filter examples that are too large
    if max_positions is not None:
        indices, _ = dataset.filter_indices_by_size(indices, max_positions)

    # create mini-batches with given size constraints
    batch_sampler = dataset.batch_by_size(
        indices,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
    )

    # return a reusable, sharded iterator
    epoch_iter = iterators.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_sampler,
        seed=seed,
        num_shards=1,
        shard_id=0,
        num_workers=2,
        epoch=1,
        buffer_size=0,
        return_dataloader=True
    )

    return epoch_iter


# class IndexerDataset(Dataset):
class IndexerDataset(FairseqDataset):
    def __init__(self, dataset, max_length=512, eos=2, pad=1, remove_offset=False, max_id=None):
        self.dataset = dataset
        self.eos = eos
        self.pad = pad
        self.max_length = max_length
        self.max_id = max_id
        self.sizes = np.array([min(len(self.dataset[idx]), max_length) for idx in range(len(self))])
        self.remove_offset = remove_offset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        if len(self.dataset[idx]) > self.max_length:
            x = x[:self.max_length]
        if self.remove_offset:
            x[1:] -= 4
        else:
            x[x >= self.max_id] = 3
        x[-1] = self.eos
        # print(x)
        # sys.exit(8)
        return dict(id=torch.tensor(idx), tokens=x)

    def ordered_indices(self):
        return np.argsort(self.sizes, kind="mergesort")

    def filter_indices_by_size(self, indices, max_position):
        return indices, None

    def num_tokens(self, index):
        return self.sizes[index]

    def num_tokens_vec(self, indices):
        return self.sizes[indices]

    def size(self, index):
        return self.sizes[index]

    def collater(self, samples, **kwargs):
        out = dict()
        assert len(samples) > 0
        for k in samples[0]:
            if samples[0][k].dim() == 0:
                out[k] = torch.tensor([dic[k] for dic in samples])
            else:
                out[k] = data_utils.collate_tokens([dic[k] for dic in samples], self.pad, **kwargs)
        return out

def collater(pad_idx, dict_list, **kwargs):
    out = dict()
    assert len(dict_list) > 0
    for k in dict_list[0]:
        if dict_list[0][k].dim() == 0:
            out[k] = torch.tensor([dic[k] for dic in dict_list])
        else:
            out[k] = data_utils.collate_tokens([dic[k] for dic in dict_list], pad_idx, **kwargs)
    return out

class ContrastiveDataset(FairseqDataset):
    def __init__(self, src_dataset, tgt_dataset, indices, levs, max_length=512, eos=2, pad=1, no_train=False, random_select=False, remove_offset=False, max_id=None):
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.levs = levs
        self.indices = indices
        self.eos = eos
        self.pad = pad
        self.max_length = max_length
        self.max_id = max_id
        self.src_sizes = np.array([min(len(self.src_dataset[idx]), max_length) for idx in range(len(self))])
        self.tgt_sizes = np.array([min(len(self.tgt_dataset[idx]), max_length) for idx in range(len(self))])
        self.remove_offset = remove_offset
    
    def __len__(self):
        return len(self.src_dataset)

    def __getitem__(self, idx):
        x = self.src_dataset[idx]
        if len(x) > self.max_length:
            x = x[:self.max_length]
        # if self.remove_offset:
        #     x[1:] -= 4
        # else:
        #     x[x >= self.max_id] = 3
        if self.remove_offset:
            x[1:] -= 4
        else:
            if self.max_id is not None:    # Thêm điều kiện kiểm tra an toàn
                x[x >= self.max_id] = 3
        x[-1] = self.eos
        ys = list()
        levs_at_ids = list()
        for i in range(self.levs.shape[1]):
            ret_idx = self.indices[idx, i] % len(self.tgt_dataset)
            y = self.tgt_dataset[ret_idx]
            if len(y) > self.max_length:
                y = y[:self.max_length]
            if self.remove_offset:
                y[1:] -= 4
            y[-1] = self.eos
            ys.append(y)
            levs_at_ids.append(self.levs[idx, i])
        levs_at_ids = torch.tensor(levs_at_ids)
        # print(dict(id=torch.tensor(idx), net_input=dict(src_tokens=x), target=ys, levs=levs_at_ids))
        return dict(id=torch.tensor(idx), net_input=dict(src_tokens=x), target=ys, levs=levs_at_ids)

    def ordered_indices(self):
        return np.argsort(self.src_sizes, kind="mergesort")

    def filter_indices_by_size(self, indices, max_position):
        return indices, None

    def num_tokens(self, index):
        return self.src_sizes[index]

    def num_tokens_vec(self, indices):
        return self.src_sizes[indices]

    def size(self, index):
        return self.src_sizes[index]

    def collater(self, samples, **kwargs):
        out = dict()
        assert len(samples) > 0
        out["id"] = torch.tensor([dic["id"] for dic in samples])
        out["net_input"] = dict()
        out["net_input"]["src_tokens"] = data_utils.collate_tokens([dic["net_input"]["src_tokens"] for dic in samples], self.pad, **kwargs)
        out["target"] = data_utils.collate_tokens([
            e for dic in samples for e in dic["target"]
        ], self.pad, **kwargs)
        out["levs"] = torch.cat([dic["levs"] for dic in samples])
        out["nsentences"] = len(samples)

        return out


    @staticmethod
    def split_in_chunks_(xs, max_tokens):
        upper = len(xs)
        ii = list()
        # for i in range(k):
        i = 0
        # while upper > len(xs) / (k * 2):
        while xs[upper - 1].item() * upper > max_tokens / 2:
            # print(i)
            left = 0
            right = upper
            ii.append(upper)
            while right - left > 1:
                middle = (left + right) // 2
                num_tokens = xs[upper - 1].item() * (upper - middle)
                # print(left, right, middle, num_tokens)
                if num_tokens >= max_tokens:
                    left = middle
                elif num_tokens == max_tokens:
                    break
                else:
                    right = middle
            upper = right
            if (upper == 0):
                break
            i += 1
        if len(ii) == 0:
            ii.append(upper)
        ii.append(0)
        return torch.tensor(ii[::-1], dtype=torch.int32)
        
    @staticmethod
    def split_in_chunks(xs, max_tokens):
        upper = len(xs)
        ii = list()
        height = xs[upper - 1].item()
        while height * upper > max_tokens / 2 and upper > 0:
            ii.append(upper)
            upper -= int(max_tokens / height)
            height = xs[upper - 1].item()
        if len(ii) == 0:
            ii.append(upper)
        ii.append(0)
        return ii[::-1]
        # return [0, len(xs) // 2, len(xs)]

    @staticmethod
    def divide_in_k(ys, pad=1, max_tokens=3000):
        # t1 = time.time()
        lens = ys.ne(pad).sum(-1) #.cpu()
        sorted_lens, sorted_idx = lens.sort()
        split_ids = ContrastiveDataset.split_in_chunks(sorted_lens, max_tokens)
        # print("split_ids", split_ids)
        new_ys = list()
        for j in range(len(split_ids) - 1):
            if split_ids[j] < split_ids[j+1]:
                max_len = sorted_lens[split_ids[j+1]-1]
                new_y = torch.full((split_ids[j+1] - split_ids[j], max_len), pad, dtype=ys.dtype, device=ys.device)
                new_y = ys[sorted_idx][split_ids[j]:split_ids[j+1], :max_len]
                new_ys.append(new_y)
        assert sum([len(y) for y in new_ys]) == len(ys), f"{sum([len(y) for y in new_ys])} vs {len(ys)}"
        # t2 = time.time()
        # print("Dt =", t2 - t1)
        return new_ys, sorted_idx

    @staticmethod
    def reconsitute_from_k(chunks, sorted_idx):
        chunks = torch.concatenate(chunks)
        out = torch.empty_like(chunks)
        out[sorted_idx] = chunks
        return out


def load_monolingual_corpus(data_path=None, dict_path=None, batch_size=None, bos="<s>", eos="</s>", unk="<unk>", pad="<pad>"):
    mono_dict = Dictionary.load(dict_path, add_special_symbols=(bos is None))
    if bos is not None and bos[:3] == "<<<":
        bos = f"[{bos[3:-3]}]"
        eos = f"[{eos[3:-3]}]"
        unk = f"[{unk[3:-3]}]"
        pad = f"[{pad[3:-3]}]"

        mono_dict.bos_word, mono_dict.unk_word, mono_dict.pad_word, mono_dict.eos_word = bos, unk, pad, eos
        # print(bos, unk, pad, eos)
        # print(mono_dict.indices['!'])
        mono_dict.bos_index = mono_dict.add_symbol(bos)
        mono_dict.pad_index = mono_dict.add_symbol(pad)
        mono_dict.eos_index = mono_dict.add_symbol(eos)
        mono_dict.unk_index = mono_dict.add_symbol(unk)
        remove_offset = True
    else:
        # bos = "<s>"
        # eos = "</s>"
        # unk = "<unk>"
        # pad = "<pad>"
        remove_offset = False

    # print(mono_dict.bos(), mono_dict.eos(), mono_dict.unk(), mono_dict.pad())

    dataset = data_utils.load_indexed_dataset(
        data_path,
        mono_dict,
        "mmap",
        combine=True,
    )

    dataset = IndexerDataset(PrependTokenDataset(dataset, mono_dict.bos()), max_length=512, eos=mono_dict.eos(), pad=mono_dict.pad(), remove_offset=remove_offset, max_id=len(mono_dict)-4)
    return load_epoch_iter(
        dataset,
        seed=0,
        max_positions=512,
        max_tokens=batch_size,
        max_sentences=100,
        required_batch_size_multiple=1,
        ignore_invalid_inputs=True
    ), mono_dict
    # return DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     collate_fn=partial(collater, mono_dict.pad_index)
    # ), mono_dict