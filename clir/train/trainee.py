
"""
Trainee is a pl.LightningModule that computes the loss so it is compatible with Trainer.
"""
import sys
import warnings
from functools import partial
import re
from pathlib import Path
import json
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
from transformers import BertModel
# from clir.models import BertWithCustomEmbedding
from clir.models import LABSEModule

from ..data.loading import get_pretrained
from ..data.data import ContrastiveDataset
from .optim import LinearLRWithWarmup, InverseSqrtLRWithWarmup, LabelSmoothingLoss, MSELevenshteinLoss, AdaptiveMarginRankLoss
from ..models import BOWModule
from .metrics import *


class BiEncoder(pl.LightningModule):
    """    
    """
    def __init__(
        self,
        *args,
        model_name_or_path=None,
        vocab_size=30145,
        pad_token_id=0,
        type_vocab_size=2,
        bow_loss=False, bow_loss_factor=0,
        freeze_regex=None, gradient_checkpointing=False,
        warmup_steps=0, lr_scheduler="linear", sqrt_lr_update_factor=100, lr=2e-5, betas=(0.9, 0.999), eps=1e-08, 
        weight_decay=0.0, label_smoothing=0.0, label_smoothing_bow=0.0,
        normalize=False,
        temp_lr=None,
        bow_lr=1e-3,
        bow_multiplicator=1.0,
        lev_train=False,
        lev_loss_type="mse",
        divide_in_k=False,
        **kwargs):
        super().__init__(*args, **kwargs)

        self.freeze_regex = freeze_regex
        self.gradient_checkpointing = gradient_checkpointing
        # scheduling and optimization
        self.warmup_steps = warmup_steps
        self.lr_scheduler = lr_scheduler
        self.sqrt_lr_update_factor = sqrt_lr_update_factor
        self.normalize = normalize
        if self.normalize and not lev_train:
            self.temp = nn.parameter.Parameter(torch.tensor(3.0))
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.metrics = None
        self.label_smoothing = label_smoothing

        # default to symmetric encoders
        # init encoders
        self.pad_token_id = pad_token_id
        
        # =====================================================================
        # HACK TỐI THƯỢNG CHO RTX 6000: BỎ QUA CLI VÀ ÉP NẠP OFFLINE TỪ Ổ CỨNG
        # =====================================================================
        model_name_or_path = '/kaggle/input/datasets/pqanhhl149/sentence-transformerslabse/LaBSE'
        
        self.model_name_or_path = model_name_or_path
        self.src_model, config = get_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            type_vocab_size=type_vocab_size,
        )
        self.tgt_model = self.src_model

        self.second_stage = lev_train
        self.divide_in_k = divide_in_k
        self.lev_loss_type = lev_loss_type
        # loss and metrics
        if self.second_stage: # Second stage Training
            self.max_tokens = 1
            if lev_loss_type == "rank":
                self.lev_loss = AdaptiveMarginRankLoss(sigma=0.5)
            else:
                self.lev_loss = MSELevenshteinLoss(loss_type=lev_loss_type)
            self.param_groups = self.parameters()
            self.param_groups = [
                {"params": self.lev_loss.parameters(), "lr": temp_lr},
                {"params": self.src_model.parameters()},
            ]
            self.metrics = MetricCollection([InBatchNDCG()])
        else: # First Stage Training
            self.max_tokens = 1
            self.log_softmax = nn.LogSoftmax(1)
            if self.label_smoothing < 1e-4:
                self.loss_fct = nn.NLLLoss(reduction='mean')
            else:
                self.loss_fct = LabelSmoothingLoss(self.label_smoothing)

            self.bow_multiplicator = bow_multiplicator
            if bow_loss and bow_loss_factor > 0.0:
                self.bow_loss_src_tgt = BOWModule(config.hidden_size, vocab_size, factor=bow_loss_factor, label_smoothing=label_smoothing_bow, bow_multiplicator=self.bow_multiplicator)
                self.bow_loss_tgt_src = BOWModule(config.hidden_size, vocab_size, factor=bow_loss_factor, label_smoothing=label_smoothing_bow, bow_multiplicator=self.bow_multiplicator)
                self.bow_metric_src_tgt = BOWRecall()
                self.bow_metric_tgt_src = BOWRecall()
            else:
                self.bow_loss_src_tgt = None
                self.bow_loss_tgt_src = None
                self.bow_metric_src_tgt = None
                self.bow_metric_tgt_src = None
            self.metrics = MetricCollection([
                InBatchAccuracy(),
                InBatchMRR()
            ])
            self.temp_lr = temp_lr
            if self.temp_lr is None:
                self.param_groups = self.parameters()
            else:
                self.param_groups = [
                    {"params": [self.temp], "lr": self.temp_lr},
                    {"params": self.src_model.parameters()},
                ]
                if bow_loss and bow_loss_factor > 0.0:
                    self.param_groups.append({"params": self.bow_loss_src_tgt.parameters(), "lr": bow_lr})
                    self.param_groups.append({"params": self.bow_loss_tgt_src.parameters(), "lr": bow_lr})
        
        self.post_init()

    # def make_input_from_fairseq(self, input):
    #     return dict(
    #         src=dict(
    #             input_ids=input["net_input"]["src_tokens"],
    #             attention_mask=input["net_input"]["src_tokens"].ne(self.pad_token_id)
    #         ),
    #         tgt=dict(
    #             input_ids=input["target"],
    #             attention_mask=input["target"].ne(self.pad_token_id)
    #         )
    #     )
        
    def forward(self, input):
        # """        
        # Parameters
        # ----------
        # input: dict
        # """
        # # --- THÊM ĐOẠN NÀY ĐỂ ĐỒNG BỘ 4 ID ĐẶC BIỆT ---
        # if "labse" in self.model_name_or_path.lower():
        #     for key in ["src", "tgt"]:
        #         # Map <s> (0) -> [CLS] (101)
        #         input[key]["input_ids"][input[key]["input_ids"] == 0] = 101
        #         # Map </s> (2) -> [SEP] (102)
        #         input[key]["input_ids"][input[key]["input_ids"] == 2] = 102
        #         # Map <pad> (1) -> [PAD] (0)
        #         input[key]["input_ids"][input[key]["input_ids"] == 1] = 0
        #         # Map <unk> (3) -> [UNK] (100) của LaBSE
        #         input[key]["input_ids"][input[key]["input_ids"] == 3] = 100
                
        #         # Cập nhật lại attention_mask vì ID pad của LaBSE giờ là 0
        #         input[key]["attention_mask"] = input[key]["input_ids"].ne(0)
        # # ---------------------------------------------
        

        # embed questions and contexts
        src_out = self.src_model(**input["src"])
        # if self.model_name_or_path not in ["labse"]:
        #     src_out = src_out.last_hidden_state[:, 0, :]
        # if "labse" not in self.model_name_or_path.lower():
        #     src_out = src_out.last_hidden_state[:, 0, :]
        # SỬA LỖI Ở ĐÂY: Luôn trích xuất Tensor từ Hugging Face Object
        if hasattr(src_out, 'last_hidden_state'):
            src_out = src_out.last_hidden_state[:, 0, :]
        self.max_tokens = max(self.max_tokens, input["src"]["input_ids"].shape[0] * input["src"]["input_ids"].shape[1])
        if self.second_stage and (
            self.divide_in_k or (
                input["tgt"]["input_ids"].shape[0] * input["tgt"]["input_ids"].shape[1] > 
                self.max_tokens * 0.9
            )
        ):
            # DIVIDE IN k FORWARD PASS
            # ACC. TO max_tokens
            # print("src", input["src"]["input_ids"].shape)
            # print("tgt", input["tgt"]["input_ids"].shape)
            # print("max_tokens", self.max_tokens)
            tgt_list, sorted_idx = ContrastiveDataset.divide_in_k(
                input["tgt"]["input_ids"],
                pad=self.pad_token_id,
                max_tokens=self.max_tokens)
            # for tgt in tgt_list:
            #     print(">> tgt", tgt.shape)
            tgt_out = [
                self.tgt_model(toks, toks.ne(self.pad_token_id))
                for toks in tgt_list
            ]
            if self.model_name_or_path not in ["labse"]:
                tgt_out = [
                   x.last_hidden_state[:, 0, :] for x in tgt_out
                ]
            tgt_out = ContrastiveDataset.reconsitute_from_k(tgt_out, sorted_idx)
        else:
            tgt_out = self.tgt_model(**input["tgt"])
            if self.model_name_or_path not in ["labse"]:
                tgt_out = tgt_out.last_hidden_state[:, 0, :]
        if self.normalize:
            return dict(
                src=nn.functional.normalize(src_out),
                tgt=nn.functional.normalize(tgt_out)
            )

        return dict(src=src_out, tgt=tgt_out)

    def step(self, inputs, _):
        """
        Calculates In-batch negatives schema loss and supports to run it in DDP mode 
        by exchanging the representations across all the nodes.
        
        Adapted from https://github.com/facebookresearch/DPR/blob/main/train_dense_encoder.py
        and https://github.com/Lightning-AI/lightning/discussions/14390
        
        Notes
        -----
        This means that the whole representations of questions and contexts, and their similarity matrix, must fit on a single GPU.
        """
        if self.second_stage:
            levs = inputs["levs"]
        # if "net_input" in inputs:
        #     inputs = self.make_input_from_fairseq(inputs)
        outputs = self(inputs)
        
        ##### FOR MULTIPROCESSING sync
        src_out = self.all_gather(outputs["src"], sync_grads=True)
        tgt_out = self.all_gather(outputs["tgt"], sync_grads=True)
        # reshape after all_gather
        if src_out.ndim > 2:
            n_gpus, N, _ = src_out.shape
            src_out = src_out.view(n_gpus*N, -1)
            n_gpus, N, _ = tgt_out.shape
            tgt_out = tgt_out.view(n_gpus*N, -1)

        # compute similarity
        if self.second_stage:
            # src_out: B x d
            # tgt_out: B x k x d
            # levs: B x k
            # k = tgt_out.shape[0] / src_out.shape[0]
            tgt_out = tgt_out.view(src_out.shape[0], -1, src_out.shape[1])
            # B x k
            similarities = torch.einsum('bd,bid->bi', src_out, tgt_out)
            # alpha = 0.6
            # pseudo_lev = torch.clamp(similarities - alpha, min=0) / (1 - alpha)
            loss = self.lev_loss(similarities, levs)
            return dict(loss=loss, similarities=similarities, levs=levs.view(similarities.shape))
        else:
            similarities = src_out @ tgt_out.T  # (B, B)
            assert similarities.size(0) == similarities.size(1)
            if self.normalize:
                similarities *= torch.exp(self.temp)
            log_probs = self.log_softmax(similarities)

            loss_ibns = self.loss_fct(log_probs, torch.arange(len(log_probs), device=log_probs.device))
            loss = loss_ibns
            if self.bow_loss_src_tgt is not None:
                out_src_tgt = self.bow_loss_src_tgt(src_out, inputs["tgt"]["input_ids"])
                loss += out_src_tgt["loss"]
                out_tgt_src = self.bow_loss_tgt_src(tgt_out, inputs["src"]["input_ids"])
                loss += out_tgt_src["loss"]
            else:
                out_src_tgt = None
                out_tgt_src = None
            return dict(loss=loss, ibns_loss=loss_ibns, bow_src_tgt=out_src_tgt, bow_tgt_src=out_tgt_src, log_probs=log_probs)
                    
    def eval_step(self, inputs, batch_idx):
        model_outputs = self.step(inputs, batch_idx)
        # metrics = batch_retrieval(model_outputs['log_probs'])
        # return dict(loss=model_outputs['loss'])
        # print(model_outputs["loss"])
        assert not model_outputs["loss"].isnan()
        return model_outputs
                
    # should be called at the end of each subclass __init__
    def post_init(self):
        if self.freeze_regex is not None:
            self.freeze(self.freeze_regex)        
        if self.gradient_checkpointing:
            self.gradient_checkpointing_enable()
        
    # def eval_step(self, batch, batch_idx):
    #     return self.step(batch, batch_idx)
    
    def log(self, name, value, **kwargs):
        """Ignores None values."""
        if value is None:
            return None
        super().log(name, value, **kwargs)
            
    def training_step(self, batch, batch_idx):
        """Step and log training metrics"""
        outputs = self.step(batch, batch_idx)
        bsz = batch["nsentences"] if "nsentences" in batch else None
        self.log("train/loss", outputs['loss'], batch_size=bsz)
        if 'bow_src_tgt' in outputs and outputs['bow_src_tgt'] is not None:
            self.log("train/ibns_loss", outputs['ibns_loss'], batch_size=bsz, sync_dist=True)
            self.log("train/bow_loss_src_tgt", outputs['bow_src_tgt']['loss'], batch_size=bsz, sync_dist=True)
            self.log("train/bow_loss_tgt_src", outputs['bow_tgt_src']['loss'], batch_size=bsz, sync_dist=True)
        if self.normalize and not self.second_stage:
            self.log("train/temperature", self.temp.data, on_step=True)
        if self.second_stage and self.lev_loss_type in ["mse", "mae"]:
            self.log("alpha", self.lev_loss.get_normalized_alpha(), on_step=True, batch_size=1, on_epoch=False, sync_dist=False)
            self.log("beta", self.lev_loss.get_normalized_beta(), on_step=True, batch_size=1, on_epoch=False, sync_dist=False)
        return outputs
    
    def validation_step(self, batch, batch_idx):
        """Step and log validation metrics"""
        outputs = self.eval_step(batch, batch_idx)
        bsz = batch["nsentences"] if "nsentences" in batch else None
        self.log("eval/loss", outputs['loss'], batch_size=bsz, sync_dist=True, on_step=False, on_epoch=True)
        if 'bow_src_tgt' in outputs and outputs['bow_src_tgt'] is not None:
            self.log("eval/ibns_loss", outputs['ibns_loss'], batch_size=bsz, sync_dist=True, on_step=False, on_epoch=True)
            self.log("eval/bow_loss_src_tgt", outputs['bow_src_tgt']['loss'], batch_size=bsz, sync_dist=True, on_step=False, on_epoch=True)
            self.log("eval/bow_loss_tgt_src", outputs['bow_tgt_src']['loss'], batch_size=bsz, sync_dist=True, on_step=False, on_epoch=True)
            self.bow_metric_src_tgt(outputs['bow_src_tgt']['logprobs'], outputs['bow_src_tgt']['target'])
            self.bow_metric_tgt_src(outputs['bow_tgt_src']['logprobs'], outputs['bow_tgt_src']['target'])
            self.log("eval/bow_acc_src_tgt", self.bow_metric_src_tgt, on_step=False, on_epoch=True)
            self.log("eval/bow_acc_tgt_src", self.bow_metric_tgt_src, on_step=False, on_epoch=True)
        if self.metrics is not None:
            if self.second_stage:
                self.metrics(outputs['similarities'], outputs['levs'])
            else:
                self.metrics(outputs['log_probs'])
            self.log_dict(self.metrics, on_step=False, on_epoch=True)
        return outputs
    
    def test_step(self, batch, batch_idx):
        """Step and log test metrics"""
        outputs = self.eval_step(batch, batch_idx)
        bsz = batch["nsentences"] if "nsentences" in batch else None
        self.log("test/loss", outputs['loss'], batch_size=bsz, sync_dist=True)
        # metrics = batch_metrics(outputs['log_probs'])
        # for key in metrics:
        #     self.log(f"eval/{key}", metrics[key], batch_size=bsz, sync_dist=True)
        return outputs
    
    def eval_epoch_end(self, *args, **kwargs):
        warnings.warn("eval_epoch_end is not implemented.")
        return {}
    
    def freeze(self, regex):
        """
        Overrides freeze to freeze only parameters that match the regex.
        Caveat: does not call .eval() so does not disable Dropout
        """
        regex = re.compile(regex)
        total, frozen = 0, 0
        print("Model parameters:\n"+"Name".ljust(120)+"\t#Trainable\t#Total")
        for name, param in self.named_parameters():
            numel = param.numel()
            if regex.match(name):
                param.requires_grad = False
                frozen += numel
            total += numel
            print(f"{name.ljust(120)}\t{(numel if param.requires_grad else 0):,d}\t{numel:,d}")
        print(f"Froze {frozen:,d} parameters out of {total:,d}")
        
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.param_groups, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        
        # FIXME: this will be overwritten when loading state from ckpt_path
        # so if you want to keep training by increasing total_steps, 
        # your LR will be 0 if the ckpt reached the previously set total_steps
        total_steps=self.trainer.estimated_stepping_batches
        if self.lr_scheduler == "linear":
            scheduler = LinearLRWithWarmup(
                optimizer,
                warmup_steps=self.warmup_steps, total_steps=total_steps
            )
        elif self.lr_scheduler == "isqrt":
            scheduler = InverseSqrtLRWithWarmup(
                optimizer,
                warmup_steps=self.warmup_steps, total_steps=total_steps, update_factor=self.sqrt_lr_update_factor
            )
        else:
            raise ValueError(f"Wrong scheduler name choice: {self.lr_scheduler}. Available: linear, isqrt")
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
        
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.normalize and not self.second_stage:
            self.temp.data = torch.clip(self.temp, -4.6, 6)
        
    
    #####################################################
    # gradient checkpointing: adapted from transformers #
    #####################################################
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
            
    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(getattr(m, "gradient_checkpointing", False) for m in self.modules())

    def save_pretrained(self, ckpt_path):
        self.src_model.save_pretrained(ckpt_path)