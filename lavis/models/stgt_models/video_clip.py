"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip4video_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

logger = logging.getLogger(__name__)

@registry.register_model("blip4video_clip")
class Blip4VideoCLIP(Blip2Base):
    """
    Blip4Video first-stage model with clip and ViT.
    Supported model types:
        - pretrain_clip_vig: pretrained model with vit-g and clip
        - pretrain_clip_viu: pretrained model with vit-u and clip
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip4video", "pretrain")
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_clip_vim": "configs/models/blip4video/clip4video_pretrain_vim.yaml",
        "pretrain_clip_viu": "configs/models/blip4video/clip4video_pretrain_viu.yaml"
    }

    def __init__(self, cfg):
        super().__init__()

        vit_model_path = cfg.get("vit_model_path", None)
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        max_txt_len = cfg.get("max_txt_len", 32)

        # uniformerv2
        freeze_mhra = cfg.get("freeze_mhra", False)
        temporal_downsample = cfg.get("temporal_downsample", True)
        no_lmhra = cfg.get("no_lmhra", False)
        double_lmhra = cfg.get("double_lmhra", False)
        lmhra_reduction = cfg.get("lmhra_reduction", 2.0)
        gmhra_layers = cfg.get("gmhra_layers", 8)
        gmhra_drop_path_rate = cfg.get("gmhra_drop_path_rate", 0.)
        gmhra_dropout = cfg.get("gmhra_dropout", 0.5)

        # vit-m
        cross_layers_num  = cfg.get("cross_layers_num", 2)
        mit_layers_num = cfg.get("mit_layers_num", 1)

        # clip parameters 
        freeze_text_encoder= cfg.get("freeze_text_encoder", False)
        multilingual= cfg.get("multilingual", False)
        multilingual_model=cfg.get("multilingual_model_path", "")
        algin_embed_dim = cfg.get("algin_embed_dim", 1024)
        context_length = cfg.get("context_length", 77)
        vocab_size = cfg.get("vocab_size", 49408)
        text_trans_width = cfg.get("text_trans_width", 1024)
        text_trans_heads = cfg.get("text_trans_heads", text_trans_width // 64)
        text_trans_layers = cfg.get("text_trans_layers", 24)
        use_learn_prompt = cfg.get("use_learn_prompt", True)
        num_learn_token = cfg.get("num_learn_token", 8)
        tetx_encoder_path = cfg.get("text_encoder_path", None)

        self.tokenizer = self.init_tokenizer()
        logger.info(f'Loading VIT. Use fp16: {vit_precision}')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, 
            use_grad_checkpoint, vit_precision, vit_model_path,
            temporal_downsample=temporal_downsample,
            no_lmhra=no_lmhra, 
            double_lmhra=double_lmhra,
            lmhra_reduction=lmhra_reduction, 
            gmhra_layers=gmhra_layers, 
            gmhra_drop_path_rate=gmhra_drop_path_rate,
            gmhra_dropout=gmhra_dropout, 
             # vit-m
            cross_layers=cross_layers_num,
            mit_layers=mit_layers_num, 
        )
        
        if freeze_vit:
            logger.info("freeze vision encoder")
            if not freeze_mhra:
                open_list = []
                for name, param in self.visual_encoder.named_parameters():
                    if 'mhra' not in name:
                        param.requires_grad = False
                    else:
                        open_list.append(name)
                logger.info(f"open module: {open_list}")
                logger.info("open ln_vision")
            else:
                for name, param in self.visual_encoder.named_parameters():
                    param.requires_grad = False
                self.visual_encoder = self.visual_encoder.eval()
                self.visual_encoder.train = disabled_train
                for name, param in self.ln_vision.named_parameters():
                    param.requires_grad = False
                self.ln_vision = self.ln_vision.eval()
                self.ln_vision.train = disabled_train
        logger.info('Loading VIT Done')

        logger.info(f'Loading CLIP TEXT ENCODER. Use fp16: {vit_precision}')
        self.text_encoder= self.init_clip_text_encoder(
                embed_dim=algin_embed_dim, context_length = context_length, 
                vocab_size = vocab_size, trans_width = text_trans_width,
                trans_heads = text_trans_heads, trans_layers = text_trans_layers, 
                use_learn_prompt=use_learn_prompt, num_learn_token=num_learn_token,
                multilingual= multilingual,multilingual_model=multilingual_model,
                precision="fp32", model_path=tetx_encoder_path)

        self.clip_vision_proj = nn.Linear(self.visual_encoder.num_features, algin_embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([])*0.07)

        if freeze_text_encoder:
            logger.info("freeze text encoder")
            open_list = []
            for name, param in self.text_encoder.named_parameters():
                if 'learing_tokens' not in name:
                    param.requires_grad = False
                else:
                    open_list.append(name)
            logger.info(f"open text encoder module: {open_list}")


    def forward(self, samples):

        if "image" in samples:
            image = samples["image"]
            image = image.unsqueeze(2)
        else:
            image = samples["video"]
        text = samples["text_input"]

        B,C,T,H,W = image.shape
        use_image = True if T == 1 else False
        # print(image.shape)
        # image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]

        image_embeds = self.ln_vision(self.visual_encoder(image))
        vis_embeds = torch.mean(image_embeds[:, [0, -1], :], dim=1)
        # print(vis_embeds.shape)
        vis_embeds =  self.clip_vision_proj(vis_embeds)
        vis_features = F.normalize(vis_embeds, dim=-1)

        text_embeds =  self.text_encoder(text, image.device)
        text_features = F.normalize(text_embeds, dim=-1)

        loss_clip =  self.text_encoder.loss(vis_features, 
                                            text_features, self.logit_scale.exp())
            
        return BlipOutput(
            loss=loss_clip,
        )

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        vis_embeds = torch.mean(image_embeds[:, [0, -1], :], dim=1)
        vis_embeds =  self.clip_vision_proj(vis_embeds)
        # vis_embeds = F.normalize(vis_embeds, dim=-1)

        return vis_embeds

    def forward_text(self, text_tokens, device):
        text_embeds =  self.text_encoder(text_tokens, device)
        # text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds
    
    def compute_sim_matrix(self, data_loader, task_cfg):
        k_test = task_cfg.get("k_test")
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation:"

        logging.info("Computing features for evaluation...")
        start_time = time.time()

        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_feats = []
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i : min(num_text, i + text_bs)]
            text_input = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)

            text_feat = self.forward_text(text_input)
            text_embed = F.normalize(text_feat)
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)
        
        vit_feats = []
        image_embeds = []
        for samples in data_loader:
            if "image" in samples:
                image = samples["image"]
            else:
                image = samples["video"]
            
            image = image.to(self.device)
            image_feat, vit_feat = self.blip4video.forward_image(image)
            image_embed = self.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)

            vit_feats.append(vit_feat.cpu())
            image_embeds.append(image_embed)
        
        vit_feats = torch.cat(vit_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        sims_matrix = []
        for image_embed in image_embeds:
            sim_q2t = image_embed @ text_embeds.t()
            sim_i2t, _ = sim_q2t.max(0)
            sims_matrix.append(sim_i2t)
        sims_matrix = torch.stack(sims_matrix, dim=0)

        score_matrix_i2t = torch.full(
            (len(data_loader.dataset.image), len(texts)), -100.0
            ).to(self.device)

        num_tasks = dist_utils.get_world_size()
        rank = dist_utils.get_rank()
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
            ):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
            image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(self.device)
            score_matrix_i2t[start + i, topk_idx] = topk_sim

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0).to(self.device)

        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
            ):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)            
            score_matrix_t2i[start + i, topk_idx] =  topk_sim

        if dist_utils.is_dist_avail_and_initialized():
            dist.barrier()
            torch.distributed.all_reduce(
                score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
                )
            torch.distributed.all_reduce(
                score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
                )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


    @classmethod
    def from_config(cls, cfg):
        
        model = cls(cfg)
        model.load_checkpoint_from_config(cfg)

        return model