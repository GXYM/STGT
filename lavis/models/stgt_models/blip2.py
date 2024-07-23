
"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from transformers import BertTokenizer
import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.common.utils import get_abs_path
from lavis.models.base_model import BaseModel
from lavis.models.clip_vit import create_clip_vit_L
from lavis.models.videoformer.clip_vit_u import create_eva_vit_u
from lavis.models.videoformer.clip_vit_m import create_eva_vit_m
from lavis.models.videoformer.clip_vit_g import create_eva_vit_g
# from lavis.models.videoformer.eva_vit import create_eva_vit_g
from lavis.models.clip_models.clip_text_encoder import create_clip_text_encoder
from lavis.models.blip4video_models.Qformer import BertConfig, BertLMHeadModel
# from lavis.models.blip4video_models.med import BertConfig, BertLMHeadModel
from lavis.models.videoformer.vit_utils import CrossGraphTransformer



class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer = BertTokenizer.from_pretrained("./lavis/bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    @classmethod
    def init_clip_text_encoder(
        cls, embed_dim=1024, 
        context_length = 77,
        vocab_size = 49408,
        trans_width = 1024,
        trans_heads = 16,
        trans_layers = 24,
        use_learn_prompt=False,
        num_learn_token=8,
        multilingual= False,
        multilingual_model="",
        precision="fp32",
        model_path=None):
        
        text_encoder = create_clip_text_encoder(embed_dim=embed_dim, 
                                                context_length = context_length,
                                                vocab_size = vocab_size, 
                                                trans_width = trans_width, 
                                                trans_heads = trans_heads,
                                                trans_layers = trans_layers, 
                                                use_learn_prompt=use_learn_prompt,
                                                num_learn_token=num_learn_token,
                                                multilingual= multilingual,
                                                multilingual_model=multilingual_model,
                                                precision=precision, 
                                                model_path=model_path, 
                                                )
        return text_encoder

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2, config_path= None):
        # encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        if config_path is None:
            config_path="./lavis/bert-base-uncased"
        # load bertconfig
        encoder_config = BertConfig.from_pretrained(config_path)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        # Qformer = BertLMHeadModel.from_pretrained(
        #     "bert-base-uncased", config=encoder_config
        # )
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_vision_encoder(
        cls, 
        model_name, img_size, drop_path_rate, 
        use_grad_checkpoint, precision, vit_model_path,
        # vit-u
        temporal_downsample=True,
        no_lmhra=False, 
        double_lmhra=False,
        lmhra_reduction=2.0, 
        gmhra_layers=8, 
        gmhra_drop_path_rate=0.,
        gmhra_dropout=0.5, 
        # vit-m
        cross_layers=1,
        mit_layers=1,
        video_frames=8,
        # vit-g
        sim_th = 0.618,
    ):
        assert model_name in [
            "clip_vit_u",
            "clip_vit_m",
            "clip_vit_g",
            "clip_vit_l",
        ], "vit model must be clip_vit_u, clip_vit_m, clip_vit_g, clip_vit_l"
        # vit uniformer v2
        if model_name == "clip_vit_u":
            visual_encoder = create_eva_vit_u(
            img_size, drop_path_rate, 
            use_grad_checkpoint, precision, vit_model_path,
            temporal_downsample=temporal_downsample,
            no_lmhra=no_lmhra, 
            double_lmhra=double_lmhra,
            lmhra_reduction=lmhra_reduction, 
            gmhra_layers=gmhra_layers, 
            gmhra_drop_path_rate=gmhra_drop_path_rate,
            gmhra_dropout=gmhra_dropout,)

        # vit m
        elif model_name == "clip_vit_m":
            visual_encoder = create_eva_vit_m(
                img_size,drop_path_rate,
                cross_layers=cross_layers,
                mit_layers=mit_layers,
                video_frames= video_frames,
                use_checkpoint=use_grad_checkpoint, 
                precision=precision, 
                vit_model_path=vit_model_path,)
        elif model_name == "clip_vit_l":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision, vit_model_path)
        # vit g
        elif model_name == "clip_vit_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision, vit_model_path)
           
        # (self, width: int, layers: int, heads: int, T: int, sim_th: int)
        # add by somoszhang
        embed_dim = visual_encoder.num_features
        graph_cross_model = CrossGraphTransformer(embed_dim, cross_layers, heads=16,
                                                T=video_frames, sim_th = sim_th, use_checkpoint=use_grad_checkpoint, )
        
        ln_vision = LayerNorm(embed_dim)
        # cls.vit_name = model_name
        return visual_encoder, ln_vision, graph_cross_model

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def get_optimizer_params(self, weight_decay, lr_scale=1):

        vit_num_layers = self.visual_encoder.get_num_layer()
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                group_name = "vit_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        # import json
        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")
    max_txt_len = kwargs.pop("max_txt_len")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_txt_len,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
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
            image = image.unsqueeze(2)
        else:
            image = samples["video"]
            
        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)
    
    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        # sim_i2t = sim_q2t.mean(0) 
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
       
        # score_matrix_i2t[start + i, topk_idx] = score*topk_sim
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim
        # score_matrix_i2t[start + i, topk_idx] = topk_sim
        # print(score_matrix_i2t[start + i, topk_idx])

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        # score_matrix_t2i[start + i, topk_idx] = score*topk_sim
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim
        # score_matrix_t2i[start + i, topk_idx] = topk_sim
    
    # max_value = torch.max(score_matrix_t2i)
    # min_value = torch.min(score_matrix_t2i)
    # print(max_value, min_value)

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

     
    # max_value = torch.max(score_matrix_t2i)
    # min_value = torch.min(score_matrix_t2i)
    # print(max_value, min_value)

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
