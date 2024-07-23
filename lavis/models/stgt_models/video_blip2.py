"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

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
from lavis.models.stgt_models.blip_outputs import BlipOutput, BlipOutputFeatures

logger = logging.getLogger(__name__)

@registry.register_model("blip4video")
@registry.register_model("blip4video_feature_extractor")
class Blip4VideosQformer(Blip2Base):
    """
    blip4video first-stage model with Q-former and ViT.
    Supported model types:
        - pretrain_qformer_vim: pretrained model with vit-m and qformer
        - pretrain_qformer_viu: pretrained model with vit-u and qformer
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip4video", "pretrain")
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_qformer_vig": "configs/models/blip4video/blip4video_pretrain_qformer_vig.yaml",
        "pretrain_qformer_vim": "configs/models/blip4video/blip4video_pretrain_qformer_vim.yaml",
        "pretrain_qformer_viu": "configs/models/blip4video/blip4video_pretrain_qformer_viu.yaml",
        "msrvtt": "configs/models/blip4video/blip4video_retrieval_msrvtt.yaml",
        "didemo": "configs/models/blip4video/blip4video_retrieval_didemo.yaml",
        "msvd": "configs/models/blip4video/blip4video_retrieval_msvd.yaml",
        "lsmdc": "configs/models/blip4video/blip4video_retrieval_lsmdc.yaml",
    }

    def __init__(self, cfg):
        super().__init__()
        
        # Q-Former
        freeze_qformer = cfg.get("freeze_qformer", True)
        qformer_model_path = cfg.get("qformer_model_path", None)
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        # VIT
        freeze_vit = cfg.get("freeze_vit", True)
        open_vit_block = cfg.get("open_vit_block", 0)
        vit_model_path = cfg.get("vit_model_path", None)
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
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
        graph_layers_num  = cfg.get("graph_layers_num", 1)
        video_frames = cfg.get("video_frames", 8)
        sim_th  = cfg.get("sim_th", 0.1)
        self.similarity_loss = cfg.get("self_similarity_loss", True)

        # algin embed dim
        algin_embed_dim = cfg.get("algin_embed_dim", 256)
        self.max_txt_len = max_txt_len
        self.tokenizer = self.init_tokenizer()
        
        logger.info(f'Loading VIT. Use fp: {vit_precision}')
        self.visual_encoder, self.ln_vision, self.graph_cross_model= self.init_vision_encoder(
                vit_model, img_size, drop_path_rate, 
                use_grad_checkpoint, vit_precision, vit_model_path,
                temporal_downsample=temporal_downsample,
                no_lmhra=no_lmhra, 
                double_lmhra=double_lmhra,
                lmhra_reduction=lmhra_reduction, 
                gmhra_layers=gmhra_layers, 
                gmhra_drop_path_rate=gmhra_drop_path_rate,
                gmhra_dropout=gmhra_dropout, 
                # vit-g
                cross_layers=graph_layers_num,
                video_frames=video_frames, 
                sim_th= sim_th,
            )
        
        if freeze_vit:
            logger.info("freeze vision encoder")
            if not freeze_mhra:
                open_list = []
                for name, param in self.visual_encoder.named_parameters():
                    if open_vit_block and name.split(".")[0] == "blocks" \
                                  and int(name.split(".")[1]) > 38-open_vit_block:
                        open_list.append(name)
                        continue
                    if 'mhra' not in name:
                        param.requires_grad = False
                    else:
                        open_list.append(name)
                logger.info(f"open module: {open_list}")
                logger.info("open graph_cross_model")
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
                for name, param in self.graph_cross_model.named_parameters():
                    param.requires_grad = False
                self.graph_cross_model = self.graph_cross_model.eval()
                self.graph_cross_model.train = disabled_train
        logger.info('Loading VIT Done')
        
        logger.info('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        if qformer_model_path:
            self.load_from_pretrained(qformer_model_path)
            logger.info('Loading Q-Former from {}'.format(qformer_model_path))
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")        
        logger.info('Loading Q-Former Done')

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, algin_embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, algin_embed_dim)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        # self.temp = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, samples):
        # print(self.temp)
        # with torch.no_grad():
        #     self.temp.clamp_(0.001, 0.1)
        
        if "image" in samples:
            image = samples["image"]
            image = image.unsqueeze(2)
        else:
            image = samples["video"]
        
        text = samples["text_input"]
        iter_eps = samples["epoch"]

        B, C, T, H, W = image.shape
        use_image = True if T == 1 else False
        # print(image.shape)
        # image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
        image_embeds = self.ln_vision(self.graph_cross_model(self.visual_encoder(image)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )
       
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
       
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(image_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]
        
        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        # sim_i2t = sim_q2t.mean(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()
        # print(text_feat.unsqueeze(1).unsqueeze(1).shape)
        # print(image_feats_all.permute(0, 2, 1).shape)
        # print(sim_t2q.shape)
        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        # sim_t2i = sim_t2q.mean(-1) 
        sim_t2i = sim_t2i  / self.temp
        rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )
        # print(targets)

        if self.similarity_loss: #retrieval finetuning  with  self-similarity loss
            images_embeddings = F.normalize(image_feats.mean(1), dim=-1)
            images_embeddings_all = F.normalize(image_feats_all.mean(1), dim=-1)
            images_similarity =   images_embeddings @ images_embeddings_all.T
            texts_similarity = text_feat @ text_feat_all.T
            
            images_similarity = torch.where(images_similarity<=0.0, 0., images_similarity)
            texts_similarity = torch.where(texts_similarity<=0.0, 0., texts_similarity)
            tal_similarity = images_similarity*texts_similarity
            tal_similarity = torch.where(tal_similarity<0.05, -100, tal_similarity)
            # targets = tal_similarity

            self.temperature = 5
            targets = F.softmax( tal_similarity * self.temperature, dim=-1)

            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.01)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.01)
            ) /2

        else:  

            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) /2

            # loss_itc = (F.nll_loss(F.log_softmax(sim_i2t, dim=1), targets)
            #              + F.nll_loss(F.log_softmax(sim_t2i, dim=1), targets))/2
        
        if torch.isnan(image_feats_all).any():
            nan_mask = torch.isnan(image_feats_all)
            image_feats_all = torch.where(nan_mask, torch.zeros_like(image_feats_all), image_feats_all)
        
        if torch.isnan(text_feat_all).any():
            nan_mask = torch.isnan(text_feat_all)
            text_feat_all = torch.where(nan_mask, torch.zeros_like(text_feat_all), text_feat_all)
        

        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with torch.no_grad():
            if "image_id" in samples.keys():
                mask = torch.eq(image_ids, image_ids_all.t())
                sim_t2i.masked_fill_(mask, -10000)
                sim_i2t.masked_fill_(mask, -10000)
            else:    
                sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
            
            
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

            # weights_t2i = torch.softmax(sim_t2i, dim=1)
            # weights_i2t = torch.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)
        # print(logits)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)

        # loss_itm = F.cross_entropy(logits, itm_labels) #*torch.tensor((iter_eps+1)/6.0)
        # loss_itm = F.cross_entropy(logits, itm_labels, label_smoothing=0.01)

        loss_itm = F.nll_loss(F.log_softmax(logits, dim=1), itm_labels) #*torch.tensor(10)

        # if loss_itc > 5.0:
        #     loss_itm = loss_itm*torch.tensor(0.1)

    
        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_itc + 1.0*loss_itm + 1.0*loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            loss_temp=self.temp,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.graph_cross_model(self.visual_encoder(image)))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions
    
    def forward_image(self, image):
        # image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = self.ln_vision(self.graph_cross_model(self.visual_encoder(image)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings).mean(dim=1)
        itm_logit = torch.softmax(itm_logit, dim=1)[:, 1]
        # itm_logit = itm_logit[:, 1] # no softmax
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.graph_cross_model(self.visual_encoder(image)))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.graph_cross_model(self.visual_encoder(image)))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        
        model = cls(cfg)
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test
        max_txt_len = self.max_txt_len

        return compute_sim_matrix(model=self, 
                                data_loader=data_loader,
                                k_test=k_test, 
                                max_txt_len=max_txt_len)
