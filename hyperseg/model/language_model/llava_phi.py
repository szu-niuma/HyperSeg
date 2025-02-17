from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import numpy as np
import pickle
import torch
import math
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import PhiModel, PhiForCausalLM, PhiConfig

from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom


from ..visual_prompt_module.context_cluster import region_pooling


from ..mipha.model.language_model.mipha_phi import MiphaPhiForCausalLM, MiphaPhiModel
from ..mipha.model.mipha_arch import MiphaMetaModel, MiphaMetaForCausalLM

from hyperseg.utils.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    SEG_TOKEN_INDEX,
    CLS_TOKEN_INDEX,
    REGION_TOKEN_INDEX,
    REFER_TOKEN_INDEX,
    TEMPORAL_TOKEN_INDEX,
)

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoderForOPTPreTrain,
)
from ..mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder

from ..multimodal_encoder.swin_trans import build_swin_b, build_swin_l

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

from .perceiver import PerceiverResampler

from ..datasets_mapper.coco_instance_mapper import COCOInstanceNewBaselineDatasetMapper
from ..datasets_mapper.coco_panoptic_mapper import COCOPanopticNewBaselineDatasetMapper
from ..datasets_mapper.coco_semantic_mapper import COCOSemanticNewBaselineDatasetMapper


@dataclass
class CausalOutputWithMask(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_mask: Optional[torch.FloatTensor] = None
    loss_dice: Optional[torch.FloatTensor] = None
    loss_SEG_class: Optional[torch.FloatTensor] = None
    loss_class_name_class: Optional[torch.FloatTensor] = None
    loss_region_class: Optional[torch.FloatTensor] = None
    loss_vos_class: Optional[torch.FloatTensor] = None
    loss_rvos_class: Optional[torch.FloatTensor] = None
    loss_llm: Optional[torch.FloatTensor] = None
    loss_box: Optional[torch.FloatTensor] = None
    loss_reid: Optional[torch.FloatTensor] = None


class HyperSegModel(MiphaPhiModel):
    """
    继承于PhiModel，实现了HyperSeg的模型
    重新构建vision_tower
    """

    # config_class = LlavaConfig

    def __init__(self, config: PhiConfig, mask_decoder_cfg=None):
        super().__init__(config)
        self.cfg = mask_decoder_cfg
        self.projector_outdim = config.hidden_size
        if hasattr(config, "mm_vision_tower"):
            swin_type = getattr(config, "swin_type", "base")
            if swin_type == "base":
                self.vision_tower_mask = build_swin_b(None)
            else:
                self.vision_tower_mask = build_swin_l(None)

            self.vision_tower_mask.image_processor = {}
            self.vision_tower_mask.image_processor["panoptic"] = COCOPanopticNewBaselineDatasetMapper(self.cfg)
            self.vision_tower_mask.image_processor["instance"] = COCOInstanceNewBaselineDatasetMapper(self.cfg)
            self.vision_tower_mask.image_processor["semantic"] = COCOSemanticNewBaselineDatasetMapper(self.cfg)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_vision_tower_mask(self):
        vision_tower = getattr(self, "vision_tower_mask", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower if hasattr(model_args, "vision_tower") else model_args.mm_vision_tower
        vision_tower_mask = model_args.vision_tower_mask if hasattr(model_args, "vision_tower_mask") else model_args.mm_vision_tower_mask

        self.config.mm_vision_tower = vision_tower
        swin_type = getattr(model_args, "swin_type", "base")
        self.config.swin_type = swin_type
        if swin_type == "base":
            vision_tower_mask = build_swin_b(vision_tower_mask)
        else:
            print("current visual encoder is swin large")
            vision_tower_mask = build_swin_l(vision_tower_mask)

        if fsdp is not None and len(fsdp) > 0:
            # self.vision_tower = [vision_tower]
            self.vision_tower_mask = [vision_tower_mask]
        else:
            # self.vision_tower = vision_tower
            self.vision_tower_mask = vision_tower_mask
        # if model_args.float32:
        #     print('convert sam parameters from fp16 to fp32')
        #     for name, module in self.vision_tower.named_modules():
        #         module = module.to(torch.float32)z

        self.config.use_mm_proj = True
        vision_tower_mask.hidden_size = 256
        vision_tower_mask.image_processor = {
            "panoptic": COCOPanopticNewBaselineDatasetMapper(self.cfg),
            "instance": COCOInstanceNewBaselineDatasetMapper(self.cfg),
            "semantic": COCOSemanticNewBaselineDatasetMapper(self.cfg),
        }


class HyperSeg(MiphaPhiForCausalLM):
    # config_class = LlavaConfig

    def __init__(
        self,
        config,
        mask_decoder_cfg=None,
        cur_vocab_size=None,
        add_cross_attn=True,
        cross_attn_index=None,
        *args,
        **kwargs,
    ):
        super().__init__(config)
        self.model = HyperSegModel(config, mask_decoder_cfg)

        self.init_config = config
        self.mask_decoder_cfg = mask_decoder_cfg
        self.cross_attn_index = cross_attn_index

        # if cur_vocab_size is not None:
        #     self.lm_head = nn.Linear(config.hidden_size, cur_vocab_size, bias=False)
        # else:
        #     self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, 51200, bias=False)
        # Initialize weights and apply final processing
        is_train_mask_decode = getattr(config, "mask_decode_train", False)
        self.is_train_mask_decode = is_train_mask_decode
        self.refer_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.class_name_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.region_embd_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.vos_time_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.region_sampler = region_pooling(num_sample_point=512)
        self.region_projector = nn.Linear(config.hidden_size, mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

        self.generative_mode = False
        self.use_temporal_query = False

        if is_train_mask_decode:
            print("Mask Decoder has been trained, init directly")
            self.initial_mask_module()
        self.post_init()

    def initial_mask_module(self, pretrained_path=None, model_args=None):
        if not self.is_train_mask_decode:
            print("Initialize mask modules...")
            self.config.mask_decode_train = True
        self.seg_query = nn.Parameter(torch.zeros([self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, self.config.hidden_size]))
        self.num_queries = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.num_classes = self.mask_decoder_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        self.seg_query_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.SEG_token_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.class_name_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

        self.enable_mgvp_seg_query = True

        if model_args is not None:
            self.enable_mgvp_seg_query = model_args.enable_mgvp_seg_query

        # FVP module in paper
        if self.enable_mgvp_seg_query:
            self.fvp_num_queries = 128
            self.mgvp_layers_num = 3
            local_fea_dim = [256, 512, 1024]
            self.mgvp_layers = PerceiverResampler(dim=local_fea_dim[1], depth=self.mgvp_layers_num)
            self.temporal_query = nn.Parameter(torch.randn(self.fvp_num_queries, local_fea_dim[1]))

            self.expanded_seg_query_project = nn.Linear(local_fea_dim[1], self.config.hidden_size)
            self.temporal_query_project = nn.Linear(self.config.hidden_size, local_fea_dim[1])

            self.local_project = nn.ModuleList()
            for i in range(self.mgvp_layers_num):
                self.local_project.append(nn.Conv2d(local_fea_dim[i], local_fea_dim[1], kernel_size=1))
                weight_init.c2_xavier_fill(self.local_project[-1])
            self.level_embed = nn.Embedding(self.mgvp_layers_num, local_fea_dim[1])

        self.mask_decoder_training_init(self.mask_decoder_cfg)
        if pretrained_path is not None:

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            def change_w(weights, old_name, new_name):
                weights[new_name] = weights[old_name]
                weights.pop(old_name)

            if pretrained_path.endswith(".pkl"):
                with open(pretrained_path, "rb") as f:
                    ckpt = pickle.load(f)
            else:
                ckpt = torch.load(pretrained_path)
            pixel_decoder_weights = get_w(ckpt["model"], "sem_seg_head.pixel_decoder")
            predictor_weights = get_w(ckpt["model"], "sem_seg_head.predictor")
            pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
            predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

            # deal some diff keys
            change_w(pixel_decoder_weights, "adapter_1.weight", "adapter_1.0.weight")
            change_w(pixel_decoder_weights, "adapter_1.norm.weight", "adapter_1.1.weight")
            change_w(pixel_decoder_weights, "adapter_1.norm.bias", "adapter_1.1.bias")
            change_w(pixel_decoder_weights, "layer_1.weight", "layer_1.0.weight")
            change_w(pixel_decoder_weights, "layer_1.norm.weight", "layer_1.1.weight")
            change_w(pixel_decoder_weights, "layer_1.norm.bias", "layer_1.1.bias")
            if "static_query.weight" in predictor_weights:
                change_w(predictor_weights, "static_query.weight", "query_feat.weight")
            if predictor_weights["query_embed.weight"].shape[0] == 200:
                predictor_weights["query_embed.weight"] = predictor_weights["query_embed.weight"][:100, :]
            diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights, strict=False)
            diff_predictor_msg = self.predictor.load_state_dict(predictor_weights, strict=False)
            print(diff_predictor_msg)
            print(diff_pixel_msg)

    def get_vision_tower_feature(self, images):
        features = self.get_model().get_vision_tower_mask()(images)
        return {
            "res2": features[0],  # bs, 128, 256, 256
            "res3": features[1],  # bs, 256, 128, 128
            "res4": features[2],  # bs, 512, 64, 64
            "res5": features[3],  # bs, 1024, 32, 32
        }

    def mask_decoder_training_init(self, cfg):

        self.size_divisibility = 32
        if cfg.MODEL.MASK_FORMER.SEG_TASK == "semantic":
            self.semantic_on = True
            self.instance_on = False
            self.panoptic_on = False
            self.referring_on = False
            self.region_on = False
            self.vis_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == "instance":
            self.semantic_on = False
            self.instance_on = True
            self.panoptic_on = False
            self.referring_on = False
            self.region_on = False
            self.vis_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == "panoptic":
            self.semantic_on = True
            self.instance_on = True
            self.panoptic_on = True
            self.referring_on = False
            self.region_on = False
            self.vis_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == "referring":
            self.semantic_on = False
            self.instance_on = False
            self.panoptic_on = False
            self.referring_on = True
            self.region_on = False
            self.vis_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == "region":
            self.semantic_on = False
            self.instance_on = False
            self.panoptic_on = False
            self.referring_on = False
            self.region_on = True
            self.vis_on = False
        elif cfg.MODEL.MASK_FORMER.SEG_TASK == "vis":
            self.semantic_on = False
            self.instance_on = False
            self.panoptic_on = False
            self.referring_on = False
            self.region_on = False
            self.vis_on = True
        else:
            raise NotImplementedError
        self.sem_seg_postprocess_before_inference = self.instance_on or self.panoptic_on or self.referring_on or self.region_on

    def SEG_instance_inference(self, SEG_cls, mask_pred, use_soft=False):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        scores = F.sigmoid(SEG_cls)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)

        mask_pred = mask_pred[topk_indices]

        result = Instances(image_size)
        if use_soft:
            clean_mask = (mask_pred > 0).float()
            clean_mask_counts = clean_mask.sum(dim=(1, 2))
            result.pred_masks = mask_pred.sigmoid()

        else:
            result.pred_masks = (mask_pred > 0).float()
            clean_mask = result.pred_masks
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * clean_mask.flatten(1)).sum(1) / (clean_mask.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        return result

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        seg_norm = cfg.MODEL.MASK_FORMER.SEG_NORM
        seg_proj = cfg.MODEL.MASK_FORMER.SEG_PROJ
        seg_fuse_score = cfg.MODEL.MASK_FORMER.FUSE_SCORE
        seg_concat = False
        print(f"current seg concat mode: {seg_concat}, seg_norm: {seg_norm}, seg_proj: {seg_proj}, seg_fuse_score: {seg_fuse_score}")
        predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(
            in_channels,
            hidden_dim,
            num_queries,
            nheads,
            dim_feedforward,
            dec_layers,
            pre_norm,
            mask_dim,
            enforce_input_project,
            seg_norm,
            seg_concat,
            seg_proj,
            seg_fuse_score,
        )
        return predictor

    def get_model(self):
        return self.model

    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES
        out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        num_features = [int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2**i) for i in range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))]
        out_feature_channels = {
            "res2": num_features[0],
            "res3": num_features[1],
            "res4": num_features[2],
            "res5": num_features[3],
        }
        backbone_feature_shape = dict()
        for name in out_features:
            backbone_feature_shape[name] = {"channel": out_feature_channels[name], "stride": out_feature_strides[name]}
        return backbone_feature_shape

    def get_encoder_image(self, images):
        return self.get_model().get_vision_tower()(images)

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ["res3", "res4", "res5"]

        return MSDeformAttnPixelDecoder(
            input_shape,
            transformer_dropout,
            transformer_nheads,
            transformer_dim_feedforward,
            transformer_enc_layers,
            conv_dim,
            mask_dim,
            transformer_in_features,
            common_stride,
        )

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        has_gt_ids = bool(hasattr(targets[0], "gt_ids"))
        for targets_per_image in targets:
            if has_gt_ids:
                inst_ids = targets_per_image.gt_ids
                valid_id = inst_ids != -1
            else:
                inst_ids = None
                valid_id = None
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": targets_per_image.gt_boxes,
                    "valid": valid_id,
                    "inst_id": inst_ids,
                    # "positive_map": positive_map,
                }
            )
        return new_targets

    def filter_valid_targets(self, det_targets, ref_targets):
        bz = len(det_targets)
        for i in range(bz):  # fliter empety object in key frame. Note that det_loss is only computed on the key frame !
            det_target = det_targets[i]
            ref_target = ref_targets[i]
            if False in det_target["valid"]:
                valid_i = det_target["valid"].clone()
                for k, v in det_target.items():
                    det_target[k] = v[valid_i]
                for k, v in ref_target.items():
                    ref_target[k] = v[valid_i]

        return det_targets, ref_targets

    def get_special_token(self, SEG, EOS):
        self.SEG_id = SEG
        self.EOS_id = EOS

    def get_class_name_embedding(self, hidden_states, cls_token_indices):
        """
        Function processes hidden states from a neural network and computes embeddings
        for class names based on specific token indices.

        hidden_states: 大模型中间隐藏层的内容
        cls_token_indices: 每个样本的类别标签
        return:
            class_name_embedding_list: 每个样本的类别名称的嵌入向量
        """
        class_name_embedding_list = []
        for current_hidden_state, current_token_indices in zip(hidden_states, cls_token_indices):
            class_id = torch.unique(current_token_indices)
            class_id = class_id[class_id != 0]
            current_class_name_embedding_list = []
            for id in class_id:
                current_class_mask = current_token_indices == id
                current_class_state = current_hidden_state[current_class_mask]
                current_class_name_embedding_list.append(current_class_state)
            current_pool_class_name_embedding = [
                self.class_name_pooling(class_name.transpose(-2, -1)).transpose(-2, -1) for class_name in current_class_name_embedding_list
            ]
            class_name_embedding_list.append(torch.cat(current_pool_class_name_embedding, dim=0))
        return torch.stack(class_name_embedding_list, dim=0)

    def embed_class_ids(self, class_name_ids, cls_indices):
        if class_name_ids is None:
            return None
        num_class = cls_indices.unique_consecutive()
        num_class = num_class[num_class >= 0]
        class_name_ids = [class_name_ids[cls_indices == idx] for idx in num_class]
        embedded_class_name = [self.get_model().embed_tokens(id) for id in class_name_ids]

        return embedded_class_name

    def embed_refer_ids(self, refer_ids):
        if refer_ids is None:
            return None
        embedded_refer = self.get_model().embed_tokens(refer_ids)
        return embedded_refer

    def concat_image_seg_cls_embeds(
        self,
        input_id,
        img_feature,
        label,
        seg_query,
        seg_query_mask,
        class_embed,
        class_name_embedding_indices,
        region_embedding_mask=None,
        region_feature_list=None,
        refer_embedding_indices=None,
        refer_embedding=None,
        temporal_query=None,
        temporal_query_mask=None,
    ):
        # Token定位
        image_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
        seg_query_indices = torch.where(input_id == SEG_TOKEN_INDEX)[0]
        cls_token_indices = torch.where(input_id == CLS_TOKEN_INDEX)[0]
        region_token_indices = torch.where(input_id == REGION_TOKEN_INDEX)[0]

        # 输入验证
        assert len(image_token_indices) == 1, "not supporting multi image index"
        # assert len(seg_query_indices) == 1, 'not supporting multi seg index'
        if class_name_embedding_indices is not None:
            assert len(cls_token_indices) == len(class_embed), "the number of <cls> tokens and class_embed needs to be same"
        if region_feature_list is not None:
            assert len(region_feature_list) == len(region_token_indices), "the munber of <region> tokens and regions needs to be same"

        # 特征拼接处理
        # 根据不同类型的 token（图像、分割、类别等）进行相应的特征拼接
        # 处理每个 token 对应的掩码和标签信息
        # cur_new_input_embeds = []        # 存储拼接后的输入嵌入
        # cur_new_seg_query_mask = []      # 存储分割查询掩码
        # cur_new_label = []               # 存储标签
        # cur_class_name_embedding_indices = [] # 存储类别名称嵌入索引
        cur_new_input_embeds = []
        cur_new_seg_query_mask = []
        if label is not None:
            cur_new_label = []
            assert label.shape == input_id.shape
        else:
            cur_new_label = None
        cur_class_name_embedding_indices = [] if class_name_embedding_indices is not None else None
        cur_refer_embedding_indices = [] if refer_embedding_indices is not None else None
        if region_embedding_mask is not None:
            enable_region_mask = True
            cur_new_region_embedding_mask = []
        else:
            enable_region_mask = False
            cur_new_region_embedding_mask = None

        if temporal_query is not None:
            enable_temporal_mask = True
            cur_new_temporal_query_mask = []
        else:
            enable_temporal_mask = False
            cur_new_temporal_query_mask = None

        # 分块处理
        chunks = []
        current_chunk = []
        for id in input_id:
            if id >= 0:
                current_chunk.append(id.item())
            else:
                if current_chunk:
                    chunks.append(torch.tensor(current_chunk, device=input_id.device))
                    current_chunk = []
                chunks.append([id])
        if current_chunk:
            chunks.append(torch.tensor(current_chunk, device=input_id.device))

        # 处理不同类型的Token
        cls_idx = 0
        region_idx = 0
        for chunk in chunks:
            chunk_len = len(chunk)
            if chunk_len == 1 and chunk[0] == IMAGE_TOKEN_INDEX:
                cur_new_input_embeds.append(img_feature)
                cur_new_seg_query_mask.append(torch.zeros(img_feature.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((img_feature.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                    )
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(torch.full((img_feature.shape[0],), 0, device=input_id.device, dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(torch.full((img_feature.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(img_feature.shape[0]))

                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(torch.zeros(img_feature.shape[0]))
            elif chunk_len == 1 and chunk[0] == SEG_TOKEN_INDEX:
                cur_new_input_embeds.append(seg_query)
                cur_new_seg_query_mask.append(torch.ones(seg_query.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((seg_query.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                    )
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(torch.full((seg_query.shape[0],), 0, device=input_id.device, dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(torch.full((seg_query.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(seg_query.shape[0]))
                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(torch.zeros(seg_query.shape[0]))

            elif chunk_len == 1 and chunk[0] == TEMPORAL_TOKEN_INDEX:
                cur_new_input_embeds.append(temporal_query)
                cur_new_temporal_query_mask.append(torch.ones(temporal_query.shape[0]))
                cur_new_seg_query_mask.append(torch.zeros(temporal_query.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((temporal_query.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                    )
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((temporal_query.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                    )
                if label is not None:
                    cur_new_label.append(torch.full((temporal_query.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(temporal_query.shape[0]))
            elif chunk_len == 1 and chunk[0] == CLS_TOKEN_INDEX:
                cls_embed = class_embed[cls_idx]
                if len(cls_embed.shape) == 1:
                    cls_embed = cls_embed.unsqueeze(0)
                cls_idx += 1
                cur_new_input_embeds.append(cls_embed)
                cur_new_seg_query_mask.append(torch.zeros(cls_embed.shape[0]))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(cls_embed.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((cls_embed.shape[0],), cls_idx, device=input_id.device, dtype=input_id.dtype)
                    )
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(torch.full((cls_embed.shape[0],), 0, device=input_id.device, dtype=input_id.dtype))
                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(torch.zeros(cls_embed.shape[0]))

                if label is not None:
                    cur_new_label.append(torch.full((cls_embed.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype))
            elif chunk_len == 1 and chunk[0] == REGION_TOKEN_INDEX:
                # region_feature_list: obj_num * point_num(1 or 256) * C(2560)
                region_feature = region_feature_list[region_idx]
                region_idx += 1
                cur_new_input_embeds.append(region_feature)
                cur_new_seg_query_mask.append(torch.zeros(region_feature.shape[0]))
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((region_feature.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                    )
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(
                        torch.full((region_feature.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                    )
                if label is not None:
                    cur_new_label.append(torch.full((region_feature.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype))
                if enable_region_mask:
                    if region_feature.shape[0] == 1:
                        cur_new_region_embedding_mask.append(torch.ones(region_feature.shape[0]))
                    else:
                        cur_new_region_embedding_mask.append(region_idx * torch.ones(region_feature.shape[0]))

                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(torch.zeros(region_feature.shape[0]))

            elif chunk_len == 1 and chunk[0] == REFER_TOKEN_INDEX:
                refer_embed = refer_embedding
                if len(refer_embed.shape) == 1:
                    refer_embed = refer_embed.unsqueeze(0)
                cur_new_input_embeds.append(refer_embed)
                cur_new_seg_query_mask.append(torch.zeros(refer_embed.shape[0]))
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(torch.zeros(refer_embed.shape[0]))

                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(
                        torch.full((refer_embed.shape[0],), 0, device=input_id.device, dtype=input_id.dtype)
                    )
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(torch.full((refer_embed.shape[0],), 1, device=input_id.device, dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(torch.full((refer_embed.shape[0],), IGNORE_INDEX, device=label.device, dtype=label.dtype))
                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(torch.zeros(refer_embed.shape[0]))

            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(input_id[:chunk_len]))
                cur_new_seg_query_mask.append(seg_query_mask[:chunk_len])
                if class_name_embedding_indices is not None:
                    cur_class_name_embedding_indices.append(class_name_embedding_indices[:chunk_len])
                if refer_embedding_indices is not None:
                    cur_refer_embedding_indices.append(refer_embedding_indices[:chunk_len])
                if label is not None:
                    cur_new_label.append(label[:chunk_len])
                if enable_region_mask:
                    cur_new_region_embedding_mask.append(region_embedding_mask[:chunk_len])
                if enable_temporal_mask:
                    cur_new_temporal_query_mask.append(temporal_query_mask[:chunk_len])

            input_id = input_id[chunk_len:]
            seg_query_mask = seg_query_mask[chunk_len:]
            if class_name_embedding_indices is not None:
                class_name_embedding_indices = class_name_embedding_indices[chunk_len:]
            if refer_embedding_indices is not None:
                refer_embedding_indices = refer_embedding_indices[chunk_len:]
            if label is not None:
                label = label[chunk_len:]
            if enable_region_mask:
                region_embedding_mask = region_embedding_mask[chunk_len:]
            if enable_temporal_mask:
                temporal_query_mask = temporal_query_mask[chunk_len:]

        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        if label is not None:
            cur_new_label = [x.to(device=self.device) for x in cur_new_label]
            cur_new_label = torch.cat(cur_new_label, dim=0)
        cur_new_seg_query_mask = [x.to(device=self.device) for x in cur_new_seg_query_mask]
        cur_new_seg_query_mask = torch.cat(cur_new_seg_query_mask, dim=0)
        if class_name_embedding_indices is not None:
            cur_class_name_embedding_indices = [x.to(device=self.device) for x in cur_class_name_embedding_indices]
            cur_class_name_embedding_indices = torch.cat(cur_class_name_embedding_indices, dim=0)
        if refer_embedding_indices is not None:
            cur_refer_embedding_indices = [x.to(device=self.device) for x in cur_refer_embedding_indices]
            cur_refer_embedding_indices = torch.cat(cur_refer_embedding_indices, dim=0)

        if enable_region_mask:
            cur_new_region_embedding_mask = [x.to(device=self.device) for x in cur_new_region_embedding_mask]
            cur_new_region_embedding_mask = torch.cat(cur_new_region_embedding_mask, dim=0)

        if enable_temporal_mask:
            cur_new_temporal_query_mask = [x.to(device=self.device) for x in cur_new_temporal_query_mask]
            cur_new_temporal_query_mask = torch.cat(cur_new_temporal_query_mask, dim=0)

        return (
            cur_new_input_embeds,  # 拼接后的输入嵌入
            cur_new_label,  # 处理后的标签
            cur_new_seg_query_mask,  # 分割查询掩码
            cur_new_temporal_query_mask,  # 时序查询掩码
            cur_class_name_embedding_indices,  # 类别名称嵌入索引
            cur_new_region_embedding_mask,  # 区域嵌入掩码
            cur_refer_embedding_indices,  # 引用嵌入索引
        )

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        class_name_embedding_indices=None,
        class_name_ids=None,
        cls_indices=None,
        instances=None,
        token_refer_id=None,
        refer_embedding_indices=None,
        expanded_seg_query=None,
        region_features=None,
        temporal_query=None,
    ):
        """
        数据预处理:
            - 处理输入的文本(input_ids)和图像(images)
            - 生成相应的嵌入向量(embeddings)和注意力掩码(attention mask)
            - 处理各种特殊类型的 token,如图像 token、分割 token、类别 token等

        特征提取与融合:
            - 通过视觉编码器提取图像特征
            - 提取区域特征(region features)用于实例分割
            - 将文本嵌入和图像特征进行拼接

        处理各种模态的标记:
            - 处理分割查询掩码(seg_query_mask)
            - 处理类别名称嵌入索引(class_name_embedding_indices)
            - 处理区域嵌入掩码(region_embedding_masks)
            - 处理引用嵌入索引(refer_embedding_indices)

        数据对齐:
            - 将不同长度的样本对齐到相同维度
            - 使用 padding 补齐较短的序列
            - 确保所有 batch 中的样本具有相同的维度
        """
        vision_tower = self.get_vision_tower()

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels, None, None, None, None

        seg_query_mask = torch.zeros_like(input_ids)

        image_features = self.encode_images(images[:, 0])

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0 and instances is not None:

            if region_features is None:
                if hasattr(instances[0][0], "ref_masks"):
                    image_features_ref = self.encode_images(images[:, 1])
                    region_masks_list = [instance[1].ref_masks for instance in instances]
                else:
                    image_features_ref = image_features
                    region_masks_list = [instance.region_masks for instance in instances]

                # [region_features_per_batch: [num_region, 1, dims]], len(region_features) = batch_size
                region_features = self.region_sampler(
                    image_features_ref, region_masks_list, original_dtype=image_features_ref.dtype, return_dtype=image_features_ref.dtype
                )

            region_embedding_masks = torch.zeros_like(input_ids)
        else:
            region_features = None
            region_embedding_masks = None

        # for video temporal_query
        if temporal_query is not None:
            temporal_query_mask = torch.zeros_like(input_ids)
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_seg_query_masks = []
        new_class_name_embedding_indices = [] if class_name_embedding_indices is not None else None
        new_refer_embedding_indices = [] if refer_embedding_indices is not None else None
        new_region_embedding_masks = [] if region_features is not None else None
        new_temporal_query_masks = [] if temporal_query is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_seg_query_mask = seg_query_mask[batch_idx]
            cur_temporal_query_mask = temporal_query_mask[batch_idx] if temporal_query is not None else None
            cur_seg_query = expanded_seg_query[batch_idx]
            cur_temporal_query = temporal_query[batch_idx] if temporal_query is not None else None
            cur_image_feature = image_features[batch_idx]
            cur_class_name_embedding_indices = class_name_embedding_indices[batch_idx] if class_name_embedding_indices is not None else None
            cur_refer_embedding_indices = refer_embedding_indices[batch_idx] if refer_embedding_indices is not None else None
            cur_region_feature_list = region_features[batch_idx] if region_features is not None else None
            cur_region_embedding_mask = region_embedding_masks[batch_idx] if region_features is not None else None
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                new_seg_query_masks.append(cur_seg_query_mask)
                # cur_image_idx += 1
                continue

            if labels is not None:
                cur_label = labels[batch_idx]
            else:
                cur_label = None

            if class_name_ids is not None:
                cur_class_name_ids = class_name_ids[batch_idx]
                cur_cls_indices = cls_indices[batch_idx]
            else:
                cur_class_name_ids = None
                cur_cls_indices = None
            if token_refer_id is not None:
                cur_token_refer_id = token_refer_id[batch_idx]
            else:
                cur_token_refer_id = None

            cur_class_name_embedding = self.embed_class_ids(cur_class_name_ids, cur_cls_indices)
            cur_refer_embedding = self.embed_refer_ids(cur_token_refer_id)

            (
                cur_input_embeds,
                cur_label,
                cur_seg_query_mask,
                cur_temporal_query_mask,
                cur_class_name_embedding_indices,
                cur_region_embedding_mask,
                cur_refer_embedding_indices,
            ) = self.concat_image_seg_cls_embeds(
                input_id=cur_input_ids,
                img_feature=cur_image_feature,
                label=cur_label,
                temporal_query=cur_temporal_query,
                temporal_query_mask=cur_temporal_query_mask,
                seg_query=cur_seg_query,
                seg_query_mask=cur_seg_query_mask,
                class_embed=cur_class_name_embedding,
                class_name_embedding_indices=cur_class_name_embedding_indices,
                region_embedding_mask=cur_region_embedding_mask,
                region_feature_list=cur_region_feature_list,
                refer_embedding_indices=cur_refer_embedding_indices,
                refer_embedding=cur_refer_embedding,
            )
            assert cur_input_embeds.shape[0] == cur_seg_query_mask.shape[0]

            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(cur_label)
            new_seg_query_masks.append(cur_seg_query_mask)
            if class_name_embedding_indices is not None:
                new_class_name_embedding_indices.append(cur_class_name_embedding_indices)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices.append(cur_refer_embedding_indices)
            if new_region_embedding_masks is not None:
                new_region_embedding_masks.append(cur_region_embedding_mask)
            if temporal_query is not None:
                new_temporal_query_masks.append(cur_temporal_query_mask)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            new_seg_query_masks_align = []
            for new_seg_query_mask in new_seg_query_masks:
                new_seg_query_mask = torch.cat(
                    (
                        new_seg_query_mask,
                        torch.zeros(
                            (max_len - new_seg_query_mask.shape[0]), dtype=new_seg_query_mask.dtype, device=new_seg_query_mask.device
                        ),
                    ),
                    dim=0,
                )
                new_seg_query_masks_align.append(new_seg_query_mask)
            new_seg_query_masks = torch.stack(new_seg_query_masks_align, dim=0)

            new_class_name_embedding_indices_align = []

            if class_name_embedding_indices is not None:
                for new_class_name_embedding_indice in new_class_name_embedding_indices:
                    new_class_name_embedding_indice = torch.cat(
                        (
                            new_class_name_embedding_indice,
                            torch.zeros(
                                (max_len - new_class_name_embedding_indice.shape[0]),
                                dtype=new_class_name_embedding_indice.dtype,
                                device=new_class_name_embedding_indice.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_class_name_embedding_indices_align.append(new_class_name_embedding_indice)
                new_class_name_embedding_indices = torch.stack(new_class_name_embedding_indices_align, dim=0)

            if refer_embedding_indices is not None:
                new_refer_embedding_indices_align = []
                for new_refer_embedding_indice in new_refer_embedding_indices:
                    new_refer_embedding_indice = torch.cat(
                        (
                            new_refer_embedding_indice,
                            torch.zeros(
                                (max_len - new_refer_embedding_indice.shape[0]),
                                dtype=new_refer_embedding_indice.dtype,
                                device=new_refer_embedding_indice.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_refer_embedding_indices_align.append(new_refer_embedding_indice)
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices_align, dim=0)

            if new_region_embedding_masks is not None:
                new_region_embedding_masks_align = []
                for new_region_embedding_mask in new_region_embedding_masks:
                    new_region_embedding_mask = torch.cat(
                        (
                            new_region_embedding_mask,
                            torch.zeros(
                                (max_len - new_region_embedding_mask.shape[0]),
                                dtype=new_region_embedding_mask.dtype,
                                device=new_region_embedding_mask.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_region_embedding_masks_align.append(new_region_embedding_mask)
                new_region_embedding_masks = torch.stack(new_region_embedding_masks_align, dim=0)

            if temporal_query is not None:
                new_temporal_query_masks_align = []
                for new_temporal_query_mask in new_temporal_query_masks:
                    new_temporal_query_mask = torch.cat(
                        (
                            new_temporal_query_mask,
                            torch.zeros(
                                (max_len - new_temporal_query_mask.shape[0]),
                                dtype=new_temporal_query_mask.dtype,
                                device=new_temporal_query_mask.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_temporal_query_masks_align.append(new_temporal_query_mask)
                new_temporal_query_masks = torch.stack(new_temporal_query_masks_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            new_seg_query_masks = torch.stack(new_seg_query_masks, dim=0)
            if class_name_embedding_indices is not None:
                new_class_name_embedding_indices = torch.stack(new_class_name_embedding_indices, dim=0)
            if refer_embedding_indices is not None:
                new_refer_embedding_indices = torch.stack(new_refer_embedding_indices, dim=0)

            if new_region_embedding_masks is not None:
                new_region_embedding_masks = torch.stack(new_region_embedding_masks, dim=0)

            if new_temporal_query_masks is not None:
                new_temporal_query_masks = torch.stack(new_temporal_query_masks, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        if temporal_query is not None:
            return (
                None,
                attention_mask,
                past_key_values,
                new_input_embeds,
                new_labels,
                new_seg_query_masks,
                new_class_name_embedding_indices,
                new_region_embedding_masks,
                new_refer_embedding_indices,
                new_temporal_query_masks,
            )

        return (
            None,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            new_seg_query_masks,
            new_class_name_embedding_indices,
            new_region_embedding_masks,
            new_refer_embedding_indices,
        )

    def mm_conv_prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images):
        """
        为多模态输入（文本和图像）准备输入数据和标签，以便输入到多模态模型中。
        它处理的是结合文本和图像的输入，并生成相应的嵌入（embeddings），标签（labels），以及注意力掩码（attention mask）。
        # 1. label -> embedding
        # 2. loss function
        """
        vision_tower = self.get_vision_tower()
        if not (vision_tower and images and input_ids.shape[1] > 1):
            if past_key_values is not None:
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat(list(images), dim=0)
            image_features = self.encode_images(concat_images)
            # 图像特征维度处理
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 对于包含图像的样本，会将图像特征和文本嵌入（包括图像令牌前后的文本）进行拼接。标签会根据需要进行处理，图像的标签会被标记为 IGNORE_INDEX
            # 对于不包含图像的样本，直接使用文本的嵌入（embed_tokens），并且保证其梯度不会被改变
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # 在模型中用于将输入的 input_ids（通常是词汇表中的索引）转换为对应的词嵌入（embeddings）向量的函数。
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                # 确保梯度反向传播，不更改 cur_input_embeds
                cur_input_embeds = cur_input_embeds + (0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            # concat text and image embedding. prepare labels, IGNORE_INDEX for image tokens
            while image_token_indices.numel() > 0:
                # 获取当前图像的特征
                cur_image_features = image_features[cur_image_idx]
                # 获取当前图像token的起始位置
                image_token_start = image_token_indices[0]

                # 处理公共部分
                cur_new_input_embeds.extend(
                    (
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start]),
                        cur_image_features,
                    )
                )

                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
                    )

                # 根据配置的不同，处理不同的逻辑
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
                    # 添加中间token并更新labels
                    cur_new_labels.append(cur_labels[image_token_start : image_token_start + 1])
                    cur_labels = cur_labels[image_token_start + 2 :]
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    # 没有额外的文本，只处理图像token
                    cur_labels = cur_labels[image_token_start + 1 :]
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]

                # 更新图像索引
                cur_image_idx += 1
                # 获取下一个图像token的索引
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            # 根据模型的配置，处理输入的 ID（cur_input_ids）并生成相应的嵌入（embeddings），同时可能还会处理标签（labels）
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # Align embedddings, labels, attn_mask from different sample into a batch
        # 主要处理输入的嵌入向量 (new_input_embeds)、标签 (new_labels) 和注意力掩码 (attention_mask) 的对齐与填充操作
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                # 对每个嵌入向量 cur_new_embed，如果其长度小于最大长度 max_len，就用零填充到相同的长度。
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)

            # 将所有对齐后的标签堆叠成一个新的张量 new_labels
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            # 对齐标签
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                # 将每个标签序列填充到最大长度 max_len，并使用一个特殊的值 IGNORE_INDEX 来填充新的标签位置。
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # 对齐注意力掩码
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            # 如果嵌入向量的形状已经一致，就直接将所有的嵌入向量堆叠成一个新的张量 new_input_embeds。
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def get_SEG_embedding(self, hidden_states, refer_embedding_indices):
        refer_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, refer_embedding_indices):
            current_refer_state = current_hidden_state[current_token_indice.bool()]
            current_pool_refer_state = self.refer_pooling(current_refer_state.transpose(-2, -1)).transpose(-2, -1)
            refer_embedding_list.append(current_pool_refer_state)
        return torch.stack(refer_embedding_list, dim=0)

    def encode_region_features(self, vp_region_masks, vp_images, return_all_point=False, gt_region_masks_list=None):

        vp_image_features = self.encode_images(vp_images)

        # [region_features_per_batch: [num_obj, 1, dims]], len(region_features) = batch_size
        region_features = self.region_sampler(
            vp_image_features,
            vp_region_masks,
            original_dtype=vp_image_features.dtype,
            return_dtype=vp_image_features.dtype,
            return_all_point=return_all_point,
            gt_region_masks_list=gt_region_masks_list,
        )
        return region_features

    def get_fvp_feat(self, cur_temporal_query, image_features):

        local_vision = []
        for i, local_img_features in enumerate(image_features.values()):
            if i == 0:
                continue  # del res2 feat
            i -= 1
            # 处理不同尺度的图像特征
            local_vision.append(self.local_project[i](local_img_features).flatten(2) + self.level_embed.weight[i][None, :, None])
            local_vision[-1] = local_vision[-1].permute(0, 2, 1)
            
        # 利用 mgvp_layers 处理时序查询和视觉特征
        cur_temporal_query = self.mgvp_layers(latents=cur_temporal_query, x=local_vision)
        # 投影到所需的维度空间
        cur_temporal_query = self.expanded_seg_query_project(cur_temporal_query)

        return cur_temporal_query

    def get_seg_query(self, hidden_states, seg_query_masks):
        seg_query_list = []
        for sample_hidden_state, sample_query_mask in zip(hidden_states, seg_query_masks):
            if torch.sum(sample_query_mask) == 0:
                continue

            unique_query_value = torch.unique(sample_query_mask)
            unique_query_value = unique_query_value[unique_query_value != 0]

            for value in unique_query_value:
                current_query_mask = sample_query_mask == value
                current_query = sample_hidden_state[current_query_mask]

                seg_query_list.append(current_query)

        seg_query = torch.stack(seg_query_list, dim=0)

        return seg_query

    def eval_seg(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_clip: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        seg_info=None,
        class_name_ids=None,
        class_name_embedding_indices=None,
        cls_indices=None,
        token_refer_id=None,
        refer_embedding_indices=None,
        padding_mask=None,
        use_soft=False,
        is_thing_list=None,
        use_temporal_query=False,
        cur_temporal_query=None,
    ):
        # 表示模型是否处于全景分割(panoptic segmentation)模式
        if self.panoptic_on:
            assert is_thing_list is not None, "is_thing_list need to be given"
            self.is_thing_list = is_thing_list

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if len(images.shape) < 5:
            # append T dim
            images = images.unsqueeze(1)
            images_clip = images_clip.unsqueeze(1)

        if (input_ids == REGION_TOKEN_INDEX).sum() != 0:
            instances = [i["instances"] for i in seg_info]
        else:
            instances = None

        # 获取图像编码特征(Vanilla Encoder)
        image_features = self.get_vision_tower_feature(images[:, 0])

        bs = input_ids.shape[0]
        # 100 * 2560 -->> bs * 100 * 2560
        expanded_seg_query = self.seg_query.unsqueeze(0).expand(bs, -1, -1)

        # 获取FVP特征
        if self.enable_mgvp_seg_query:
            # 扩展时序查询到批次大小
            if cur_temporal_query is None:
                cur_temporal_query = self.temporal_query.unsqueeze(0).expand(bs, -1, -1)  # sequential temporal_query between each frame
            cur_temporal_query = self.get_fvp_feat(cur_temporal_query, image_features)

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            seg_query_mask,
            class_name_embedding_indices,
            region_embedding_masks,
            refer_embedding_indices,
            cur_temporal_query_masks,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values,
            labels,
            images_clip,
            class_name_embedding_indices,
            class_name_ids,
            cls_indices,
            instances,
            token_refer_id,
            refer_embedding_indices,
            expanded_seg_query,
            temporal_query=cur_temporal_query,
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state

        if use_temporal_query and cur_temporal_query_masks is not None:
            cur_temporal_query = self.get_seg_query(hidden_states, cur_temporal_query_masks)
            cur_temporal_query = self.temporal_query_project(cur_temporal_query)

        seg_query = self.get_seg_query(hidden_states, seg_query_mask)
        # concat the original query
        # seg_query = torch.cat((seg_query, self.seg_query.unsqueeze(0).expand(bs, -1, -1)), dim=2)

        seg_query = self.seg_query_projector(seg_query)

        bs = seg_query.shape[0]

        if refer_embedding_indices is not None:
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices)
        else:
            SEG_embedding = None

        if class_name_embedding_indices is not None:
            class_name_embedding = self.get_class_name_embedding(hidden_states, class_name_embedding_indices)
        else:
            class_name_embedding = None

        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(image_features)

        if refer_embedding_indices is not None:
            SEG_embedding = self.SEG_token_projector(SEG_embedding)

        if class_name_embedding_indices is not None:
            class_name_embedding = self.class_name_projector(class_name_embedding)

        if region_embedding_masks is not None:
            region_embedding_list = self.get_region_embedding(hidden_states, region_embedding_masks)
            region_embedding_list = [self.region_projector(region_embedding) for region_embedding in region_embedding_list]
        else:
            region_embedding_list = None

        if self.vis_on:
            using_box = return_hs = True
            using_boxiou = False
        else:
            using_box = using_boxiou = return_hs = False

        mask_outputs = self.predictor(
            multi_scale_features,
            mask_features,
            None,
            seg_query,
            SEG_embedding,
            class_name_embedding,
            region_embedding_list,
            using_box=using_box,
            using_boxiou=using_boxiou,
            return_hs=return_hs,
        )

        mask_pred_results = mask_outputs["pred_masks"]
        images = [x[0] for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        SEG_cls_results = mask_outputs["pred_SEG_logits"]
        class_name_cls_results = mask_outputs["pred_class_name_logits"]
        region_cls_results = mask_outputs["pred_region_logits"]

        # mask_pred_results_0 = (mask_pred_results>0).float()
        # mask_pred_results_sum = mask_pred_results_0.sum(dim=(2,3))
        del mask_outputs
        processed_results = []
        if SEG_cls_results is None:
            SEG_cls_results = [None]
        if class_name_cls_results is None:
            class_name_cls_results = [None]
        for _seg_info, SEG_cls_result, class_name_cls_result, mask_pred_result, input_per_image, image_size in zip(
            seg_info, SEG_cls_results, class_name_cls_results, mask_pred_results, seg_info, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            if padding_mask is None:
                padding_mask = input_per_image.get("padding_mask")
            non_padding_indices = np.where(~np.array(padding_mask))
            min_y, max_y = np.min(non_padding_indices[0]), np.max(non_padding_indices[0])
            min_x, max_x = np.min(non_padding_indices[1]), np.max(non_padding_indices[1])
            original_height = max_y - min_y + 1
            original_width = max_x - min_x + 1
            processed_results.append({})
            # gt = _seg_info['instances'].gt_masks
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, [original_height, original_width], height, width
                )
                if SEG_cls_result is not None:
                    SEG_cls_result = SEG_cls_result.to(mask_pred_result)

            if self.referring_on:
                instance_r = retry_if_cuda_oom(self.SEG_instance_inference)(
                    SEG_cls_result.float(), mask_pred_result.float(), use_soft=use_soft
                )
                processed_results[-1]["instances"] = instance_r

            if use_temporal_query:
                if cur_temporal_query is not None:
                    return processed_results, cur_temporal_query

            return processed_results

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_clip: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        seg_info=None,
        class_name_ids=None,
        class_name_embedding_indices=None,
        cls_indices=None,
        random_idx=None,
        token_refer_id=None,
        refer_embedding_indices=None,
        global_step=None,
        dataset_type=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), f"this batch contain different dataset_type: {dataset_type}"
            batch_dataset_type = dataset_type[0]
            # print(batch_dataset_type)
        else:
            batch_dataset_type = []
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if images is not None and len(images.shape) < 5:
            # append T dim
            images = images.unsqueeze(1)
        if images_clip is not None and len(images_clip.shape) < 5:
            # append T dim
            images_clip = images_clip.unsqueeze(1)

        if "mm_conv" in batch_dataset_type:
            seg_query_mask = None
            class_name_embedding_indices = None
            region_embedding_masks = None
            SEG_token_indices = None
            cur_images_clip = images_clip[:, 0] if images_clip is not None else None
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.mm_conv_prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, cur_images_clip
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        """
            执行大模型的感知和预测部分
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        # print(logits.dtype)
        # print(self.lm_head.weight.dtype)

        if class_name_embedding_indices is not None:
            class_name_embedding = self.get_class_name_embedding(hidden_states, class_name_embedding_indices)  # bs 134 2560

        else:
            class_name_embedding = None

        if refer_embedding_indices is not None:
            SEG_embedding = self.get_SEG_embedding(hidden_states, refer_embedding_indices)
            # condition_embedding = SEG_embedding
        else:
            SEG_embedding = None

        # TODO: 作者这边损失函数代码还没公开
        loss = None
        if "mm_conv" in batch_dataset_type and labels is None:
            return CausalOutputWithMask(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def eval_vqa(
        self,
        do_sample=True,
        temperature=0.2,
        num_beams=1,
        max_new_tokens=128,
        eos_token_id=None,
        use_cache=True,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        images_clip: Optional[torch.FloatTensor] = None,
    ):

        output_ids = self.generate(
            input_ids=input_ids,
            images_clip=images_clip,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
        )

        return output_ids
