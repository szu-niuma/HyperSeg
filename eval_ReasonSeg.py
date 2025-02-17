import argparse
import torch
import torch.distributed as distributed
import os
from enum import Enum
import json
from tqdm import tqdm
import shortuuid
import numpy as np

import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipImageProcessor
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
import pickle
from pathlib import Path


from hyperseg.utils.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_SEG_TOKEN,
    SEG_TOKEN_INDEX,
    CLS_TOKEN_INDEX,
)
from hyperseg.utils.builder import load_pretrained_model

from hyperseg.utils import conversation as conversation_lib

from hyperseg.eval.eval_dataset.eval_datasets import DataCollatorForCOCODatasetV2, Reason_dataset_test


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    if output.device != target.device:
        # print(output.device)
        # print(target.device)
        output = output.to(target.device)

    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def parse_outputs(outputs, gt_mask):

    res_list = []
    for output in outputs:
        # gt = output['gt'].cpu().numpy().astype(np.uint8)

        pred_mask = output["instances"].pred_masks
        # pred_mask = pred_mask.cpu().numpy()
        scores = output["instances"].scores  # .cpu().numpy()
        try:
            pred_cls = output["instances"].pred_classes  # .cpu().numpy()
        except:
            pred_cls = None

        res = {"pred": pred_mask, "gt": gt_mask, "scores": scores, "pred_cls": pred_cls}
        res_list.append(res)
    return res_list


def compute_metric(intersection_meter, union_meter, acc_iou_meter, gt_cls, results_list):
    pred_list = []
    gt_list = []
    results_list = list(results_list)
    for results in results_list:
        gt = results["gt"]
        preds = results["pred"]
        scores = results["scores"]
        # preds = preds.astype(np.uint8)
        # pick mask with maximum score
        topk_scores, idx = torch.topk(scores, 1)
        # idx = idx.cpu().numpy()
        topk_preds = preds[idx, :]
        if results["pred_cls"] is not None:
            topk_pred_cls = results["pred_cls"][idx]
        max_acc_iou = -1
        max_iou = 0
        max_intersection = 0
        max_union = 0
        max_i = 0
        # here topk=1, len(topk_preds)=1
        for i, pred_ in enumerate(topk_preds):
            intersection, union, _ = intersectionAndUnionGPU(pred_.int().contiguous().clone(), gt.int().contiguous(), 2, ignore_index=255)
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0  # no-object target
            fore_acc_iou = acc_iou[1]
            if fore_acc_iou > max_acc_iou:
                max_acc_iou = fore_acc_iou
                max_iou = acc_iou
                max_intersection = intersection
                max_union = union
                max_i = i
        intersection_meter.update(max_intersection)
        union_meter.update(max_union)
        acc_iou_meter.update(max_iou, n=1)
        pred_list.append(topk_preds[max_i])
        gt_list.append(gt)

    return pred_list, gt_list


@dataclass
class DataArguments:

    local_rank: int = 0

    lora_enable: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    vision_tower: str = "google/siglip-so400m-patch14-384"
    vision_tower_mask: str = "./pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"

    lazy_preprocess: bool = False
    is_multimodal: bool = False
    # model_path: Optional[str] = field(default="../model/HyperSeg-3B")
    model_path: Optional[str] = field(default="zhumj34/Mipha-3B")
    mask_config: Optional[str] = field(default="./hyperseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)

    model_map_name: str = "HyperSeg"
    version: str = "llava_phi"
    output_dir: str = "./output/reasonseg"
    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 8
    seg_task: Optional[str] = field(default="referring")

    sum_ref_pred_answer: bool = False

    # enable FVP module in paper
    enable_mgvp_seg_query: bool = field(default=True)

    visualize: bool = True

    # reason seg
    reason_path: str = "./dataset/ReasonSeg"
    reason_seg_data: str = "ReasonSeg|val"
    explanatory: float = -1


def init_distributed_mode(para):
    para.distributed = True
    if torch.cuda.device_count() <= 1:
        para.distributed = False
        para.local_rank = 0
        para.world_size = 1

    if para.distributed:
        # Init distributed environment
        distributed.init_process_group(backend="nccl")

        local_rank = distributed.get_rank()
        world_size = distributed.get_world_size()
        torch.cuda.set_device(local_rank)
        print("I am rank %d in this world of size %d!" % (local_rank, world_size))
        para.local_rank = local_rank
        para.world_size = world_size


def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    init_distributed_mode(data_args)
    # model_path = os.path.expanduser(data_args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        data_args.model_path,
        model_args=data_args,
        mask_config=data_args.mask_config,
        device="cuda",
    )

    device = torch.device(data_args.local_rank if torch.cuda.is_available() else "cpu")
    # model.to(dtype=torch.float32, device=device)

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
    data_collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)

    if data_args.visualize:
        os.makedirs(os.path.join(data_args.output_dir, "vis"), exist_ok=True)

    val_data = data_args.reason_seg_data

    save_suffix = val_data

    if data_args.local_rank == 0:
        print(f"------cur reason_seg benchmark is {save_suffix} -------")

    eval_dataset = Reason_dataset_test(reason_path=data_args.reason_path, tokenizer=tokenizer, data_args=data_args)
    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    if not data_args.distributed:
        val_sampler = None
    else:
        val_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=dataloader_params["batch_size"],
        shuffle=False,
        num_workers=dataloader_params["num_workers"],
        pin_memory=False,
        sampler=val_sampler,
        collate_fn=data_collator,
    )

    def load_ref_dataset():
        return Reason_dataset_test(reason_path=data_args.reason_path, tokenizer=tokenizer, data_args=data_args)

    DatasetCatalog.register(save_suffix, load_ref_dataset)
    MetadataCatalog.get(save_suffix).set(
        stuff_classes=["object"],
    )

    do_eval(model, eval_dataloader, save_suffix, data_args, device)


def do_eval(model, eval_dataloader, save_suffix, data_args, device):

    model.eval()
    save_list = []
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            # torch.cuda.empty_cache()

            image_path = inputs["seg_info"][0]["file_name"]
            refer_text = inputs["seg_info"][0]["sentences"][0]
            gt_mask = inputs["seg_info"][0]["instances"].gt_masks[0].numpy().astype(np.uint8)  # .to(device)
            gt_mask = torch.tensor(gt_mask, device=device)

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            inputs["token_refer_id"] = [ids.to(device) for ids in inputs["token_refer_id"]]
            outputs = model.eval_seg(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=inputs["images"].float(),
                images_clip=inputs["images_clip"].float(),
                seg_info=inputs["seg_info"],
                token_refer_id=inputs["token_refer_id"],
                refer_embedding_indices=inputs["refer_embedding_indices"],
                labels=inputs["labels"],
            )
            gt_cls = inputs["seg_info"][0]["instances"].gt_classes

            cur_res = parse_outputs(outputs, gt_mask)
            torch.cuda.synchronize()
            pred, gt_mask = compute_metric(intersection_meter, union_meter, acc_iou_meter, gt_cls, cur_res)
            # save_list.append({'pred':pred[0],'gt':gt_mask[0],'name':inputs['seg_info'][0]['file_name']})

            if data_args.visualize:

                pred_mask = pred[0]
                image_np = cv2.imread(image_path)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

                pred_mask = pred_mask.detach().cpu().numpy()
                pred_mask = pred_mask > 0

                save_img = image_np.copy()
                save_img[pred_mask] = (image_np * 0.5 + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask]
                save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

                fn = os.path.split(image_path)[-1]
                save_path = os.path.join(data_args.output_dir, "vis", refer_text + fn)
                cv2.imwrite(save_path, save_img)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    msg = "benchmark: {}: giou: {:.4f}, ciou: {:.4f}".format(save_suffix, giou, ciou)
    if data_args.local_rank == 0:
        print(msg)
    save_path = os.path.join(data_args.output_dir, "pred_pkl")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_path, f"pred_{save_suffix}.txt"), "w") as f:
        f.write(msg)


if __name__ == "__main__":
    evaluation()
