# Interface for accessing the YouTubeVIS dataset.

# The following API functions are defined:
#  YTVOS       - YTVOS api class that loads YouTubeVIS annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  loadRes    - Load algorithm results and create API for accessing them.

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import time
import sys
import logging
import random
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
from pycocotools import mask as maskUtils
import os
from collections import defaultdict

from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


PYTHON_VERSION = sys.version_info[0]


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class YTVOS:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.vids = {}, {}, {}, {}
        self.vidToAnns, self.catToVids = defaultdict(list), defaultdict(list)
        if annotation_file is not None:
            print("loading annotations into memory...")
            tic = time.time()
            dataset = json.load(open(annotation_file, "r"))
            assert type(dataset) == dict, f"annotation file format {type(dataset)} not supported"
            print("Done (t={:0.2f}s)".format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        anns, cats, vids = {}, {}, {}
        vidToAnns, catToVids = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset and self.dataset["annotations"] is not None:
            for ann in self.dataset["annotations"]:
                vidToAnns[ann["video_id"]].append(ann)
                anns[ann["id"]] = ann

        if "videos" in self.dataset:
            for vid in self.dataset["videos"]:
                vids[vid["id"]] = vid

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset and self.dataset["annotations"] is not None:
            for ann in self.dataset["annotations"]:
                catToVids[ann["category_id"]].append(ann["video_id"])

        print("index created!")

        # create class members
        self.anns = anns
        self.vidToAnns = vidToAnns
        self.catToVids = catToVids
        self.vids = vids
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset["info"].items():
            print(f"{key}: {value}")

    def getAnnIds(self, vidIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param vidIds  (int array)     : get anns for given vids
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(vidIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            if len(vidIds) != 0:
                lists = [self.vidToAnns[vidId] for vidId in vidIds if vidId in self.vidToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann["category_id"] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann["avg_area"] > areaRng[0] and ann["avg_area"] < areaRng[1]]
        return [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd] if iscrowd is not None else [ann["id"] for ann in anns]

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        cats = self.dataset["categories"]
        if not len(catNms) == len(supNms) == len(catIds) == 0:
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat["name"] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat["supercategory"] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat["id"] in catIds]
        return [cat["id"] for cat in cats]

    def getVidIds(self, vidIds=[], catIds=[]):
        """
        Get vid ids that satisfy given filter conditions.
        :param vidIds (int array) : get vids for given ids
        :param catIds (int array) : get vids with all given cats
        :return: ids (int array)  : integer array of vid ids
        """
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(vidIds) == len(catIds) == 0:
            ids = self.vids.keys()
        else:
            ids = set(vidIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToVids[catId])
                else:
                    ids &= set(self.catToVids[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadVids(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying vid
        :return: vids (object array) : loaded vid objects
        """
        if _isArrayLike(ids):
            return [self.vids[id] for id in ids]
        elif type(ids) == int:
            return [self.vids[ids]]

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = YTVOS()
        res.dataset["videos"] = list(self.dataset["videos"])

        print("Loading and preparing results...")
        tic = time.time()
        if type(resFile) == str:  # or type(resFile) == unicode:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, "results in not an array of objects"
        annsVidIds = [ann["video_id"] for ann in anns]
        assert set(annsVidIds) == (set(annsVidIds) & set(self.getVidIds())), "Results do not correspond to current coco set"
        if "segmentations" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                ann["areas"] = []
                if "bboxes" not in ann:
                    ann["bboxes"] = []
                for seg in ann["segmentations"]:
                    # now only support compressed RLE format as segmentation results
                    if seg:
                        ann["areas"].append(maskUtils.area(seg))
                        if len(ann["bboxes"]) < len(ann["areas"]):
                            ann["bboxes"].append(maskUtils.toBbox(seg))
                    else:
                        ann["areas"].append(None)
                        if len(ann["bboxes"]) < len(ann["areas"]):
                            ann["bboxes"].append(None)
                ann["id"] = id + 1
                l = [a for a in ann["areas"] if a]
                ann["avg_area"] = np.array(l).mean() if l else 0
                ann["iscrowd"] = 0
        print("DONE (t={:0.2f}s)".format(time.time() - tic))

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    def annToRLE(self, ann, frameId):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.vids[ann["video_id"]]
        h, w = t["height"], t["width"]
        segm = ann["segmentations"][frameId]
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            return maskUtils.merge(rles)
        elif type(segm["counts"]) == list:
            # uncompressed RLE
            return maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            return segm

    def annToMask(self, ann, frameId):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, frameId)
        return maskUtils.decode(rle)


def load_refytvos_json(json_file, image_path_yv="", image_path_davis="", has_mask=True, vos=True, is_train=True):

    with open(json_file) as f:
        data = json.load(f)

    dataset_dicts = []

    if not is_train:
        for vid_dict in data["videos"]:
            record = {
                "file_names": [os.path.join(image_path_yv, vid_dict["file_names"][i]) for i in range(vid_dict["length"])],
                "height": vid_dict["height"],
                "width": vid_dict["width"],
                "length": vid_dict["length"],
                "video_id": vid_dict["id"],
                "video": vid_dict["video"],
                "exp_id": vid_dict.get("exp_id"),
                "expressions": vid_dict["expressions"],
                "has_mask": has_mask,
                "task": "rvos",
                "dataset_name": "rvos",
            }
            dataset_dicts.append(record)
        return dataset_dicts

    ann_keys = ["iscrowd", "category_id", "id"]
    num_instances_without_valid_segmentation = 0
    non_mask_count = 0

    for vid_dict, anno_dict_list in zip(data["videos"], data["annotations"]):
        assert vid_dict["id"] == anno_dict_list["video_id"]
        record = {}
        if "davis" in vid_dict["file_names"][0]:
            dataset_name = "davis"
            record["file_names"] = [
                os.path.join(image_path_davis, vid_dict["file_names"][i].replace("davis", "")) for i in range(vid_dict["length"])
            ]
        elif "youtube" in vid_dict["file_names"][0]:
            dataset_name = "youtube"
            record["file_names"] = [
                os.path.join(image_path_yv, vid_dict["file_names"][i].replace("youtube", "")) for i in range(vid_dict["length"])
            ]
        else:
            dataset_name = "youtube"
            record["file_names"] = [os.path.join(image_path_yv, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]

        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        record["expressions"] = vid_dict["expressions"]

        video_objs = []
        for frame_idx in range(record["length"]):
            anno = anno_dict_list
            obj = {key: anno[key] for key in ann_keys if key in anno}
            _bboxes = anno.get("bboxes", None)
            _segm = anno.get("segmentations", None)
            if has_mask:
                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    non_mask_count += 1
                    continue
            else:
                if not (_bboxes and _bboxes[frame_idx]):
                    continue

            bbox = _bboxes[frame_idx]
            obj["bbox"] = bbox
            obj["bbox_mode"] = BoxMode.XYWH_ABS

            if has_mask:
                segm = _segm[frame_idx]
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = maskUtils.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if not segm:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            frame_objs = [obj]
            video_objs.append(frame_objs)
        record["annotations"] = video_objs

        record["has_mask"] = has_mask

        record["task"] = "rvos"
        record["dataset_name"] = dataset_name
        dataset_dicts.append(record)

    return dataset_dicts


def load_revos_json(revos_path, is_train=True):

    if is_train:
        json_file = os.path.join(revos_path, "meta_expressions_train_.json")
    else:
        json_file = os.path.join(revos_path, "meta_expressions_valid_.json")

    mask_json = os.path.join(revos_path, "mask_dict.json")
    with open(mask_json) as fp:
        mask_dict = json.load(fp)

    with open(json_file) as f:
        meta_expressions = json.load(f)["videos"]  # {'video1', 'video2', ...}

    video_list = list(meta_expressions.keys())

    dataset_dicts = []

    for vid_ in video_list:
        vid_dict = meta_expressions[vid_]
        video_path = os.path.join(revos_path, vid_)

        record = {
            "file_names": [os.path.join(video_path, f"{frame}.jpg") for frame in vid_dict["frames"]],
            "height": vid_dict["height"],
            "width": vid_dict["width"],
            "length": len(vid_dict["frames"]),
            "video_id": vid_dict["vid_id"],
            "video": vid_,
            "expressions": (vid_dict["expressions"].values() if is_train else vid_dict["expressions"]),
            "task": "revos",
            "dataset_name": "revos",
        }

        if is_train:
            dataset_dicts.append(record)
        else:
            for exp in record["expressions"]:
                record["exp_id"] = exp
                dataset_dicts.append(record)
    return dataset_dicts


def filter_empty_instances_soft(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1  # invalid instances are marked with -1
    return instances
