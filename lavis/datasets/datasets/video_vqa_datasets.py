"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import random
from collections import OrderedDict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)

import logging
logger = logging.getLogger(__name__)

class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )


class VideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def _build_class_labels(self, ans_path):
        ans2label = json.load(open(ans_path))

        self.class_labels = ans2label

    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."

        try:
            ann = self.annotation[index]

            vname = ann["video"]
            vpath = os.path.join(self.vis_root, vname)

            frms = self.vis_processor(vpath)
            question = self.text_processor(ann["question"])
            c, t, h, w = frms.shape
            if c !=3 or t==0:
                logger.warning(
                    f"Caught data error when loading video {vpath}, "
                    f"randomly sample a new video as replacement")
                index = random.randint(0, len(self) - 1)
                return self.__getitem__(index)
        except Exception as e:
            logger.warning(
                    f"Caught exception {e} when loading video {vpath}, "
                    f"randomly sample a new video as replacement")
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)
        
        return {
            "video": frms,
            "text_input": question,
            "answers": self._get_answer_label(ann["answer"]),
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
