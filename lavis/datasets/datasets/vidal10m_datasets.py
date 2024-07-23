"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset

import logging
logger = logging.getLogger(__name__)

class VIDAL10MDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos (e.g. msvd/videos/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        try:
            ann = self.annotation[index]
            vname = ann["video"]
            video_path = os.path.join(self.vis_root, vname)
            video = self.vis_processor(video_path)
            caption = self.text_processor(ann["caption"])
            c, t, h, w = video.shape
            if c !=3 or t==0:
                logger.warning(
                    f"Caught data error when loading video {video_path}, "
                    f"randomly sample a new video as replacement")
                index = random.randint(0, len(self) - 1)
                return self.__getitem__(index)
        except Exception as e:
            logger.warning(
                    f"Caught exception {e} when loading video {video_path}, "
                    f"randomly sample a new video as replacement")
            index = random.randint(0, len(self) - 1)
            return self.__getitem__(index)

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            # "image_id": self.img_ids[ann["image_id"]],
        }


if __name__ == "__main__":
    from torchvision import transforms

    def to_image_text_pair(sample):
        return sample[0], sample[1]["caption"]

    
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    normalize = transforms.Normalize(mean, std)

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
