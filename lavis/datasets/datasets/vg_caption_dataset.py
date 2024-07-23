import os
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset


class VGCaptionDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_id"].split("_")[-1]) + '.jpg'
        # image_path = os.path.join(self.vis_root, ann["image"])
        # image = Image.open(image_path).convert("RGB")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as ee:
            print('=======file not found:',ee)
            return self.__getitem__(index-1)


        image = self.vis_processor(image)
        caption = ann["caption"]

        # img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

        return {
            "image": image,
            "text_input": caption
        }