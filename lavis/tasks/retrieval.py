"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import os

import numpy as np
import torch
from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("retrieval")
class RetrievalTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls(cfg=run_cfg)

    def evaluation(self, model, data_loader, **kwargs):
        # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
        score_i2t, score_t2i = model.compute_sim_matrix(data_loader, task_cfg=self.cfg)

        if is_main_process():
            eval_result = self._report_metrics(
                score_i2t,
                score_t2i,
                data_loader.dataset.txt2img,
                data_loader.dataset.img2txt,
            )
            logging.info(eval_result)
        else:
            eval_result = None
        
        # with open("zret_result/video_infor.jsonl", 'a+') as f:
        #     json.dump(data_loader.dataset.struct, f, ensure_ascii=False)
        #     f.write('\n')
        
        # score_i2t = score_i2t + 700
        # score_i2t[score_i2t == -100] = 0.0
        
        
        # score_t2i = score_t2i + 700
        # score_t2i[score_t2i == -100] = 0.0
        
        # np.savetxt('zret_result/score_i2t.csv', score_i2t, fmt='%.3f', delimiter=',')
        # np.savetxt('zret_result/score_t2i.csv', score_t2i, fmt='%.3f', delimiter=',')

        # # debug by somos
        # with open("zret_result/txt2img.json", "w") as f:
        #     json.dump(data_loader.dataset.txt2img, f)
        # with open("zret_result/img2txt.json", "w") as f:
        #     json.dump(data_loader.dataset.img2txt, f)

        return eval_result

    def after_evaluation(self, val_result, **kwargs):
        return val_result

    @staticmethod
    @torch.no_grad()
    def _report_metrics(scores_i2t, scores_t2i, txt2img, img2txt):

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # with open("vis_json/msrvtt_R.txt", "a+") as f:
        #     struct = {}
        #     for th in range(1, 21):
        #         struct[f"txt_r{5*th}"]=  100.0 * len(np.where(ranks < 5*th)[0]) / len(ranks)
        #         struct[f"MedianR"]=  np.median(ranks) + 1
        #     f.write(str(struct)+"\n")

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # with open("vis_json/msrvtt_R.txt", "a+") as f:
        #     struct = {}
        #     for th in range(1, 21):
        #         struct[f"img_r{5*th}"]=  100.0 * len(np.where(ranks < 5*th)[0]) / len(ranks)
        #         struct[f"MedianR"]=  np.median(ranks) + 1
        #     f.write(str(struct)+"\n")


        MedianR = np.median(ranks) + 1
        # MeanR = np.mean(ranks) + 1

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (ir1 + ir5 + ir10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
            "MedianR": MedianR,
            # "MeanR": MeanR,
        }
        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(eval_result) + "\n")
        return eval_result
