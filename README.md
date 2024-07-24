# Video-Language Alignment via Spatio–Temporal Graph Transformer (STGT)
Video-Language Alignment via Spatio–Temporal Graph Transformer； https://arxiv.org/abs/2407.11677 

![](https://github.com/GXYM/STGT/blob/main/framework.png)  

## 1. ToDo

- [x] Release eval code
- [x] Release scripts for testing
- [x] Release pre-trained models
- [x] Release fine-tune models on each benchmarks
- [ ] Release train codes
- [ ] Release train scripts

NOTE：After the paper is accepted, we will open source the code of train.

## 2. Prepare Dataset   
1. Download the corresponding video data through the [download_scripts.](https://github.com/GXYM/TextBPN/blob/main/vis/1.png)  we have collected.

## 3. Environment
 * 1. We provide environment dependencies, first install [requirements_all.txt](https://github.com/GXYM/STGT/blob/main/requirements_all.txt), then install [requirements_all-1.txt](https://github.com/GXYM/STGT/blob/main/requirements_all-1.txt)
```
pip install -r requirements_all.txt
pip install -r requirements_all-1.txt
```
 *  2. You can also run the [pip_install.sh](https://github.com/GXYM/STGT/blob/main/pip_install.sh) directly
```
sh pip_install.sh
```

## 4. DownLoad Models
The model link have been shared herer.

|         Data    |  W10M+VIDAL4M-256|W10M+VIDAL4M-1024 | W10M+VIDAL7M-256 |Extracted Code|
|:------------------:	|:-----------:  |:-----------:	|:-------:|:-------:|
| Pre-training Models |  [checkpoint_w10m_v4m_256.pth](https://pan.baidu.com/s/1eB7-ViWPf1l9gdDhkYXFsQ) | [checkpoint_w10m_v4m_1024.pth](https://pan.baidu.com/s/1jP9rLMyyZ2mteD7kwu1irw) 	| [checkpoint_w10m_v7m_256.pth](https://pan.baidu.com/s/1afl0BzUzhkbn_P3eSIF8TQ) |gxym|

|         Dataset   |  model-1| model-2 |Extracted Code|
|:------------------:	|:-----------:	|:-------:|:-------:|
| didemo_ret| [checkpoint_best_w10m_v4m_1024.pth](https://pan.baidu.com/s/1yezEntt8w0rQVG99jy12JA)| [checkpoint_best_w10m_v7m_256.pth](https://pan.baidu.com/s/1yezEntt8w0rQVG99jy12JA)|gxym|
| lsmdc_ret | [checkpoint_best_w10m_v4m_1024.pth](https://pan.baidu.com/s/19zdiscvvoeeJjZ9v5zMIrg)| [checkpoint_best_w10m_v4m_256.pth](https://pan.baidu.com/s/19zdiscvvoeeJjZ9v5zMIrg)|gxym|
| msrvtt_reT| [checkpoint_best_w10m_v4m_1024.pth](https://pan.baidu.com/s/1NC7vGWW5hkwP8V72Fwpxig)| [checkpoint_best_w10m_v7m_256.pth](https://pan.baidu.com/s/1NC7vGWW5hkwP8V72Fwpxig)|gxym|
| msvd_ret  | [checkpoint_best_w10m_v4m_1024.pth](https://pan.baidu.com/s/18QUC_gUMleswxymVKR-zSA)| [checkpoint_best_w10m_v7m_256.pth](https://pan.baidu.com/s/18QUC_gUMleswxymVKR-zSA)|gxym|
  
CLIP VIT pretrained models are [here](https://pan.baidu.com/s/13ITPJF2HFjep06BosK7E4w)

## 5.Eval and Testing

You can find the corresponding evaluation script [here](https://github.com/GXYM/STGT/tree/main/run_scripts/stgt/eval), configure the model path, and run it directly.  
```
DiDemo:  eval_didemo_ret_pretrain_vig.sh
LSMDC:   eval_lsmdc_ret_pretrain_vig.sh
MSRVTT:  eval_msrvtt_ret_pretrain_vig.sh
MSVD:    eval_msvd_ret_pretrain_vig.sh
```

We also provide an [ALPRO](https://github.com/GXYM/STGT/tree/main/run_scripts/alpro) evaluation scripts, and you can download its model for comparative testing.  

NOTE: Due to the desensitization process of the code, we cannot guarantee that there are no bugs in the code, but we will promptly fix these bugs.
 ## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/GXYM/DRRG/blob/master/LICENSE.md) file for details


