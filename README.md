# STGT
Video-Language Alignment via Spatio–Temporal Graph Transformer； https://arxiv.org/abs/2407.11677 
![](https://github.com/GXYM/TextBPN/blob/main/vis/1.png)  

## 1. ToDo List

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
The model link will be shared as soon as possible, but it will take some time.

## 5.Eval and Testing

You can find the corresponding evaluation script [here](https://github.com/GXYM/STGT/tree/main/run_scripts/stgt/eval), configure the model path, and run it directly.  
```
DiDemo:  eval_didemo_ret_pretrain_vig.sh
LSMDC:   eval_lsmdc_ret_pretrain_vig.sh
MSRVTT:  eval_msrvtt_ret_pretrain_vig.sh
MSVD:    eval_msvd_ret_pretrain_vig.sh
```


![](https://github.com/GXYM/TextBPN/blob/main/vis/2.png)
![](https://github.com/GXYM/TextBPN/blob/main/vis/3.png)


 ## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/GXYM/DRRG/blob/master/LICENSE.md) file for details


