
NNODE=1
NUM_GPUS=8
#MASTER_NODE='SH-IDC1-10-140-1-1'##

torchrun  --nproc_per_node=${NUM_GPUS} evaluate.py --cfg-path lavis/projects/stgt/eval/lsmdc_ret_eval_pretrain_vig.yaml
