NNODE=1
NUM_GPUS=8
#MASTER_NODE='SH-IDC1-10-140-1-1'##

torchrun  --nproc_per_node=${NUM_GPUS} train.py --cfg-path lavis/projects/alpro/train/msrvtt_qa_ft.yaml
