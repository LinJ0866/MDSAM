CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch  --nproc_per_node=1 --master-port=8989 train.py \
    --batch_size 8 \
    --num_workers 12 \
    --lr_rate 0.0005 \
    --data_path /home/linj/workspace/vsod/datasets/ \
    --sam_ckpt ./sam_vit_b_01ec64.pth \
    --dataset rdvs \
    --img_size 512 \
    --epoch 20

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch  --nproc_per_node=1 --master-port=8989 train.py \
    --batch_size 8 \
    --num_workers 12 \
    --lr_rate 0.0005 \
    --data_path /home/linj/workspace/vsod/datasets/ \
    --sam_ckpt ./sam_vit_b_01ec64.pth \
    --dataset vidsod_100 \
    --img_size 512 \
    --epoch 10

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch  --nproc_per_node=1 --master-port=8989 train.py \
    --batch_size 8 \
    --num_workers 12 \
    --lr_rate 0.0005 \
    --data_path /home/linj/workspace/vsod/datasets/ \
    --sam_ckpt ./sam_vit_b_01ec64.pth \
    --dataset dvisal \
    --img_size 512 \
    --epoch 10