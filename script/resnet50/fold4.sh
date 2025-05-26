export CUDA_VISIBLE_DEVICES=3
cd /data/zhaohaoyu/zxy/DS/mae
python main_train.py \
    --model_type resnet \
    --model resnet50 \
    --pretrained \
    --batch_size 64 \
    --epochs 50 \
    --data_path /data/zhaohaoyu/zxy/DS/outputs_5cv/fold_4 \
    --log_dir log/resnet50/fold4 \
    --output_dir log/resnet50/fold4 \
    