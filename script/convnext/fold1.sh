export CUDA_VISIBLE_DEVICES=0
# cd /data/zhaohaoyu/zxy/DS/mae
python main_train.py \
    --model_type convnext \
    --model convnext_tiny \
    --pretrained \
    --batch_size 64 \
    --epochs 50 \
    --data_path /data/zhaohaoyu/zxy/DS/outputs_5cv/fold_1 \
    --log_dir log/convnext_tiny/fold1 \
    --output_dir log/convnext_tiny/fold1 \