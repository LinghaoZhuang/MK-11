export CUDA_VISIBLE_DEVICES=0
python main_train.py \
    --model_type convnext \
    --model convnext_tiny \
    --pretrained \
    --batch_size 64 \
    --epochs 50 \
    --data_path outputs_5cv/fold_1 \
    --log_dir log/convnext_tiny/fold1 \
    --output_dir log/convnext_tiny/fold1 \