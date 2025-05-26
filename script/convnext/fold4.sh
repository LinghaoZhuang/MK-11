export CUDA_VISIBLE_DEVICES=3   
python main_train.py \
    --model_type convnext \
    --model convnext_tiny \
    --pretrained \
    --batch_size 64 \
    --epochs 50 \
    --data_path outputs_5cv/fold_4 \
    --log_dir log/convnext_tiny/fold4 \
    --output_dir log/convnext_tiny/fold4 \
    