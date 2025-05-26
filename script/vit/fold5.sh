export CUDA_VISIBLE_DEVICES=4
python main_train.py \
    --model_type vit \
    --model vit_base_patch16 \
    --pretrained \
    --batch_size 64 \
    --epochs 50 \
    --data_path outputs_5cv/fold_5 \
    --log_dir log/vit_base/fold5 \
    --output_dir log/vit_base/fold5 \
    