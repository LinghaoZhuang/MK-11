export CUDA_VISIBLE_DEVICES=0
python main_train.py \
    --model_type resnet \
    --model resnet50 \
    --pretrained \
    --batch_size 64 \
    --epochs 50 \
    --data_path outputs_5cv/fold_1 \
    --log_dir log/resnet50/fold1 \
    --output_dir log/resnet50/fold1 \
    