export CUDA_VISIBLE_DEVICES=1
python main_train.py \
    --model_type vgg \
    --model vgg16 \
    --pretrained \
    --batch_size 64 \
    --epochs 50 \
    --data_path outputs_5cv/fold_2 \
    --log_dir log/vgg16/fold2 \
    --output_dir log/vgg16/fold2 \
    