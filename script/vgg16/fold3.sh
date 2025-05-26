export CUDA_VISIBLE_DEVICES=2
cd /data/zhaohaoyu/zxy/DS/mae
python main_train.py \
    --model_type vgg \
    --model vgg16 \
    --pretrained \
    --batch_size 64 \
    --epochs 50 \
    --data_path /data/zhaohaoyu/zxy/DS/outputs_5cv/fold_3 \
    --log_dir log/vgg16/fold3 \
    --output_dir log/vgg16/fold3 \
    