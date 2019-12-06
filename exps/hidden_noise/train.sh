python $CS548_DIR/hidden/run.py train \
  --img_size 256 --msg_l 31 --device 0 --batch_size 12 \
  --noise_type combined --enc_scale 1 --dec_scale 0.7 --adv_scale 0.001 --epochs 300 --save_freq 5 --test_freq 5
