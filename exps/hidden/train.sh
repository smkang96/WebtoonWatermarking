python $CS548_DIR/hidden/run.py train \
  --img_size 256 --msg_l 31 --device 0 --batch_size 8 \
  --noise_type no_noise --enc_scale 1 --dec_scale 1 --adv_scale 0.001 --epochs 100 --save_freq 1 --test_freq 1
