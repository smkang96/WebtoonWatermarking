python $CS548_DIR/hidden_no_adv/run.py train \
  --img_size 256 --msg_l 31 --device 0 --batch_size 8 \
  --noise_type no_noise --enc_scale 1 --dec_scale 1 --epochs 50 --save_freq 1 --test_freq 1
