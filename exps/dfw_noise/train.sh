python $CS548_DIR/dfw/run.py train \
  --img_size 256 --msg_l 31 --device 0 --test_device 0 --batch_size 16 \
  --noise_type combined --enc_scale 0.01 --dec_scale 1 --epochs 400 --annealing_epochs 200 --save_freq 1 --test_freq 1 
