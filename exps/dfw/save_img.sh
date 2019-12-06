python $CS548_DIR/dfw/run.py save_img \
  --img_size 256 --msg_l 26 --device 0 --noise_type no_noise --n_imgs 10

python $CS548_DIR/dfw/run.py save_img \
  --img_size 256 --msg_l 26 --device 0 --noise_type crop --n_imgs 10

python $CS548_DIR/dfw/run.py save_img \
  --img_size 256 --msg_l 26 --device 0 --noise_type cropout --n_imgs 10

python $CS548_DIR/dfw/run.py save_img \
  --img_size 256 --msg_l 26 --device 0 --noise_type dropout --n_imgs 10

python $CS548_DIR/dfw/run.py save_img \
  --img_size 256 --msg_l 26 --device 0 --noise_type resize --n_imgs 10

python $CS548_DIR/dfw/run.py save_img \
  --img_size 256 --msg_l 26 --device 0 --noise_type jpeg --n_imgs 10