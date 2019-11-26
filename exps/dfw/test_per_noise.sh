python $CS548_DIR/dfw/run.py test \
  --img_size 256 --msg_l 26 --device 0 --batch_size 256 \
  --noise_type no_noise --enc_scale 0.01 --dec_scale 1
  
python $CS548_DIR/dfw/run.py test \
  --img_size 256 --msg_l 26 --device 0 --batch_size 256 \
  --noise_type crop --enc_scale 0.01 --dec_scale 1

python $CS548_DIR/dfw/run.py test \
  --img_size 256 --msg_l 26 --device 0 --batch_size 256 \
  --noise_type cropout --enc_scale 0.01 --dec_scale 1
  
python $CS548_DIR/dfw/run.py test \
  --img_size 256 --msg_l 26 --device 0 --batch_size 256 \
  --noise_type dropout --enc_scale 0.01 --dec_scale 1
  
python $CS548_DIR/dfw/run.py test \
  --img_size 256 --msg_l 26 --device 0 --batch_size 256 \
  --noise_type resize --enc_scale 0.01 --dec_scale 1
  
python $CS548_DIR/dfw/run.py test \
  --img_size 256 --msg_l 26 --device 0 --batch_size 256 \
  --noise_type jpeg --enc_scale 0.01 --dec_scale 1

