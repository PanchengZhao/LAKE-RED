python main.py --base ldm/models/ldm/inpainting_big/config_LAKERED.yaml \
               --resume ldm/models/ldm/inpainting_big/LAKERED_init.ckpt \
               --stage 1 \
               -t \
               --gpus 0, \
               --logdir logs/LAKERED_Train \