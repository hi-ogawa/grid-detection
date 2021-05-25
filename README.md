Example https://colab.research.google.com/drive/1E9Yi4mlpYn7l9gBs3V3gp7zFVLejGXJy

```
pip install -r requirements.txt

# Synthesize training data
python -m src.synthesize 20000 --out-dir data/synthesize/v3--20000

# Merge synthesized data
python -m src.synthesize_merge --in-dirs data/synthesize/xxx data/synthesize/yyy --out-dir data/synthesize/zzz

# Training
python -m src.main \
  --data.directory data/synthesize/v3--20000 \
  --data.num_workers 2 \
  --data.batch_size 32 \
  --model.hidden_size 1024 \
  --model.hidden_depth 2 \
  --model.augmentation false \
  --model.lr 0.0001 \
  --model.lr_type 2 \
  --model.lr_step 20 \
  --model.lr_gamma 0.4 \
  --trainer.gpus 1 \
  --trainer.max_epochs 100

# Show log
tensorboard --logdir lightnight_logs
```
