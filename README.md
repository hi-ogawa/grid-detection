```
# Synthesize training data
python src/synthesize.py 5000 --out-dir data/synthesize/v1--5000

# Training
python src/main.py --data.directory data/synthesize/v1--5000

# Show log
tensorboard --logdir lightnight_logs
```
