# Teton assignment
To setup, activate and update environment, run
```
make setup
source .venv/bin/activate
make install
```

To classify `<video_path>`, run `python -m teton.videomae <video_path>`.
This uses an experimental dynamic crop.
Add `--square_crop --minimum_crop 1400` for a static crop.
