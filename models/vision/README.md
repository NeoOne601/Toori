# Desktop Vision Assets

This directory stores local desktop inference assets for Toori.

- `mobilenetv2-12.onnx`
  Primary lightweight ONNX image model used by the desktop runtime.
- `imagenet-simple-labels.json`
  Label list used to turn the top ONNX class index into a readable tag.

Fetch the assets with:

```bash
python3.11 scripts/download_desktop_models.py
```
