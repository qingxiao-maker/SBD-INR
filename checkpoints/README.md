# Checkpoint placement

Place the trained SBD-INR checkpoint for `data/kodim23.png` here and name it:

```text
kodim23_sbd_inr.pth
```

The expected relative path is therefore:

```text
checkpoints/kodim23_sbd_inr.pth
```

Use the `save_weight.pth` produced by the `ours` experiment with the following
configuration:

- 8 bit planes;
- hidden width 512;
- five hidden SIREN layers;
- 2D decoupled backbone enabled;
- FiLM-PBD enabled;
- image size 256.

Renaming the file does not modify its contents. The evaluation script expects a
checkpoint dictionary containing the `model_state_dict` key, matching the format
written by `lossless_bitplane_inr.py`.

