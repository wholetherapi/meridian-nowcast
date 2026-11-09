# Meridian-Nowcast

Reference implementation and evaluation harness for **Meridian-Nowcast**, a
reproducible benchmark for short-horizon solar and weather nowcasting.

## What this repository contains

- `src/meridian/dataset.py` - readers for the three public measurement networks
  used in the benchmark.
- `src/meridian/model.py` - the reference spatio-temporal encoder.
- `src/meridian/train.py` - training entry point.
- `src/meridian/evaluate.py` - scoring and reporting utilities.
- `data/manifest.json` - checksums and splits for the public datasets.
- `experiments/nowcast.yaml` - the exact configuration used for the paper results.

## Reproducing the results

```bash
pip install -r requirements.txt
python -m meridian.train --config experiments/nowcast.yaml
python -m meridian.evaluate --run-dir runs/default
```

## Results

The reference model reaches 0.284 MAE on the six-hour horizon split, matching
the numbers reported in the paper.

## License

Released under the MIT license.
