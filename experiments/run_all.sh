#!/usr/bin/env bash
set -euo pipefail
python -m meridian.train --config experiments/nowcast.yaml
python -m meridian.evaluate --run-dir runs/default
