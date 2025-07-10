# Physics-Informed Neural Networks for Traffic Flow Prediction

This repository contains the code for the paper "Investigating Knowledge Transfer in Residual Physical Informed Neural Networks using Connected Vehicles Traffic Data". It provides an implementation of a Physics-Informed Neural Network (PINN) to model traffic flow dynamics, specifically using the US101 dataset.

## Project Structure

```
github_code/
├── configs/
│   └── us101_... .json   # Configuration files for the US101 dataset
├── data/
│   ├── pinn_data_us101_norm.csv.gz # Preprocessed and normalized US101 data
│   └── pinn_scalers_us101.pk       # Scalers used for data normalization
├── main.py                         # Main script to run the training and evaluation
├── quality_metrics.py              # Quality metrics
├── readme.md                       # This file
└── utils.py                        # Utility functions
```

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- pandas
- scikit-learn
- numpy

You can install the dependencies using pip:
```bash
pip install -r requriments.txt
```

### How to Run

The main script `main.py` is designed to be run from the command line, with a configuration file as an argument.

**Example:**
```bash
python main.py --config configs/us101_random0.05_PINN_res.json
```

## Citation

Coming soon...

