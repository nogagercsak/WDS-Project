# Predicting Forest Fire Severity Using Climate Data🌲🌍

Machine learning model for predicting forest fire risk using meteorological data from Portugal's Montesinho Natural Park.

## Overview

This project develops a custom rule-based classification model that predicts fire risk based on weather conditions and fire indices. Created for the Winston Data Scholars Program at NYU GSTEM (Summer 2025).

**Key Results:**
- 91.36% fire detection rate (recall)
- Custom Fire Detection Rate (FDR) metric for asymmetric risk assessment
- Outperforms k-NN baseline in fire detection capability

## Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Forest+Fires)
- 517 observations (2000-2003)
- Features: Temperature, humidity, wind, rain, Fire Weather Index components (FFMC, DMC, DC, ISI)
- Target: Binary classification (fire/no-fire)

## Approach

The model uses a weighted scoring system based on:
- Fire Weather Index thresholds
- Meteorological conditions
- Interaction effects between risk factors
- Seasonal and spatial patterns

Evaluation prioritizes fire detection over precision, using a custom FDR metric:
```
FDR = 0.7 × Recall + 0.3 × Precision
```

## Performance

| Metric | Custom Model | k-NN (k=11) |
|--------|-------------|-------------|
| Accuracy | 57.05% | 62.18% |
| Fire Detection Rate | 80.52% | 69.43% |
| Recall | 91.36% | 72.84% |
| Precision | 55.22% | 61.46% |

The custom model excels at detecting actual fires (91% recall) at the cost of more false positives, which aligns with fire management priorities where missing a fire is far more dangerous than false alarms.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/forest-fire-prediction.git
cd forest-fire-prediction

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook forest_fire_prediction.ipynb
```

## Usage

```python
from fire_risk_model import custom_fire_risk_model
import pandas as pd

# Load data
data = pd.read_csv('fire_data.csv')

# Get predictions
predictions, risk_scores = custom_fire_risk_model(data)
```
## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Key Findings

- Temperature and humidity are strongest predictors
- Interaction effects (heat + drought + wind) significantly increase risk
- Summer months show highest fire frequency
- Domain knowledge improves detection over pure ML approaches

## Acknowledgments

- Winston Data Scholars Program at NYU
- Professor Fanny Shum
- UCI Machine Learning Repository

*Developed for Winston Data Scholars Program, NYU GSTEM Summer 2025*
