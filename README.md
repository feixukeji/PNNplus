# PNNplus

`PNNplus` is a Python library that implements Parametric Neural Networks (PNN) for use in high-energy physics and beyond.

## Installation

```bash
pip install pnnplus
```

## Usage

Function descriptions can be found in `pnnplus.py`.

Example usage:

```python
from pnnplus import *

pnn_plus = PNNplus(features=['f1', 'f2', 'f3', 'f4'], mass_columns=['mass'], weight_column='weight')
pnn_plus.load_dataset('../dataset/signal.csv', '../dataset/background.csv')

pnn_plus.plot_feature_distribution()
feature_correlation = pnn_plus.calc_feature_correlation_all()
pnn_plus.plot_mass_distribution()

pnn_plus.transform_dataset(save_feature_scaler='feature_scaler.pkl', save_mass_scaler='mass_scaler.pkl')
# pnn_plus.transform_dataset(load_feature_scaler='feature_scaler.pkl', load_mass_scaler='mass_scaler.pkl')

pnn_plus.train_model(save_file='model.keras')
# pnn_plus.load_model(save_file='model.keras')
# evaluate_results = pnn_plus.evaluate_model()

auc_df = pnn_plus.calc_auc_all()
pnn_plus.plot_score_all()
pnn_plus.plot_cut_efficiency_all()
importance_dfs = pnn_plus.calc_feature_importance_all()
```

## License

PNNplus is licensed under the GNU General Public License v3.0.