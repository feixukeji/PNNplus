# PNNplus

`PNNplus` is a Python library that implements Parametric Neural Networks (PNN) for use in high-energy physics and beyond.

## Installation

```bash
pip install pnnplus
```

## Usage

```python
from pnnplus import *

pnn_plus = PNNplus(features=['f1', 'f2', 'f3', 'f4'], mass_columns=['mass'], weight_column='weight', random_seed=69)
pnn_plus.load_data('../data/signal.csv', '../data/background.csv')

pnn_plus.plot_feature_distribution(mass_list=[[200], [400], [600]])
feature_correlation = pnn_plus.calc_feature_correlation_all(mass_list=[[200], [400], [600]])
pnn_plus.plot_mass_distribution()

pnn_plus.transform_data(save_feature_scaler='feature_scaler.pkl', save_mass_scaler='mass_scaler.pkl')
# pnn_plus.transform_data(load_feature_scaler='feature_scaler.pkl', load_mass_scaler='mass_scaler.pkl')

pnn_plus.train_model(save_file='model.keras')
# pnn_plus.load_model(save_file='model.keras')
evaluate_results = pnn_plus.evaluate_model()

auc_df = pnn_plus.calc_auc_all(mass_list=[[200], [400], [600]])
pnn_plus.plot_score_all(mass_list=[[200], [400], [600]])
pnn_plus.plot_cut_efficiency_all(mass_list=[[200], [400], [600]])
importance_dfs = pnn_plus.calc_feature_importance_all(mass_list=[[200], [400], [600]])
```

## License

PNNplus is licensed under the GNU General Public License v3.0.