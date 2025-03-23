# PNNplus

`PNNplus` is a Python library that implements Parametric Neural Networks (PNN) for use in high-energy physics and beyond.

## Installation

```bash
pip install pnnplus
# Optional dependencies for high-energy physics (HEP) applications
# To install these, use: pip install "pnnplus[hep]"
```

## Example

```python
from pnnplus import *

pnn_plus = PNNplus(features=['f1', 'f2', 'f3', 'f4'], mass_columns=['mass'], weight_column='weight')
signal_statistics_df, background_statistics_df, experiment_statistics_df = pnn_plus.load_dataset(signal_path='../dataset/signal.csv', background_path='../dataset/background.csv', balance_signal_on_mass=True)

pnn_plus.plot_feature_distribution()
feature_correlation = pnn_plus.calc_feature_correlation_all()
pnn_plus.plot_mass_distribution()

pnn_plus.transform_dataset(save_feature_scaler='feature_scaler.pkl', save_mass_scaler='mass_scaler.pkl')
# pnn_plus.transform_dataset(load_feature_scaler='feature_scaler.pkl', load_mass_scaler='mass_scaler.pkl')

pnn_plus.train_model(model_path='model.keras')
# pnn_plus.load_model(model_path='model.keras')
# evaluate_results = pnn_plus.evaluate_model()

auc_df = pnn_plus.calc_auc_all()
pnn_plus.plot_score_all()
pnn_plus.plot_cut_efficiency_all()
importance_dfs = pnn_plus.calc_feature_importance_all()

predictions_signal = pnn_plus.predict_original(pnn_plus.X_signal, pnn_plus.mass_signal, root_path="pnnplus_signal.root")
predictions_background = pnn_plus.predict_original(pnn_plus.X_background, pnn_plus.mass_background, root_path="pnnplus_background.root")
```

## Documentation

If there is any inconsistency between this document and the function descriptions in [pnnplus.py](./pnnplus/pnnplus.py), please refer to [pnnplus.py](./pnnplus/pnnplus.py) as the authoritative source.

### `PNNplus` Class

#### `__init__(features, mass_columns=['mass'], weight_column='weight', background_type_column=None, random_seed=69)`
Initialize the PNNplus class with features, mass columns, weight column, background type column, and random seed.
- **Args:**
  - `features (list)`: List of feature column names.
  - `mass_columns (list)`: List of mass column names.
  - `weight_column (str)`: Name of the weight column.
  - `background_type_column (str)`: Name of the background type column.
  - `random_seed (int)`: Random seed for reproducibility.

#### `load_dataset(signal_path=None, background_path=None, experiment_path=None, signal_df=None, background_df=None, experiment_df=None, pre_selection=None, balance_signal_on_mass=False, background_mass_distribution='discrete', balance_signal_background=False, test_size=0.2)`
Load datasets from CSV files or DataFrames, split into training and test datasets, and check statistics. (CSV table headers should match the names of the features, mass_columns, and weight_column.)
- **Args:**
  - `signal_path (str)`: Path to the signal dataset file.
  - `background_path (str)`: Path to the background dataset file.
  - `experiment_path (str)`: Path to the experiment dataset file.
  - `signal_df (pd.DataFrame)`: DataFrame containing the signal dataset.
  - `background_df (pd.DataFrame)`: DataFrame containing the background dataset.
  - `experiment_df (pd.DataFrame)`: DataFrame containing the experiment dataset.
  - `pre_selection (callable)`: A function to apply pre-selection cuts to the data. It should take a DataFrame and return a boolean mask.
  - `balance_signal_on_mass (bool)`: Whether to balance the weights of the signal samples, making the sum of the weights equal for all masses when training the model.
  - `background_mass_distribution (str)`: Distribution type for the mass of background ('discrete', 'continuous', or 'original'). If 'discrete', the mass is sampled from the discrete distribution of the signal masses. If 'continuous', the mass is sampled from a uniform distribution within the range of the signal masses. If 'original', the original background mass distribution is used.
  - `balance_signal_background (bool)`: Whether to balance the weights of the signal and background samples, making the sum of the weights equal for both when training the model.
  - `test_size (float)`: Proportion of the dataset to include in the test split.
- **Returns:**
  - `signal_statistics_df (pd.DataFrame)`: DataFrame containing signal statistics.
  - `background_statistics_df (pd.DataFrame)`: DataFrame containing background statistics.
  - `experiment_statistics_df (pd.DataFrame)`: DataFrame containing experiment statistics.

#### `plot_feature_distribution(feature_list=None, mass_list=None, background_type_list=None, bins=50, density=True, log_scale=False, background_bar_stacked=True, plot_show=True, save_fig=False)`
Plot the feature distribution.
- **Args:**
  - `feature_list (list)`: List of features or tuples (feature, min, max) to plot the distribution for. If None, plot for all features.
  - `mass_list (list)`: List of signal mass values to plot the feature distribution for. If None, plot for all masses.
  - `background_type_list (list)`: List of background types to plot the feature distribution for. If None, plot for all types.
  - `bins (int)`: Number of bins for the histogram.
  - `density (bool)`: Whether to normalize the histogram to form a density plot.
  - `log_scale (bool)`: Whether to use a logarithmic scale for the y-axis.
  - `background_bar_stacked (bool)`: Whether to stack the background bars for different types.
  - `plot_show (bool)`: Whether to display the plots.
  - `save_fig (bool)`: Whether to save the plots as images.

#### `calc_feature_correlation_all(mass_list=None, plot_show=True, save_fig=False)`
Calculate the feature correlation for all masses.
- **Args:**
  - `mass_list (list)`: List of signal mass values to calculate feature correlation for. If None, calculate for all masses.
  - `plot_show (bool)`: Whether to display the plots.
  - `save_fig (bool)`: Whether to save the plots as images.
- **Returns:**
  - `correlation_dfs (list)`: DataFrames containing feature correlation for signal and background.

#### `plot_mass_distribution(bins=100, density=True, plot_show=True, save_fig=False)`
Plot the mass distribution.
- **Args:**
  - `bins (int)`: Number of bins for the histogram.
  - `density (bool)`: Whether to normalize the histogram to form a density plot.
  - `plot_show (bool)`: Whether to display the plot.
  - `save_fig (bool)`: Whether to save the plot as images.

#### `transform_dataset(load_feature_scaler=None, load_mass_scaler=None, save_feature_scaler=None, save_mass_scaler=None)`
Transform the dataset using StandardScaler. Optionally load or save the scalers.
- **Args:**
  - `load_feature_scaler (str)`: Path to load the feature scaler.
  - `load_mass_scaler (str)`: Path to load the mass scaler.
  - `save_feature_scaler (str)`: Path to save the feature scaler.
  - `save_mass_scaler (str)`: Path to save the mass scaler.

#### `train_model(model=None, ignore_negative_weights=True, epochs=20, batch_size=1024, validation_split=0.2, verbose=2, model_path=None)`
Train the model using the training dataset.
- **Args:**
  - `model (tf.keras.Model)`: Model to train. If None, use the default PNNplus model.
  - `ignore_negative_weights (bool)`: Whether to ignore samples with negative weights during training.
  - `epochs (int)`: Number of epochs to train.
  - `batch_size (int)`: Batch size for training.
  - `validation_split (float)`: Fraction of the training dataset to be used as validation dataset.
  - `verbose (int)`: Verbosity mode.
  - `model_path (str)`: Path to save the trained model.

#### `load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})`
Load a trained model from a file.
- **Args:**
  - `model_path (str)`: Path to the saved model file.
  - `custom_objects (dict)`: Custom objects for loading the model.

#### `evaluate_model(ignore_negative_weights=True, batch_size=1024, verbose=2)`
Evaluate the model using the test dataset.
- **Args:**
  - `ignore_negative_weights (bool)`: Whether to ignore samples with negative weights during evaluation.
  - `batch_size (int)`: Batch size for evaluation.
  - `verbose (int)`: Verbosity mode.
- **Returns:**
  - `evaluation (list)`: Evaluation metrics.

#### `predict(X_trans: np.ndarray, mass_trans: np.ndarray, batch_size=1024, verbose=2) -> np.ndarray`
Make predictions using the trained model on the transformed features.
- **Args:**
  - `X_trans (np.ndarray)`: Transformed features.
  - `mass_trans (np.ndarray)`: Transformed masses.
  - `batch_size (int)`: Batch size for prediction.
  - `verbose (int)`: Verbosity mode.
- **Returns:**
  - `predictions (np.ndarray)`: Model predictions.

#### `predict_original(X: np.ndarray, mass: np.ndarray, batch_size=1024, verbose=2, root_path=None) -> np.ndarray`
Make predictions using the trained model on the original features. Optionally save the events with predictions to a ROOT file.
- **Args:**
  - `X (np.ndarray)`: Original features.
  - `mass (np.ndarray)`: Original masses.
  - `batch_size (int)`: Batch size for prediction.
  - `verbose (int)`: Verbosity mode.
  - `root_path (str)`: ROOT file path to save the events with predictions.
- **Returns:**
  - `predictions (np.ndarray)`: Model predictions.

#### `calc_auc_all(mass_list=None, sample_size=1000000, plot_show=True, save_fig=False)`
Calculate the AUC score for all masses and optionally plot the ROC curve and AUC vs Mass figure.
- **Args:**
  - `mass_list (list)`: List of mass values to calculate AUC for. If None, calculate for all masses.
  - `sample_size (int)`: Number of samples to use. If greater than the total number of samples, use all samples.
  - `plot_show (bool)`: Whether to display the plots.
  - `save_fig (bool)`: Whether to save the plots as images.
- **Returns:**
  - `auc_df (pd.DataFrame)`: DataFrame containing mass values and corresponding AUC scores.

#### `plot_score_all(mass_list=None, sample_size=1000000, bins=50, plot_show=True, save_fig=False)`
Plot the output score distribution for all masses.
- **Args:**
  - `mass_list (list)`: List of mass values to plot the score distribution for. If None, plot for all masses.
  - `sample_size (int)`: Number of samples to use. If greater than the total number of samples, use all samples.
  - `bins (int)`: Number of bins for the histogram.
  - `plot_show (bool)`: Whether to display the plots.
  - `save_fig (bool)`: Whether to save the plots as images.

#### `plot_cut_efficiency_all(mass_list=None, signal_numbers=None, background_number=None, sample_size=1000000, n_cuts=1000, plot_show=True, save_fig=False)`
Plot the cut efficiency and signal significance for all masses.
- **Args:**
  - `mass_list (list)`: List of mass values to plot the cut efficiency for. If None, plot for all masses.
  - `signal_numbers (list)`: List of weighted numbers of signal samples for each mass value. If None, use the weighted number of signal samples.
  - `background_number (float)`: Weighted number of background samples. If None, use the weighted number of background samples.
  - `sample_size (int)`: Number of samples to use. If greater than the total number of samples, use all samples.
  - `n_cuts (int)`: Number of cut values to evaluate.
  - `plot_show (bool)`: Whether to display the plots.
  - `save_fig (bool)`: Whether to save the plots as images.

#### `calc_feature_importance_all(mass_list=None, sample_size=100000, steps=50)`
Calculate feature importance for all masses using integrated gradients.
- **Args:**
  - `mass_list (list)`: List of mass values to calculate feature importance for. If None, calculate for all masses.
  - `sample_size (int)`: Number of samples to use. If greater than the total number of samples, use all samples.
  - `steps (int)`: Number of steps for integrated gradients.
- **Returns:**
  - `importance_dfs (list)`: DataFrames containing feature importance scores.

### Utility Functions

#### `focal_loss(gamma=2., alpha=0.25)`
Focal Loss for binary classification.
- **Args:**
  - `gamma (float)`: Focusing parameter.
  - `alpha (float)`: Balancing parameter.
- **Returns:**
  - `loss (callable)`: Focal loss function.

#### `pnnplus_model(X_dim, mass_dim, units=[300, 150, 100, 50], dropout_rate=0.25, learning_rate=3e-4, loss_function='binary_crossentropy')`
Define the model architecture.
- **Args:**
  - `X_dim (int)`: Dimension of the features.
  - `mass_dim (int)`: Dimension of the masses.
  - `units (list)`: List of units in the dense layers.
  - `dropout_rate (float or list)`: Dropout rate(s) for the dropout layers.
  - `learning_rate (float)`: Learning rate for the optimizer.
  - `loss_function (str or callable)`: Loss function for the model.
- **Returns:**
  - `model (tf.keras.Model)`: Compiled Keras model.

#### `calc_feature_correlation(X_input, weights)`
Calculate the weighted correlation matrix.
- **Args:**
  - `X_input (np.ndarray)`: Sample features.
  - `weights (np.ndarray)`: Sample weights.
- **Returns:**
  - `weighted_corr (np.ndarray)`: Weighted correlation matrix.

#### `calc_auc(y_true, y_pred, weights, plot_show=True, save_fig=False, filename=None)`
Calculate the AUC score and optionally plot the ROC curve.
- **Args:**
  - `y_true (np.ndarray)`: True labels.
  - `y_pred (np.ndarray)`: Predicted labels.
  - `weights (np.ndarray)`: Sample weights.
  - `plot_show (bool)`: Whether to display the plot.
  - `save_fig (bool)`: Whether to save the plot as images.
  - `filename (str)`: Filename for the saved images.
- **Returns:**
  - `auc (float)`: AUC score.

#### `weighted_ks_2samp(dataset1, dataset2, weights1, weights2)`
Compute the Weighted Kolmogorov-Smirnov statistic.
- **Args:**
  - `dataset1 (np.ndarray)`: First dataset samples.
  - `dataset2 (np.ndarray)`: Second dataset samples.
  - `weights1 (np.ndarray)`: Weights for the first dataset samples.
  - `weights2 (np.ndarray)`: Weights for the second dataset samples.
- **Returns:**
  - `ks_stat (float)`: KS statistic.
  - `p_value (float)`: Two-tailed p-value.

#### `plot_score(y_train, y_pred_train, weights_train, y_test, y_pred_test, weights_test, bins=50, plot_show=True, save_fig=False, filename=None)`
Plot the output score distribution for training and test dataset.
- **Args:**
  - `y_train (np.ndarray)`: True labels for training dataset.
  - `y_pred_train (np.ndarray)`: Predicted labels for training dataset.
  - `weights_train (np.ndarray)`: Sample weights for training dataset.
  - `y_test (np.ndarray)`: True labels for test dataset.
  - `y_pred_test (np.ndarray)`: Predicted labels for test dataset.
  - `weights_test (np.ndarray)`: Sample weights for test dataset.
  - `bins (int)`: Number of bins for the histogram.
  - `plot_show (bool)`: Whether to display the plot.
  - `save_fig (bool)`: Whether to save the plot as images.
  - `filename (str)`: Filename for the saved images.

#### `plot_cut_efficiency(y_true, y_pred, weights, signal_number=None, background_number=None, n_cuts=1000, plot_show=True, save_fig=False, filename=None)`
Plot the cut efficiency and signal significance as a function of cut value.
- **Args:**
  - `y_true (np.ndarray)`: True labels.
  - `y_pred (np.ndarray)`: Predicted labels.
  - `weights (np.ndarray)`: Sample weights.
  - `signal_number (float)`: Weighted number of signal samples.
  - `background_number (float)`: Weighted number of background samples.
  - `n_cuts (int)`: Number of cut values to evaluate.
  - `plot_show (bool)`: Whether to display the plot.
  - `save_fig (bool)`: Whether to save the plot as images.
  - `filename (str)`: Filename for the saved images.

#### `calc_feature_importance(model, X_input, m_input, weights, steps=50)`
Calculate feature importance using integrated gradients.
- **Args:**
  - `X_input (tf.Tensor)`: Transformed features.
  - `m_input (tf.Tensor)`: Transformed masses.
  - `weights (tf.Tensor)`: Sample weights.
  - `steps (int)`: Number of steps for integrated gradients.
- **Returns:**
  - `importance (tf.Tensor)`: Feature importance scores.

## License

PNNplus is licensed under the GNU General Public License v3.0.