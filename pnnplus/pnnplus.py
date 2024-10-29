import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Multiply, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display


def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for binary classification.
    
    Args:
        gamma (float): Focusing parameter.
        alpha (float): Balancing parameter.
    
    Returns:
        loss (callable): Focal loss function.
    """
    def focal_loss_fixed(y_true, y_pred, sample_weight=None):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = K.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = - alpha_t * K.pow((1 - p_t), gamma) * K.log(p_t)
        
        if sample_weight is not None:
            fl = fl * sample_weight
        
        return K.mean(fl)
    return focal_loss_fixed

def pnnplus_model(X_dim, mass_dim, units=[300, 150, 100, 50], dropout_rate=0.25, learning_rate=3e-4, loss_function='binary_crossentropy'):
    """
    Define the model architecture.
    
    Args:
        X_dim (int): Dimension of the features.
        mass_dim (int): Dimension of the masses.
        units (list): List of units in the dense layers.
        dropout_rate (float or list): Dropout rate(s) for the dropout layers.
        learning_rate (float): Learning rate for the optimizer.
        loss_function (str or callable): Loss function for the model.
    
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    if isinstance(dropout_rate, list) and len(dropout_rate) != len(units):
        raise ValueError("Number of dropout rates must match the number of dense layers.")

    def affine_conditioning(x, mass, units):
        scaling = Dense(units, activation='linear')(mass)
        bias = Dense(units, activation='linear')(mass)
        return Add()([Multiply()([x, scaling]), bias])
    
    X_input = Input(shape=(X_dim,))
    mass_input = Input(shape=(mass_dim,))

    x = Dense(units[0], activation='relu')(X_input)
    x = affine_conditioning(x, mass_input, units[0])
    if isinstance(dropout_rate, list):
        x = Dropout(dropout_rate[0])(x)
    else:
        x = Dropout(dropout_rate)(x)

    for i in range(1, len(units)):
        x = Dense(units[i], activation='relu')(x)
        x = affine_conditioning(x, mass_input, units[i])
        if isinstance(dropout_rate, list):
            x = Dropout(dropout_rate[i])(x)
        else:
            x = Dropout(dropout_rate)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[X_input, mass_input], outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, weighted_metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

def calc_feature_correlation(X_input, weights):
    """
    Calculate the weighted correlation matrix.
    
    Args:
        X_input (np.ndarray): Sample features.
        weights (np.ndarray): Sample weights.
    
    Returns:
        weighted_corr (np.ndarray): Weighted correlation matrix.
    """
    weights = weights / np.sum(weights)
    X_mean = np.average(X_input, axis=0, weights=weights)
    X_centered = X_input - X_mean
    weighted_cov = np.dot(weights * X_centered.T, X_centered)
    weighted_std = np.sqrt(np.diag(weighted_cov))
    weighted_corr = weighted_cov / np.outer(weighted_std, weighted_std)
    return weighted_corr

def calc_auc(y_true, y_pred, weights, plot_roc=True):
    """
    Calculate the AUC score and optionally plot the ROC curve.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        weights (np.ndarray): Sample weights.
        plot_roc (bool): Whether to plot the ROC curve.
    
    Returns:
        auc (float): AUC score.
    """
    auc = roc_auc_score(y_true, y_pred, sample_weight=weights)
    if plot_roc:
        fpr, tpr, _ = roc_curve(y_true, y_pred, sample_weight=weights)
        with matplotlib.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
            plt.figure(figsize=(6, 4))
            plt.grid()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.5f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([-0.005, 1.005])
            plt.ylim([-0.005, 1.005])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
            plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
            plt.show()
    return auc
    
def weighted_ks_2samp(dataset1, dataset2, weights1, weights2):
    """
    Compute the Weighted Kolmogorov-Smirnov statistic.
    
    Args:
        dataset1 (np.ndarray): First dataset samples.
        dataset2 (np.ndarray): Second dataset samples.
        weights1 (np.ndarray): Weights for the first dataset samples.
        weights2 (np.ndarray): Weights for the second dataset samples.
    
    Returns:
        ks_stat (float): KS statistic.
        p_value (float): Two-tailed p-value.
    """
    weights1 = weights1 / np.sum(weights1)
    weights2 = weights2 / np.sum(weights2)
    
    dataset_all = np.concatenate([dataset1, dataset2])
    cdf1 = np.cumsum(weights1[np.argsort(dataset1)])
    cdf2 = np.cumsum(weights2[np.argsort(dataset2)])
    
    cdf1_interp = np.interp(dataset_all, np.sort(dataset1), cdf1, left=0, right=1)
    cdf2_interp = np.interp(dataset_all, np.sort(dataset2), cdf2, left=0, right=1)
    
    ks_stat = np.max(np.abs(cdf1_interp - cdf2_interp))
    
    n1 = len(dataset1)
    n2 = len(dataset2)
    en = np.sqrt(n1 * n2 / (n1 + n2))
    p_value = 2 * np.exp(-2 * (ks_stat * en) ** 2)
    
    return ks_stat, p_value

def plot_score(y_train, y_pred_train, weights_train, y_test, y_pred_test, weights_test, bins=50):
    """
    Plot the output score distribution for training and test dataset.
    
    Args:
        y_train (np.ndarray): True labels for training dataset.
        y_pred_train (np.ndarray): Predicted labels for training dataset.
        weights_train (np.ndarray): Sample weights for training dataset.
        y_test (np.ndarray): True labels for test dataset.
        y_pred_test (np.ndarray): Predicted labels for test dataset.
        weights_test (np.ndarray): Sample weights for test dataset.
        bins (int): Number of bins for the histogram.
    """
    with matplotlib.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        plt.figure(figsize=(8, 5))
        plt.hist(y_pred_test[y_test == 1], bins=bins, range=[0, 1], alpha=0.5, label='Signal (Test)', weights=weights_test[y_test == 1], color='blue', density=True)
        plt.hist(y_pred_test[y_test == 0], bins=bins, range=[0, 1], alpha=0.5, label='Background (Test)', weights=weights_test[y_test == 0], color='red', density=True)

        hist_signal_train, bin_edges = np.histogram(y_pred_train[y_train == 1], bins=bins, range=[0, 1], weights=weights_train[y_train == 1], density=True)
        hist_background_train, _ = np.histogram(y_pred_train[y_train == 0], bins=bins, range=[0, 1], weights=weights_train[y_train == 0], density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        plt.scatter(bin_centers, hist_signal_train, label='Signal (Train)', color='blue', marker='o', s=10)
        plt.scatter(bin_centers, hist_background_train, label='Background (Train)', color='red', marker='o', s=10)

        plt.xlabel('Output Score')
        plt.ylabel('Density')
        plt.title('Output Score Distribution: Train vs Test')
        plt.legend()
        plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
        plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
        plt.show()

def plot_cut_efficiency(y_true, y_pred, weights, signal_number=None, background_number=None, n_cuts=1000):
    """
    Plot the cut efficiency and signal significance as a function of cut value.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        weights (np.ndarray): Sample weights.
        signal_number (float): Weighted number of signal events.
        background_number (float): Weighted number of background events.
        n_cuts (int): Number of cut values to evaluate.
    """
    cut_values = np.linspace(0, 1, n_cuts)
    signal_efficiencies = []
    background_efficiencies = []
    Ss = []
    Bs = []
    signal_significances = []

    sample_signal_number = np.sum(weights[y_true == 1])
    sample_background_number = np.sum(weights[y_true == 0])

    if signal_number is None:
        signal_number = sample_signal_number
    if background_number is None:
        background_number = sample_background_number

    for cut in cut_values:
        signal_weight = np.sum(weights[(y_true == 1) & (y_pred >= cut)])
        background_weight = np.sum(weights[(y_true == 0) & (y_pred >= cut)])

        signal_eff = signal_weight / sample_signal_number
        background_eff = background_weight / sample_background_number

        S = signal_eff * signal_number
        B = background_eff * background_number

        if S + B > 0:
            signal_sig = S / np.sqrt(S + B)
        else:
            signal_sig = 0

        signal_efficiencies.append(signal_eff)
        background_efficiencies.append(background_eff)
        Ss.append(S)
        Bs.append(B)
        signal_significances.append(signal_sig)

    signal_efficiencies = np.array(signal_efficiencies)
    background_efficiencies = np.array(background_efficiencies)
    Ss = np.array(Ss)
    Bs = np.array(Bs)
    signal_significances = np.array(signal_significances)

    max_signal_sig_index = np.argmax(signal_significances)
    optimal_cut = cut_values[max_signal_sig_index]
    signal_eff_opt = signal_efficiencies[max_signal_sig_index]
    background_eff_opt = background_efficiencies[max_signal_sig_index]
    S_opt = Ss[max_signal_sig_index]
    B_opt = Bs[max_signal_sig_index]
    signal_sig_opt = signal_significances[max_signal_sig_index]

    with matplotlib.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        _, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(cut_values, signal_efficiencies, 'b-', label='Signal Efficiency')
        ax1.plot(cut_values, background_efficiencies, 'r-', label='Background Efficiency')
        ax1.set_xlabel('Cut Value')
        ax1.set_ylabel('Efficiency')
        ax1.legend(loc='upper right')
        ax1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
        ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))

        ax2 = ax1.twinx()
        ax2.plot(cut_values, signal_significances, 'g-', label='S/sqrt(S+B)')
        ax2.set_ylabel('S/sqrt(S+B)', color='g')
        ax2.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))

        ax1.axvline(optimal_cut, color='gray', linestyle='--', label=f'Optimal Cut = {optimal_cut:.4f}')
        ax1.legend(loc='center right')

        plt.title('Cut Efficiency and S/sqrt(S+B) vs Cut Value')
        plt.show()

    print(f"Optimal Cut Value: {optimal_cut:.6f}")
    print(f"Signal Efficiency at Optimal Cut: {signal_eff_opt:.6f}")
    print(f"Background Efficiency at Optimal Cut: {background_eff_opt:.6f}")
    print(f"S (Signal Number) at Optimal Cut: {S_opt:.6f}")
    print(f"B (Background Number) at Optimal Cut: {B_opt:.6f}")
    print(f"S/sqrt(S+B) at Optimal Cut: {signal_sig_opt:.6f}")

def calc_feature_importance(model, X_input, m_input, weights, steps=50):
    """
    Calculate feature importance using integrated gradients.
    
    Args:
        X_input (tf.Tensor): Transformed features.
        m_input (tf.Tensor): Transformed masses.
        weights (tf.Tensor): Sample weights.
        steps (int): Number of steps for integrated gradients.
    
    Returns:
        importance (tf.Tensor): Feature importance scores.
    """
    baseline = tf.reduce_mean(X_input, axis=0)
    interpolated_inputs = [baseline + (i / steps) * (X_input - baseline) for i in range(steps + 1)]

    gradients = []
    for i in range(steps + 1):
        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs[i])
            predictions = model([interpolated_inputs[i], m_input])
        gradients.append(tape.gradient(predictions, interpolated_inputs[i]))

    integrated_gradients = tf.reduce_mean(gradients, axis=0) * (X_input - baseline)
    importance = tf.reduce_sum(tf.abs(integrated_gradients) * tf.expand_dims(weights, axis=1), axis=0)
    importance /= tf.reduce_sum(importance)

    return importance
    

class PNNplus:
    def __init__(self, features, mass_columns=['mass'], weight_column='weight', background_type_column=None, random_seed=69):
        """
        Initialize the PNNplus class with features, mass columns, weight column, background type column, and random seed.
        
        Args:
            features (list): List of feature column names.
            mass_columns (list): List of mass column names.
            weight_column (str): Name of the weight column.
            background_type_column (str): Name of the background type column.
            random_seed (int): Random seed for reproducibility.
        """
        self.features = features
        self.mass_columns = mass_columns
        self.weight_column = weight_column
        self.background_type_column = background_type_column
        self.random_seed = random_seed
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        self.dataset_loaded = False
        self.dataset_transformed = False
        self.model_trained = False
        print("Note: All numbers and plots output by PNNplus are weighted.")

    def load_dataset(self, signal_path=None, background_path=None, experiment_path=None, pre_selection=None, background_mass_distribution='discrete', balance_signal_background=False, test_size=0.2):
        """
        Load datasets from CSV files and split into training and test datasets. (CSV table headers should match the names of the features, mass_columns, and weight_column.)
        
        Args:
            signal_path (str): Path to the signal dataset file.
            background_path (str): Path to the background dataset file.
            experiment_path (str): Path to the experiment dataset file.
            pre_selection (callable): A function to apply pre-selection cuts to the data. It should take a DataFrame and return a boolean mask.
            background_mass_distribution (str): Distribution type for the mass of background ('discrete', 'continuous', or 'original').
            balance_signal_background (bool): Whether to balance the weights of the signal and background samples, making the sum of the weights equal for both.
            test_size (float): Proportion of the dataset to include in the test split.
        """
        self.X_signal = None
        self.mass_signal = None
        self.unique_mass = []
        self.weights_signal = None

        self.X_background = None
        self.mass_background = None
        self.weights_background = None
        self.background_number = 0
        self.background_types = None
        self.unique_background_types = []

        self.X_experiment = None
        self.weights_experiment = None

        self.X_train = None
        self.X_test = None
        self.mass_train = None
        self.mass_test = None
        self.y_train = None
        self.y_test = None
        self.weights_train = None
        self.weights_test = None
        
        if isinstance(signal_path, str):
            signal_df = pd.read_csv(signal_path)
            if pre_selection is not None:
                signal_df = signal_df[pre_selection(signal_df)]
            self.X_signal = signal_df[self.features].values
            self.mass_signal = signal_df[self.mass_columns].values
            self.unique_mass = [list(mass) for mass in np.unique(self.mass_signal, axis=0)]
            y_signal = np.ones(len(signal_df))
            self.weights_signal = signal_df[self.weight_column].values

        if isinstance(background_path, str):
            background_df = pd.read_csv(background_path)
            if pre_selection is not None:
                background_df = background_df[pre_selection(background_df)]
            self.X_background = background_df[self.features].values
            if background_mass_distribution == 'discrete':
                if isinstance(signal_path, str):
                    mass_weighted_counts = signal_df.groupby(self.mass_columns)[self.weight_column].sum()
                    mass_probabilities = mass_weighted_counts / mass_weighted_counts.sum()
                    chosen_masses = np.random.choice(mass_weighted_counts.index, size=len(background_df), p=mass_probabilities)
                    self.mass_background = np.array([[mass] if np.isscalar(mass) else list(mass) for mass in chosen_masses])
            elif background_mass_distribution == 'continuous':
                if isinstance(signal_path, str):
                    self.mass_background = np.random.uniform(low=self.mass_signal.min(axis=0), high=self.mass_signal.max(axis=0), size=(len(background_df), len(self.mass_columns)))
            elif background_mass_distribution == 'original':
                self.mass_background = background_df[self.mass_columns].values
            else:
                raise ValueError("Invalid background_mass_distribution. Choose 'discrete', 'continuous', or 'original'.")
            y_background = np.zeros(len(background_df))
            self.weights_background = background_df[self.weight_column].values
            self.background_number = np.sum(self.weights_background)
            if balance_signal_background and isinstance(signal_path, str):
                signal_weight_sum = np.sum(self.weights_signal)
                background_weight_sum = np.sum(self.weights_background)
                self.weights_background = self.weights_background / background_weight_sum * signal_weight_sum
            if self.background_type_column is not None:
                self.background_types = background_df[self.background_type_column].values
                self.unique_background_types = np.unique(self.background_types)

        if isinstance(experiment_path, str):
            experiment_df = pd.read_csv(experiment_path)
            if pre_selection is not None:
                experiment_df = experiment_df[pre_selection(experiment_df)]
            self.X_experiment = experiment_df[self.features].values
            self.weights_experiment = experiment_df[self.weight_column].values

        if isinstance(signal_path, str) and isinstance(background_path, str):
            X = np.vstack((self.X_signal, self.X_background))
            mass = np.vstack((self.mass_signal, self.mass_background))
            y = np.hstack((y_signal, y_background))
            weights = np.hstack((self.weights_signal, self.weights_background))
            self.X_train, self.X_test, self.mass_train, self.mass_test, self.y_train, self.y_test, self.weights_train, self.weights_test = train_test_split(X, mass, y, weights, test_size=test_size, random_state=self.random_seed)

        self.dataset_loaded = True
        self.dataset_transformed = False

    def plot_feature_distribution(self, feature_list=None, signal_mass_list=None, background_type_list=None, bins=50, density=True, log_scale=False, background_bar_stacked=True):
        """
        Plot the feature distribution.
        
        Args:
            feature_list (list): List of feature names to plot the distribution for. If None, plot for all features.
            signal_mass_list (list): List of signal mass values to plot the feature distribution for. If None, plot for all masses.
            background_type_list (list): List of background types to plot the feature distribution for. If None, plot for all types.
            bins (int): Number of bins for the histogram.
            density (bool): Whether to normalize the histogram to form a density plot.
            log_scale (bool): Whether to use a logarithmic scale for the y-axis.
            background_bar_stacked (bool): Whether to stack the background bars for different types.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before plotting feature distribution. Please call load_dataset() first.")
        
        if feature_list is None:
            feature_list = self.features
        if signal_mass_list is None:
            signal_mass_list = self.unique_mass
        if background_type_list is None:
            background_type_list = self.unique_background_types

        for feature_idx, feature in enumerate(feature_list):
            X = []
            if self.X_signal is not None:
                X.append(self.X_signal[:, feature_idx])
            if self.X_background is not None:
                X.append(self.X_background[:, feature_idx])
            if self.X_experiment is not None:
                X.append(self.X_experiment[:, feature_idx])
            X = np.concatenate(X)
            bin_edges = np.histogram_bin_edges(X, bins=bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            with matplotlib.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
                if self.X_background is not None and self.X_experiment is not None:
                    _, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05}, figsize=(8, 6))
                else:
                    _, ax_top = plt.subplots(figsize=(8, 5))
                
                for mass_value in signal_mass_list:
                    signal_mask = np.all(self.mass_signal == mass_value, axis=1)
                    ax_top.hist(self.X_signal[signal_mask, feature_idx], bins=bin_edges, histtype='step', label=f'Signal (Mass={mass_value})', density=density, weights=self.weights_signal[signal_mask])
                
                if self.X_background is not None:
                    if self.background_types is not None and background_bar_stacked:
                        hist_features = []
                        hist_weights = []
                        for background_type in background_type_list:
                            background_mask = self.background_types == background_type
                            hist_features.append(self.X_background[background_mask, feature_idx])
                            hist_weights.append(self.weights_background[background_mask])
                        ax_top.hist(hist_features, bins=bin_edges, histtype='barstacked', label=background_type_list, density=density, weights=hist_weights)
                        hist_background, _ = np.histogram(np.concatenate(hist_features), bins=bin_edges, density=density, weights=np.concatenate(hist_weights))
                    else:
                        ax_top.hist(self.X_background[:, feature_idx], bins=bin_edges, histtype='step', label='Background', density=density, weights=self.weights_background)
                        hist_background, _ = np.histogram(self.X_background[:, feature_idx], bins=bin_edges, density=density, weights=self.weights_background)
                
                if self.X_experiment is not None:
                    hist_experiment, _ = np.histogram(self.X_experiment[:, feature_idx], bins=bin_edges, density=density, weights=self.weights_experiment)
                    ax_top.scatter(bin_centers, hist_experiment, label='Data', color='black', marker='o', s=8)
                    if not density:
                        hist_experiment_error = np.sqrt(np.histogram(self.X_experiment[:, feature_idx], bins=bin_edges, density=False, weights=self.weights_experiment**2)[0])
                        ax_top.errorbar(bin_centers, hist_experiment, yerr=hist_experiment_error, fmt='none', color='black', elinewidth=1)
                
                ax_top.set_ylabel('Density' if density else 'Events')
                ax_top.set_title(f'{feature} Distribution')
                ax_top.legend()
                ax_top.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                ax_top.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                if log_scale:
                    ax_top.set_yscale('log')
                
                if self.X_background is not None and self.X_experiment is not None:
                    hist_ratio = hist_experiment / hist_background
                    ax_bottom.grid()
                    ax_bottom.scatter(bin_centers, hist_ratio, color='black', marker='o', s=8)
                    if not density:
                        hist_ratio_error = hist_experiment_error / hist_background
                        ax_bottom.errorbar(bin_centers, hist_ratio, yerr=hist_ratio_error, fmt='none', color='black', elinewidth=1)
                    ax_bottom.set_xlabel(f'{feature}')
                    ax_bottom.set_ylabel('Data/MC')
                    ax_bottom.set_ylim(0.5, 1.5)
                    ax_bottom.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                    ax_bottom.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                else:
                    ax_top.set_xlabel(f'{feature}')
                
                plt.show()
    
    def calc_feature_correlation_all(self, signal_mass_list=None):
        """
        Calculate the feature correlation for all masses.
        
        Args:
            signal_mass_list (list): List of signal mass values to calculate feature correlation for. If None, calculate for all masses.
        
        Returns:
            correlation_dfs (list): DataFrames containing feature correlation for signal and background.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before calculating feature correlation. Please call load_dataset() first.")
        
        if signal_mass_list is None:
            signal_mass_list = self.unique_mass
        
        correlation_dfs = []
        for mass_value in signal_mass_list:
            signal_mask = np.all(self.mass_signal == mass_value, axis=1)
            correlation_signal = calc_feature_correlation(self.X_signal[signal_mask], self.weights_signal[signal_mask])
            correlation_df_signal = pd.DataFrame(correlation_signal, columns=self.features, index=self.features)
            correlation_dfs.append((mass_value, correlation_df_signal))

            plt.figure(figsize=(8, 5))
            sns.heatmap(correlation_df_signal * 100, annot=True, cmap='coolwarm', vmin=-100, vmax=100, fmt=".0f")
            plt.title(f"Signal Feature Correlation for Mass = {mass_value} (×100)")
            plt.show()

        if self.X_background is not None:
            correlation_background = calc_feature_correlation(self.X_background, self.weights_background)
            correlation_df_background = pd.DataFrame(correlation_background, columns=self.features, index=self.features)
            correlation_dfs.append(('Background', correlation_df_background))

            plt.figure(figsize=(8, 5))
            sns.heatmap(correlation_df_background * 100, annot=True, cmap='coolwarm', vmin=-100, vmax=100, fmt=".0f")
            plt.title("Background Feature Correlation (×100)")
            plt.show()

        if self.X_experiment is not None:
            correlation_experiment = calc_feature_correlation(self.X_experiment, self.weights_experiment)
            correlation_df_experiment = pd.DataFrame(correlation_experiment, columns=self.features, index=self.features)
            correlation_dfs.append(('Data', correlation_df_experiment))

            plt.figure(figsize=(8, 5))
            sns.heatmap(correlation_df_experiment * 100, annot=True, cmap='coolwarm', vmin=-100, vmax=100, fmt=".0f")
            plt.title("Data Feature Correlation (×100)")
            plt.show()
        
        return correlation_dfs

    def plot_mass_distribution(self, bins=100):
        """
        Plot the mass distribution.
        
        Args:
            bins (int): Number of bins for the histogram.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before plotting mass distribution. Please call load_dataset() first.")
        
        with matplotlib.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
            for i, mass_column in enumerate(self.mass_columns):
                plt.figure(figsize=(8, 5))
                plt.hist(self.mass_signal[:, i], bins=bins, histtype='step', label='Signal', weights=self.weights_signal, density=True)
                plt.hist(self.mass_background[:, i], bins=bins, histtype='step', label='Background', weights=self.weights_background, density=True)
                plt.xlabel(mass_column)
                plt.ylabel('Density')
                plt.title(f'{mass_column} Distribution')
                plt.legend()
                plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                plt.show()

    def transform_dataset(self, load_feature_scaler=None, load_mass_scaler=None, save_feature_scaler=None, save_mass_scaler=None):
        """
        Transform the dataset using StandardScaler. Optionally load or save the scalers.
        
        Args:
            load_feature_scaler (str): Path to load the feature scaler.
            load_mass_scaler (str): Path to load the mass scaler.
            save_feature_scaler (str): Path to save the feature scaler.
            save_mass_scaler (str): Path to save the mass scaler.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before transforming. Please call load_dataset() first.")
        
        if load_feature_scaler is not None:
            self.feature_scaler = joblib.load(load_feature_scaler)
        elif not hasattr(self, 'feature_scaler'):
            self.feature_scaler = StandardScaler()
        if load_mass_scaler is not None:
            self.mass_scaler = joblib.load(load_mass_scaler)
        elif not hasattr(self, 'mass_scaler'):
            self.mass_scaler = StandardScaler()

        if not hasattr(self.feature_scaler, 'mean_'):
            self.feature_scaler.fit(self.X_train)
        if not hasattr(self.mass_scaler, 'mean_'):
            self.mass_scaler.fit(self.mass_train)

        if self.X_train is not None:
            self.X_train_trans = self.feature_scaler.transform(self.X_train)
        if self.X_test is not None:
            self.X_test_trans = self.feature_scaler.transform(self.X_test)
        if self.X_experiment is not None:
            self.X_experiment_trans = self.feature_scaler.transform(self.X_experiment)

        if save_feature_scaler is not None:
            joblib.dump(self.feature_scaler, save_feature_scaler)
        if save_mass_scaler is not None:
            joblib.dump(self.mass_scaler, save_mass_scaler)
        
        self.dataset_transformed = True

    def train_model(self, model=None, epochs=20, batch_size=1024, validation_split=0.2, verbose=2, save_file=None):
        """
        Train the model using the training dataset.
        
        Args:
            model (tf.keras.Model): Model to train. If None, use the default PNNplus model.
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of the training dataset to be used as validation dataset.
            verbose (int): Verbosity mode.
            save_file (str): Path to save the trained model.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before training the model. Please call load_dataset() first.")
        if not self.dataset_transformed:
            raise RuntimeError("Dataset must be transformed before training the model. Please call transform_dataset() first.")
        
        if model is None:
            self.model = pnnplus_model(len(self.features), len(self.mass_columns))
        else:
            self.model = model

        self.model.fit([self.X_train_trans, self.mass_scaler.transform(self.mass_train)], self.y_train, sample_weight=self.weights_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

        if save_file is not None:
            self.model.save(save_file)
        
        self.model_trained = True

    def load_model(self, save_file, custom_objects={'focal_loss_fixed': focal_loss()}):
        """
        Load a trained model from a file.
        
        Args:
            save_file (str): Path to the saved model file.
            custom_objects (dict): Custom objects for loading the model.
        """        
        self.model = tf.keras.models.load_model(save_file, custom_objects=custom_objects)
        self.model_trained = True

    def evaluate_model(self, verbose=2):
        """
        Evaluate the model using the test dataset.
        
        Args:
            verbose (int): Verbosity mode.
        
        Returns:
            evaluation (list): Evaluation metrics.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before evaluation. Please call load_dataset() first.")
        if not self.dataset_transformed:
            raise RuntimeError("Dataset must be transformed before evaluation. Please call transform_dataset() first.")
        if not self.model_trained:
            raise RuntimeError("Model must be trained or loaded before evaluation. Please call train_model() or load_model() first.")
        
        return self.model.evaluate([self.X_test_trans, self.mass_scaler.transform(self.mass_test)], self.y_test, sample_weight=self.weights_test, verbose=verbose)

    def predict(self, X_trans, mass_trans):
        """
        Make predictions using the trained model.
        
        Args:
            X_trans (np.ndarray): Transformed features.
            mass_trans (np.ndarray): Transformed masses.
        
        Returns:
            predictions (np.ndarray): Model predictions.
        """
        if not self.model_trained:
            raise RuntimeError("Model must be trained or loaded before making predictions. Please call train_model() or load_model() first.")
        
        return self.model.predict([X_trans, mass_trans])

    def calc_auc_all(self, mass_list=None, plot_roc=True, plot_auc=True):
        """
        Calculate the AUC score for all masses and optionally plot the ROC curve and AUC vs Mass figure.
        
        Args:
            mass_list (list): List of mass values to calculate AUC for. If None, calculate for all masses.
            plot_roc (bool): Whether to plot the ROC curve.
            plot_auc (bool): Whether to plot the AUC vs Mass figure.
        
        Returns:
            auc_df (pd.DataFrame): DataFrame containing mass values and corresponding AUC scores.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before calculating AUC. Please call load_dataset() first.")
        if not self.dataset_transformed:
            raise RuntimeError("Dataset must be transformed before calculating AUC. Please call transform_dataset() first.")
        if not self.model_trained:
            raise RuntimeError("Model must be trained or loaded before calculating AUC. Please call train_model() or load_model() first.")

        if mass_list is None:
            mass_list = self.unique_mass
        
        mass_auc = []
        if np.sum(self.weights_test < 0) > 0:
            print("Warning: Negative weights are detected. Only samples with positive weights are used for AUC calculation.")
        positive_weight_mask_test = self.weights_test > 0
        X_test_trans_tmp = self.X_test_trans[positive_weight_mask_test]
        mass_test_tmp = self.mass_test[positive_weight_mask_test]
        y_test_tmp = self.y_test[positive_weight_mask_test]
        weights_test_tmp = self.weights_test[positive_weight_mask_test]

        for mass_value in mass_list:
            mass_test_tmp[y_test_tmp == 0] = mass_value
            mask = np.all(mass_test_tmp == mass_value, axis=1)
            y_pred_test = self.predict(X_test_trans_tmp[mask], self.mass_scaler.transform(mass_test_tmp[mask]))

            if np.sum(mask) > 0:
                if plot_roc:
                    print(f'ROC Curve for Mass = {mass_value}:')
                auc = calc_auc(y_test_tmp[mask].ravel(), y_pred_test.ravel(), weights_test_tmp[mask].ravel(), plot_roc=plot_roc)
                mass_auc.append((mass_value, auc))

        mass_vals, auc_vals = zip(*mass_auc)

        auc_df = pd.DataFrame({'Mass': mass_vals, 'AUC': auc_vals})
        display(auc_df)

        if plot_auc:
            if len(self.mass_columns) == 1:
                with matplotlib.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
                    plt.figure(figsize=(8, 5))
                    plt.grid()
                    plt.plot(mass_vals, auc_vals, marker='o')
                    plt.xlabel('Mass')
                    plt.ylabel('AUC')
                    plt.title('AUC vs Mass')
                    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                    plt.show()
            elif len(self.mass_columns) == 2:
                mass1_vals = [mass[0] for mass in mass_vals]
                mass2_vals = [mass[1] for mass in mass_vals]
                mass1_unique = np.unique(mass1_vals)
                mass2_unique = np.unique(mass2_vals)
                auc_matrix = np.zeros((len(mass1_unique), len(mass2_unique)))
                for i, mass1 in enumerate(mass1_unique):
                    for j, mass2 in enumerate(mass2_unique):
                        auc = auc_df[(mass1_vals == mass1) & (mass2_vals == mass2)]['AUC'].values
                        auc_matrix[i, j] = auc[0] if len(auc) > 0 else np.nan
                plt.figure(figsize=(8, 5))
                sns.heatmap(auc_matrix, annot=True, cmap='coolwarm', xticklabels=mass2_unique, yticklabels=mass1_unique, fmt=".3f")
                plt.xlabel(f'{self.mass_columns[1]}')
                plt.ylabel(f'{self.mass_columns[0]}')
                plt.title('AUC vs Mass')
                plt.show()
            else:
                print("Warning: AUC vs Mass plot is not supported for more than 2 mass columns.")

        return auc_df

    def plot_score_all(self, mass_list=None, bins=50):
        """
        Plot the output score distribution for all masses.
        
        Args:
            mass_list (list): List of mass values to plot the score distribution for. If None, plot for all masses.
            bins (int): Number of bins for the histogram.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before plotting score distribution. Please call load_dataset() first.")
        if not self.dataset_transformed:
            raise RuntimeError("Dataset must be transformed before plotting score distribution. Please call transform_dataset() first.")
        if not self.model_trained:
            raise RuntimeError("Model must be trained or loaded before plotting score distribution. Please call train_model() or load_model() first.")
        
        if mass_list is None:
            mass_list = self.unique_mass

        mass_train_tmp = self.mass_train
        mass_test_tmp = self.mass_test

        for mass_value in mass_list:
            mass_train_tmp[self.y_train == 0] = mass_value
            mass_test_tmp[self.y_test == 0] = mass_value
            train_mask = np.all(mass_train_tmp == mass_value, axis=1)
            test_mask = np.all(mass_test_tmp == mass_value, axis=1)
            y_pred_train = self.predict(self.X_train_trans[train_mask], self.mass_scaler.transform(mass_train_tmp[train_mask]))
            y_pred_test = self.predict(self.X_test_trans[test_mask], self.mass_scaler.transform(mass_test_tmp[test_mask]))

            print(f'Output Score Distribution for Mass = {mass_value}:')
            plot_score(self.y_train[train_mask].ravel(), y_pred_train.ravel(), self.weights_train[train_mask].ravel(), self.y_test[test_mask].ravel(), y_pred_test.ravel(), self.weights_test[test_mask].ravel(), bins=bins)

    def plot_cut_efficiency_all(self, mass_list=None, signal_numbers=None, background_number=None, n_cuts=1000):
        """
        Plot the cut efficiency and signal significance for all masses.
        
        Args:
            mass_list (list): List of mass values to plot the cut efficiency for. If None, plot for all masses.
            signal_numbers (list): List of weighted numbers of signal events for each mass value. If None, use the weighted number of signal samples.
            background_number (float): Weighted number of background events. If None, use the weighted number of background samples.
            n_cuts (int): Number of cut values to evaluate.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before plotting cut efficiency. Please call load_dataset() first.")
        if not self.dataset_transformed:
            raise RuntimeError("Dataset must be transformed before plotting cut efficiency. Please call transform_dataset() first.")
        if not self.model_trained:
            raise RuntimeError("Model must be trained or loaded before plotting cut efficiency. Please call train_model() or load_model() first.")
        
        if mass_list is None:
            mass_list = self.unique_mass

        if signal_numbers is None:
            signal_numbers = [np.sum(self.weights_signal[np.all(self.mass_signal == mass_value, axis=1)]) for mass_value in mass_list]
        elif len(signal_numbers) != len(mass_list):
            raise ValueError("Number of signal_numbers must match the number of mass_list.")

        if background_number is None:
            background_number = self.background_number

        mass_test_tmp = self.mass_test

        for mass_value, signal_number in zip(mass_list, signal_numbers):
            mass_test_tmp[self.y_test == 0] = mass_value
            mask = np.all(mass_test_tmp == mass_value, axis=1)
            y_pred_test = self.predict(self.X_test_trans[mask], self.mass_scaler.transform(mass_test_tmp[mask]))

            print(f'Cut Efficiency for Mass = {mass_value}:')
            plot_cut_efficiency(self.y_test[mask].ravel(), y_pred_test.ravel(), self.weights_test[mask].ravel(), signal_number=signal_number, background_number=background_number, n_cuts=n_cuts)

    def calc_feature_importance_all(self, mass_list=None, sample_size=100000, steps=50):
        """
        Calculate feature importance for all masses using integrated gradients.
        
        Args:
            mass_list (list): List of mass values to calculate feature importance for. If None, calculate for all masses.
            sample_size (int): Number of samples to use for calculating feature importance.
            steps (int): Number of steps for integrated gradients.
        
        Returns:
            importance_dfs (list): DataFrames containing feature importance scores.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before calculating feature importance. Please call load_dataset() first.")
        if not self.dataset_transformed:
            raise RuntimeError("Dataset must be transformed before calculating feature importance. Please call transform_dataset() first.")
        if not self.model_trained:
            raise RuntimeError("Model must be trained or loaded before calculating feature importance. Please call train_model() or load_model() first.")
        
        if mass_list is None:
            mass_list = self.unique_mass

        mass_test_tmp = self.mass_test
        
        importance_dfs = []
        for mass_value in mass_list:
            mass_test_tmp[self.y_test == 0] = mass_value
            mask = np.all(mass_test_tmp == mass_value, axis=1)
            X_input = tf.convert_to_tensor(self.X_test_trans[mask], dtype=tf.float32)
            m_input = tf.convert_to_tensor(self.mass_scaler.transform(mass_test_tmp[mask]), dtype=tf.float32)
            weights = tf.convert_to_tensor(self.weights_test[mask], dtype=tf.float32)

            if X_input.shape[0] > sample_size:
                indices = np.random.choice(X_input.shape[0], sample_size, replace=False)
                X_input = tf.gather(X_input, indices)
                m_input = tf.gather(m_input, indices)
                weights = tf.gather(weights, indices)

            importance = calc_feature_importance(self.model, X_input, m_input, weights, steps=steps)
            importance_df = pd.DataFrame({'Feature': self.features, 'Importance': importance.numpy()})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            importance_dfs.append((mass_value, importance_df))

            styled_df = importance_df.style.background_gradient(subset=['Importance'], cmap='Blues')
            print(f"Feature Importance for Mass = {mass_value}:")
            display(styled_df)

        return importance_dfs
