import os
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
    weighted_cov = np.einsum('i,ij,ik->jk', weights, X_centered, X_centered)
    weighted_std = np.sqrt(np.diag(weighted_cov))
    weighted_corr = weighted_cov / np.outer(weighted_std, weighted_std)
    return weighted_corr

def calc_auc(y_true, y_pred, weights, plot_show=True, save_fig=False, filename=None):
    """
    Calculate the AUC score and optionally plot the ROC curve.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        weights (np.ndarray): Sample weights.
        plot_show (bool): Whether to display the plot.
        save_fig (bool): Whether to save the plot as images.
        filename (str): Filename for the saved images.
    
    Returns:
        auc (float): AUC score.
    """
    auc = roc_auc_score(y_true, y_pred, sample_weight=weights)
    if plot_show or save_fig:
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
            if save_fig:
                os.makedirs('figure/', exist_ok=True)
                if filename is None:
                    filename = 'roc_curve'
                plt.savefig(f'figure/{filename}.png')
                plt.savefig(f'figure/{filename}.pdf')
            if plot_show:
                plt.show()
            else:
                plt.close()
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

def plot_score(y_train, y_pred_train, weights_train, y_test, y_pred_test, weights_test, bins=50, plot_show=True, save_fig=False, filename=None):
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
        plot_show (bool): Whether to display the plot.
        save_fig (bool): Whether to save the plot as images.
        filename (str): Filename for the saved images.
    """
    with matplotlib.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
        plt.figure(figsize=(8, 5))

        plt.hist(y_pred_test[y_test == 1], bins=bins, range=[0, 1], alpha=0.5, label='Signal (Test)', weights=weights_test[y_test == 1], color='blue', density=True)
        plt.hist(y_pred_test[y_test == 0], bins=bins, range=[0, 1], alpha=0.5, label='Background (Test)', weights=weights_test[y_test == 0], color='red', density=True)

        if y_train is not None and len(y_train) > 0:
            hist_signal_train, bin_edges = np.histogram(y_pred_train[y_train == 1], bins=bins, range=[0, 1], weights=weights_train[y_train == 1], density=True)
            hist_background_train, _ = np.histogram(y_pred_train[y_train == 0], bins=bins, range=[0, 1], weights=weights_train[y_train == 0], density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            plt.scatter(bin_centers, hist_signal_train, label='Signal (Train)', color='blue', marker='o', s=10)
            plt.scatter(bin_centers, hist_background_train, label='Background (Train)', color='red', marker='o', s=10)

        plt.xlabel('Output Score')
        plt.ylabel('Density')
        plt.title('Output Score Distribution')
        plt.legend()
        plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
        plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
        if save_fig:
            os.makedirs('figure/', exist_ok=True)
            if filename is None:
                filename = 'output_score_distribution'
            plt.savefig(f'figure/{filename}.png')
            plt.savefig(f'figure/{filename}.pdf')
        if plot_show:
            plt.show()
        else:
            plt.close()

def plot_cut_efficiency(y_true, y_pred, weights, signal_number=None, background_number=None, n_cuts=1000, plot_show=True, save_fig=False, filename=None):
    """
    Plot the cut efficiency and signal significance as a function of cut value.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        weights (np.ndarray): Sample weights.
        signal_number (float): Weighted number of signal samples.
        background_number (float): Weighted number of background samples.
        n_cuts (int): Number of cut values to evaluate.
        plot_show (bool): Whether to display the plot.
        save_fig (bool): Whether to save the plot as images.
        filename (str): Filename for the saved images.
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
        if save_fig:
            os.makedirs('figure/', exist_ok=True)
            if filename is None:
                filename = 'cut_efficiency'
            plt.savefig(f'figure/{filename}.png')
            plt.savefig(f'figure/{filename}.pdf')
        if plot_show:
            plt.show()
        else:
            plt.close()

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
    delta = X_input - baseline
    interpolated_inputs = [baseline + (i / steps) * delta for i in range(steps + 1)]

    gradients = []
    for i in range(steps + 1):
        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs[i])
            predictions = model([interpolated_inputs[i], m_input])
        gradients.append(tape.gradient(predictions, interpolated_inputs[i]))

    integrated_gradients = tf.reduce_mean(gradients, axis=0) * delta
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

    def load_dataset(self, signal_path=None, background_path=None, experiment_path=None, signal_df=None, background_df=None, experiment_df=None, pre_selection=None, balance_signal_on_mass=False, background_mass_distribution='discrete', balance_signal_background=False, test_size=0.2):
        """
        Load datasets from CSV files or DataFrames and split into training and test datasets. (CSV table headers should match the names of the features, mass_columns, and weight_column.)
        
        Args:
            signal_path (str): Path to the signal dataset file.
            background_path (str): Path to the background dataset file.
            experiment_path (str): Path to the experiment dataset file.
            signal_df (pd.DataFrame): DataFrame containing the signal dataset.
            background_df (pd.DataFrame): DataFrame containing the background dataset.
            experiment_df (pd.DataFrame): DataFrame containing the experiment dataset.
            pre_selection (callable): A function to apply pre-selection cuts to the data. It should take a DataFrame and return a boolean mask.
            balance_signal_on_mass (bool): Whether to balance the weights of the signal samples, making the sum of the weights equal for all masses when training the model.
            background_mass_distribution (str): Distribution type for the mass of background ('discrete', 'continuous', or 'original'). If 'discrete', the mass is sampled from the discrete distribution of the signal masses. If 'continuous', the mass is sampled from a uniform distribution within the range of the signal masses. If 'original', the original background mass distribution is used.
            balance_signal_background (bool): Whether to balance the weights of the signal and background samples, making the sum of the weights equal for both when training the model.
            test_size (float): Proportion of the dataset to include in the test split.
        """
        self.X_signal = None
        self.mass_signal = None
        self.unique_mass = []
        self.weights_signal = None
        self.signal_numbers_original = []

        self.X_background = None
        self.mass_background = None
        self.weights_background = None
        self.background_number_original = 0
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
        
        if signal_df is None and isinstance(signal_path, str):
            signal_df = pd.read_csv(signal_path)
        if signal_df is not None:
            if pre_selection is not None:
                signal_df = signal_df[pre_selection(signal_df)]
            self.X_signal = signal_df[self.features].values
            self.mass_signal = signal_df[self.mass_columns].values
            self.unique_mass = [list(mass) for mass in np.unique(self.mass_signal, axis=0)]
            y_signal = np.ones(len(signal_df))
            self.weights_signal = signal_df[self.weight_column].values
            self.signal_numbers_original = [np.sum(self.weights_signal[np.all(self.mass_signal == mass_value, axis=1)]) for mass_value in self.unique_mass]
            if balance_signal_on_mass:
                signal_weight_sum_each_mass = np.sum(self.weights_signal) / len(self.unique_mass)
                for mass_value in self.unique_mass:
                    mask = np.all(self.mass_signal == mass_value, axis=1)
                    self.weights_signal[mask] *= signal_weight_sum_each_mass / np.sum(self.weights_signal[mask])

        if background_df is None and isinstance(background_path, str):
            background_df = pd.read_csv(background_path)
        if background_df is not None:
            if pre_selection is not None:
                background_df = background_df[pre_selection(background_df)]
            self.X_background = background_df[self.features].values
            if background_mass_distribution == 'discrete':
                if signal_df is not None:
                    mass_weighted_counts = signal_df.groupby(self.mass_columns)[self.weight_column].sum()
                    mass_probabilities = mass_weighted_counts / mass_weighted_counts.sum()
                    if np.any(mass_probabilities < 0):
                        negative_mass = mass_weighted_counts.index[mass_probabilities < 0]
                        raise ValueError(f"A negative sum of weights is detected for mass: {negative_mass}")
                    chosen_masses = np.random.choice(mass_weighted_counts.index, size=len(background_df), p=mass_probabilities)
                    self.mass_background = np.array([[mass] if np.isscalar(mass) else list(mass) for mass in chosen_masses])
            elif background_mass_distribution == 'continuous':
                if signal_df is not None:
                    self.mass_background = np.random.uniform(low=self.mass_signal.min(axis=0), high=self.mass_signal.max(axis=0), size=(len(background_df), len(self.mass_columns)))
            elif background_mass_distribution == 'original':
                self.mass_background = background_df[self.mass_columns].values
            else:
                raise ValueError("Invalid background_mass_distribution. Choose 'discrete', 'continuous', or 'original'.")
            y_background = np.zeros(len(background_df))
            self.weights_background = background_df[self.weight_column].values
            self.background_number_original = np.sum(self.weights_background)
            if balance_signal_background and signal_df is not None:
                signal_weight_sum = np.sum(self.weights_signal)
                background_weight_sum = np.sum(self.weights_background)
                self.weights_background = self.weights_background / background_weight_sum * signal_weight_sum
            if self.background_type_column is not None:
                self.background_types = background_df[self.background_type_column].values
                self.unique_background_types = np.unique(self.background_types)

        if experiment_df is None and isinstance(experiment_path, str):
            experiment_df = pd.read_csv(experiment_path)
        if experiment_df is not None:
            if pre_selection is not None:
                experiment_df = experiment_df[pre_selection(experiment_df)]
            self.X_experiment = experiment_df[self.features].values
            self.weights_experiment = experiment_df[self.weight_column].values

        if signal_df is not None and background_df is not None:
            X = np.vstack((self.X_signal, self.X_background))
            mass = np.vstack((self.mass_signal, self.mass_background))
            y = np.hstack((y_signal, y_background))
            weights = np.hstack((self.weights_signal, self.weights_background))
            self.X_train, self.X_test, self.mass_train, self.mass_test, self.y_train, self.y_test, self.weights_train, self.weights_test = train_test_split(X, mass, y, weights, test_size=test_size, random_state=self.random_seed)

        self.dataset_loaded = True
        self.dataset_transformed = False

    def plot_feature_distribution(self, feature_list=None, mass_list=None, background_type_list=None, bins=50, density=True, log_scale=False, background_bar_stacked=True, plot_show=True, save_fig=False):
        """
        Plot the feature distribution.
        
        Args:
            feature_list (list): List of features or tuples (feature, min, max) to plot the distribution for. If None, plot for all features.
            mass_list (list): List of signal mass values to plot the feature distribution for. If None, plot for all masses.
            background_type_list (list): List of background types to plot the feature distribution for. If None, plot for all types.
            bins (int): Number of bins for the histogram.
            density (bool): Whether to normalize the histogram to form a density plot.
            log_scale (bool): Whether to use a logarithmic scale for the y-axis.
            background_bar_stacked (bool): Whether to stack the background bars for different types.
            plot_show (bool): Whether to display the plots.
            save_fig (bool): Whether to save the plots as images.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before plotting feature distribution. Please call load_dataset() first.")
        
        if feature_list is None:
            feature_list = self.features
        if mass_list is None:
            mass_list = self.unique_mass
        else:
            mass_list = [[mass] if np.isscalar(mass) else mass for mass in mass_list]
        if background_type_list is None:
            background_type_list = self.unique_background_types

        for feature_item in feature_list:
            if isinstance(feature_item, tuple):
                feature, min_val, max_val = feature_item
                feature_idx = self.features.index(feature)
                bin_edges = np.linspace(min_val, max_val, bins + 1)
            else:
                feature = feature_item
                feature_idx = self.features.index(feature)
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
                
                for mass_value in mass_list:
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
                
                if save_fig:
                    os.makedirs('figure/', exist_ok=True)
                    filename = f'feature_distribution_{feature}'
                    plt.savefig(f'figure/{filename}.png')
                    plt.savefig(f'figure/{filename}.pdf')
                if plot_show:
                    plt.show()
                else:
                    plt.close()
    
    def calc_feature_correlation_all(self, mass_list=None, plot_show=True, save_fig=False):
        """
        Calculate the feature correlation for all masses.
        
        Args:
            mass_list (list): List of signal mass values to calculate feature correlation for. If None, calculate for all masses.
            plot_show (bool): Whether to display the plots.
            save_fig (bool): Whether to save the plots as images.
        
        Returns:
            correlation_dfs (list): DataFrames containing feature correlation for signal and background.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before calculating feature correlation. Please call load_dataset() first.")
        
        if mass_list is None:
            mass_list = self.unique_mass
        else:
            mass_list = [[mass] if np.isscalar(mass) else mass for mass in mass_list]
        
        correlation_dfs = []
        for mass_value in mass_list:
            signal_mask = np.all(self.mass_signal == mass_value, axis=1)
            correlation_signal = calc_feature_correlation(self.X_signal[signal_mask], self.weights_signal[signal_mask])
            correlation_df_signal = pd.DataFrame(correlation_signal, columns=self.features, index=self.features)
            correlation_dfs.append((mass_value, correlation_df_signal))

            plt.figure(figsize=(8, 5))
            sns.heatmap(correlation_df_signal * 100, annot=True, cmap='coolwarm', vmin=-100, vmax=100, fmt=".0f")
            plt.title(f"Signal Feature Correlation for Mass = {mass_value} (×100)")
            if save_fig:
                os.makedirs('figure/', exist_ok=True)
                mass_str = ','.join(map(str, mass_value))
                filename = f'feature_correlation_signal_{mass_str}'
                plt.savefig(f'figure/{filename}.png')
                plt.savefig(f'figure/{filename}.pdf')
            if plot_show:
                plt.show()
            else:
                plt.close()

        if self.X_background is not None:
            correlation_background = calc_feature_correlation(self.X_background, self.weights_background)
            correlation_df_background = pd.DataFrame(correlation_background, columns=self.features, index=self.features)
            correlation_dfs.append(('Background', correlation_df_background))

            plt.figure(figsize=(8, 5))
            sns.heatmap(correlation_df_background * 100, annot=True, cmap='coolwarm', vmin=-100, vmax=100, fmt=".0f")
            plt.title("Background Feature Correlation (×100)")
            if save_fig:
                os.makedirs('figure/', exist_ok=True)
                filename = 'feature_correlation_background'
                plt.savefig(f'figure/{filename}.png')
                plt.savefig(f'figure/{filename}.pdf')
            if plot_show:
                plt.show()
            else:
                plt.close()

        if self.X_experiment is not None:
            correlation_experiment = calc_feature_correlation(self.X_experiment, self.weights_experiment)
            correlation_df_experiment = pd.DataFrame(correlation_experiment, columns=self.features, index=self.features)
            correlation_dfs.append(('Data', correlation_df_experiment))

            plt.figure(figsize=(8, 5))
            sns.heatmap(correlation_df_experiment * 100, annot=True, cmap='coolwarm', vmin=-100, vmax=100, fmt=".0f")
            plt.title("Data Feature Correlation (×100)")
            if save_fig:
                os.makedirs('figure/', exist_ok=True)
                filename = 'feature_correlation_data'
                plt.savefig(f'figure/{filename}.png')
                plt.savefig(f'figure/{filename}.pdf')
            if plot_show:
                plt.show()
            else:
                plt.close()
        
        return correlation_dfs

    def plot_mass_distribution(self, bins=100, density=True, plot_show=True, save_fig=False):
        """
        Plot the mass distribution.
        
        Args:
            bins (int): Number of bins for the histogram.
            density (bool): Whether to normalize the histogram to form a density plot.
            plot_show (bool): Whether to display the plot.
            save_fig (bool): Whether to save the plot as images.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before plotting mass distribution. Please call load_dataset() first.")
        
        with matplotlib.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
            for i, mass_column in enumerate(self.mass_columns):
                plt.figure(figsize=(8, 5))
                plt.hist(self.mass_signal[:, i], bins=bins, histtype='step', label='Signal', weights=self.weights_signal, density=density)
                plt.hist(self.mass_background[:, i], bins=bins, histtype='step', label='Background', weights=self.weights_background, density=density)
                plt.xlabel(mass_column)
                plt.ylabel('Density' if density else 'Events')
                plt.title(f'{mass_column} Distribution')
                plt.legend()
                plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                if save_fig:
                    os.makedirs('figure/', exist_ok=True)
                    filename = f'{mass_column}_distribution'
                    plt.savefig(f'figure/{filename}.png')
                    plt.savefig(f'figure/{filename}.pdf')
                if plot_show:
                    plt.show()
                else:
                    plt.close()

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

    def train_model(self, model=None, ignore_negative_weights=True, epochs=20, batch_size=1024, validation_split=0.2, verbose=2, model_path=None):
        """
        Train the model using the training dataset.
        
        Args:
            model (tf.keras.Model): Model to train. If None, use the default PNNplus model.
            ignore_negative_weights (bool): Whether to ignore samples with negative weights during training.
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of the training dataset to be used as validation dataset.
            verbose (int): Verbosity mode.
            model_path (str): Path to save the trained model.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before training the model. Please call load_dataset() first.")
        if not self.dataset_transformed:
            raise RuntimeError("Dataset must be transformed before training the model. Please call transform_dataset() first.")
        
        if model is None:
            self.model = pnnplus_model(len(self.features), len(self.mass_columns))
        else:
            self.model = model

        X_train_trans_tmp = self.X_train_trans
        mass_train_tmp = self.mass_train
        y_train_tmp = self.y_train
        weights_train_tmp = self.weights_train
        if not ignore_negative_weights and np.sum(weights_train_tmp < 0) > 0:
            print("Warning: Negative weights are detected. This may cause problems depending on the specific model architecture.")
        if ignore_negative_weights:
            positive_weight_mask_train = weights_train_tmp > 0
            X_train_trans_tmp = X_train_trans_tmp[positive_weight_mask_train]
            mass_train_tmp = mass_train_tmp[positive_weight_mask_train]
            y_train_tmp = y_train_tmp[positive_weight_mask_train]
            weights_train_tmp = weights_train_tmp[positive_weight_mask_train]

        self.model.fit([X_train_trans_tmp, self.mass_scaler.transform(mass_train_tmp)], y_train_tmp, sample_weight=weights_train_tmp, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)

        if model_path is not None:
            self.model.save(model_path)
        
        self.model_trained = True

    def load_model(self, model_path, custom_objects={'focal_loss_fixed': focal_loss()}):
        """
        Load a trained model from a file.
        
        Args:
            model_path (str): Path to the saved model file.
            custom_objects (dict): Custom objects for loading the model.
        """        
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        self.model_trained = True

    def evaluate_model(self, ignore_negative_weights=True, batch_size=1024, verbose=2):
        """
        Evaluate the model using the test dataset.
        
        Args:
            ignore_negative_weights (bool): Whether to ignore samples with negative weights during evaluation.
            batch_size (int): Batch size for evaluation.
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
        
        X_test_trans_tmp = self.X_test_trans
        mass_test_tmp = self.mass_test
        y_test_tmp = self.y_test
        weights_test_tmp = self.weights_test
        if not ignore_negative_weights and np.sum(weights_test_tmp < 0) > 0:
            print("Warning: Negative weights are detected. This may cause problems depending on the specific model architecture.")
        if ignore_negative_weights:
            positive_weight_mask_test = weights_test_tmp > 0
            X_test_trans_tmp = X_test_trans_tmp[positive_weight_mask_test]
            mass_test_tmp = mass_test_tmp[positive_weight_mask_test]
            y_test_tmp = y_test_tmp[positive_weight_mask_test]
            weights_test_tmp = weights_test_tmp[positive_weight_mask_test]

        return self.model.evaluate([X_test_trans_tmp, self.mass_scaler.transform(mass_test_tmp)], y_test_tmp, sample_weight=weights_test_tmp, batch_size=batch_size, verbose=verbose)

    def predict(self, X_trans: np.ndarray, mass_trans: np.ndarray, batch_size=1024, verbose=2) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X_trans (np.ndarray): Transformed features.
            mass_trans (np.ndarray): Transformed masses.
            batch_size (int): Batch size for prediction.
            verbose (int): Verbosity mode.
        
        Returns:
            predictions (np.ndarray): Model predictions.
        """
        if not self.model_trained:
            raise RuntimeError("Model must be trained or loaded before making predictions. Please call train_model() or load_model() first.")
        
        return self.model.predict([X_trans, mass_trans], batch_size=batch_size, verbose=verbose)

    def calc_auc_all(self, mass_list=None, sample_size=1000000, plot_show=True, save_fig=False):
        """
        Calculate the AUC score for all masses and optionally plot the ROC curve and AUC vs Mass figure.
        
        Args:
            mass_list (list): List of mass values to calculate AUC for. If None, calculate for all masses.
            sample_size (int): Number of samples to use. If greater than the total number of samples, use all samples.
            plot_show (bool): Whether to display the plots.
            save_fig (bool): Whether to save the plots as images.
        
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
        else:
            mass_list = [[mass] if np.isscalar(mass) else mass for mass in mass_list]
        
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
            mask_cnt = np.sum(mask)
            if mask_cnt > 0:
                if mask_cnt > sample_size:
                    indices = np.random.choice(mask_cnt, sample_size, replace=False)
                else:
                    indices = np.arange(mask_cnt)

                X_test_input = X_test_trans_tmp[mask][indices]
                mass_test_input = self.mass_scaler.transform(mass_test_tmp[mask][indices])
                y_test_input = y_test_tmp[mask][indices].ravel()
                y_pred_test_input = self.predict(X_test_input, mass_test_input).ravel()
                weights_test_input = weights_test_tmp[mask][indices].ravel()

                print(f'ROC Curve for Mass = {mass_value}:')
                mass_str = ','.join(map(str, mass_value))
                auc = calc_auc(y_test_input, y_pred_test_input, weights_test_input, plot_show=plot_show, save_fig=save_fig, filename=f'roc_curve_{mass_str}')
                mass_auc.append((mass_value, auc))

        mass_vals, auc_vals = zip(*mass_auc)

        auc_df = pd.DataFrame({'Mass': mass_vals, 'AUC': auc_vals})
        display(auc_df)

        if plot_show or save_fig:
            if len(self.mass_columns) == 1:
                with matplotlib.rc_context({'xtick.direction': 'in', 'ytick.direction': 'in'}):
                    plt.figure(figsize=(8, 5))
                    plt.grid()
                    plt.plot([mass[0] for mass in mass_vals], auc_vals, marker='o')
                    plt.xlabel('Mass')
                    plt.ylabel('AUC')
                    plt.title('AUC vs Mass')
                    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
                    if save_fig:
                        os.makedirs('figure/', exist_ok=True)
                        plt.savefig('figure/auc_vs_mass.png')
                        plt.savefig('figure/auc_vs_mass.pdf')
                    if plot_show:
                        plt.show()
                    else:
                        plt.close()
            elif len(self.mass_columns) == 2:
                mass1_vals = np.array([mass[0] for mass in mass_vals])
                mass2_vals = np.array([mass[1] for mass in mass_vals])
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
                if save_fig:
                    os.makedirs('figure/', exist_ok=True)
                    plt.savefig('figure/auc_vs_mass.png')
                    plt.savefig('figure/auc_vs_mass.pdf')
                if plot_show:
                    plt.show()
                else:
                    plt.close()
            else:
                print("Warning: AUC vs Mass plot is not supported for more than 2 mass columns.")

        return auc_df

    def plot_score_all(self, mass_list=None, sample_size=1000000, bins=50, plot_show=True, save_fig=False):
        """
        Plot the output score distribution for all masses.
        
        Args:
            mass_list (list): List of mass values to plot the score distribution for. If None, plot for all masses.
            sample_size (int): Number of samples to use. If greater than the total number of samples, use all samples.
            bins (int): Number of bins for the histogram.
            plot_show (bool): Whether to display the plots.
            save_fig (bool): Whether to save the plots as images.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before plotting score distribution. Please call load_dataset() first.")
        if not self.dataset_transformed:
            raise RuntimeError("Dataset must be transformed before plotting score distribution. Please call transform_dataset() first.")
        if not self.model_trained:
            raise RuntimeError("Model must be trained or loaded before plotting score distribution. Please call train_model() or load_model() first.")
        
        if mass_list is None:
            mass_list = self.unique_mass
        else:
            mass_list = [[mass] if np.isscalar(mass) else mass for mass in mass_list]

        mass_train_tmp = self.mass_train.copy()
        mass_test_tmp = self.mass_test.copy()

        for mass_value in mass_list:
            if self.y_train is not None and len(self.y_train) > 0:
                mass_train_tmp[self.y_train == 0] = mass_value
                train_mask = np.all(mass_train_tmp == mass_value, axis=1)
                train_mask_cnt = np.sum(train_mask)
                if train_mask_cnt > sample_size:
                    train_indices = np.random.choice(train_mask_cnt, sample_size, replace=False)
                else:
                    train_indices = np.arange(train_mask_cnt)
                X_train_input = self.X_train_trans[train_mask][train_indices]
                mass_train_input = self.mass_scaler.transform(mass_train_tmp[train_mask][train_indices])
                y_train_input = self.y_train[train_mask][train_indices].ravel()
                y_pred_train_input = self.predict(X_train_input, mass_train_input).ravel()
                weights_train_input = self.weights_train[train_mask][train_indices].ravel()
            else:
                y_train_input = None
                y_pred_train_input = None
                weights_train_input = None

            mass_test_tmp[self.y_test == 0] = mass_value
            test_mask = np.all(mass_test_tmp == mass_value, axis=1)
            test_mask_cnt = np.sum(test_mask)
            if test_mask_cnt > sample_size:
                test_indices = np.random.choice(test_mask_cnt, sample_size, replace=False)
            else:
                test_indices = np.arange(test_mask_cnt)
            X_test_input = self.X_test_trans[test_mask][test_indices]
            mass_test_input = self.mass_scaler.transform(mass_test_tmp[test_mask][test_indices])
            y_test_input = self.y_test[test_mask][test_indices].ravel()
            y_pred_test_input = self.predict(X_test_input, mass_test_input).ravel()
            weights_test_input = self.weights_test[test_mask][test_indices].ravel()

            print(f'Output Score Distribution for Mass = {mass_value}:')
            mass_str = ','.join(map(str, mass_value))
            plot_score(y_train_input, y_pred_train_input, weights_train_input, y_test_input, y_pred_test_input, weights_test_input, bins=bins, plot_show=plot_show, save_fig=save_fig, filename=f'output_score_distribution_{mass_str}')

    def plot_cut_efficiency_all(self, mass_list=None, signal_numbers=None, background_number=None, sample_size=1000000, n_cuts=1000, plot_show=True, save_fig=False):
        """
        Plot the cut efficiency and signal significance for all masses.
        
        Args:
            mass_list (list): List of mass values to plot the cut efficiency for. If None, plot for all masses.
            signal_numbers (list): List of weighted numbers of signal samples for each mass value. If None, use the weighted numbers of signal samples in the original dataset.
            background_number (float): Weighted number of background samples. If None, use the weighted number of background samples in the original dataset.
            sample_size (int): Number of samples to use. If greater than the total number of samples, use all samples.
            n_cuts (int): Number of cut values to evaluate.
            plot_show (bool): Whether to display the plots.
            save_fig (bool): Whether to save the plots as images.
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset must be loaded before plotting cut efficiency. Please call load_dataset() first.")
        if not self.dataset_transformed:
            raise RuntimeError("Dataset must be transformed before plotting cut efficiency. Please call transform_dataset() first.")
        if not self.model_trained:
            raise RuntimeError("Model must be trained or loaded before plotting cut efficiency. Please call train_model() or load_model() first.")
        
        if mass_list is None:
            mass_list = self.unique_mass
        else:
            mass_list = [[mass] if np.isscalar(mass) else mass for mass in mass_list]

        if signal_numbers is None:
            signal_numbers = self.signal_numbers_original
        elif len(signal_numbers) != len(mass_list):
            raise ValueError("Number of signal_numbers must match the number of mass_list.")

        if background_number is None:
            background_number = self.background_number_original

        mass_test_tmp = self.mass_test.copy()

        for mass_value, signal_number in zip(mass_list, signal_numbers):
            mass_test_tmp[self.y_test == 0] = mass_value
            mask = np.all(mass_test_tmp == mass_value, axis=1)
            mask_cnt = np.sum(mask)
            if mask_cnt > sample_size:
                indices = np.random.choice(mask_cnt, sample_size, replace=False)
            else:
                indices = np.arange(mask_cnt)

            X_test_input = self.X_test_trans[mask][indices]
            mass_test_input = self.mass_scaler.transform(mass_test_tmp[mask][indices])
            y_test_input = self.y_test[mask][indices].ravel()
            y_pred_test_input = self.predict(X_test_input, mass_test_input).ravel()
            weights_test_input = self.weights_test[mask][indices].ravel()

            print(f'Cut Efficiency for Mass = {mass_value}:')
            mass_str = ','.join(map(str, mass_value))
            plot_cut_efficiency(y_test_input, y_pred_test_input, weights_test_input, signal_number=signal_number, background_number=background_number, n_cuts=n_cuts, plot_show=plot_show, save_fig=save_fig, filename=f'cut_efficiency_{mass_str}')

    def calc_feature_importance_all(self, mass_list=None, sample_size=100000, steps=50):
        """
        Calculate feature importance for all masses using integrated gradients.
        
        Args:
            mass_list (list): List of mass values to calculate feature importance for. If None, calculate for all masses.
            sample_size (int): Number of samples to use. If greater than the total number of samples, use all samples.
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
        else:
            mass_list = [[mass] if np.isscalar(mass) else mass for mass in mass_list]

        mass_test_tmp = self.mass_test.copy()
        
        importance_dfs = []
        for mass_value in mass_list:
            mass_test_tmp[self.y_test == 0] = mass_value
            mask = np.all(mass_test_tmp == mass_value, axis=1)
            mask_cnt = np.sum(mask)
            if mask_cnt > sample_size:
                indices = np.random.choice(mask_cnt, sample_size, replace=False)
            else:
                indices = np.arange(mask_cnt)
            X_test_input = tf.convert_to_tensor(self.X_test_trans[mask][indices], dtype=tf.float32)
            m_test_input = tf.convert_to_tensor(self.mass_scaler.transform(mass_test_tmp[mask][indices]), dtype=tf.float32)
            weights_test_input = tf.convert_to_tensor(self.weights_test[mask][indices], dtype=tf.float32)

            importance = calc_feature_importance(self.model, X_test_input, m_test_input, weights_test_input, steps=steps)
            importance_df = pd.DataFrame({'Feature': self.features, 'Importance': importance.numpy()})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            importance_dfs.append((mass_value, importance_df))

            styled_df = importance_df.style.background_gradient(subset=['Importance'], cmap='Blues')
            print(f"Feature Importance for Mass = {mass_value}:")
            display(styled_df)

        return importance_dfs
