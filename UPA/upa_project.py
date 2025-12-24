# upa_project.py
"""
Unsupervised Prediction Alignment (UPA) Implementation
for Breast Cancer Wisconsin (Diagnostic) Dataset

Based on: "Automatic correction of performance drift under acquisition shift 
in medical image classification" - Nature Communications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import interpolate
from scipy.stats import wasserstein_distance, ks_2samp
import warnings
warnings.filterwarnings('ignore')
import os

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== 1. LOAD AND ANALYZE DATASET ====================

def load_and_analyze_dataset():
    """
    Load and analyze the Breast Cancer Wisconsin dataset
    """
    print("=" * 80)
    print("BREAST CANCER WISCONSIN DIAGNOSTIC DATASET")
    print("=" * 80)
    
    # Load the dataset
    try:
        df = pd.read_csv('data.csv')
        print("✓ Dataset loaded successfully")
    except FileNotFoundError:
        print("✗ ERROR: data.csv not found!")
        print("\nPlease ensure:")
        print("1. The file 'data.csv' is in the same folder as this script")
        print("2. You have extracted it from the zip file")
        return None
    
    # Display basic information
    print(f"\nDataset Shape: {df.shape}")
    print(f"Samples: {df.shape[0]}, Features: {df.shape[1] - 2} (excluding 'id' and 'diagnosis')")
    
    # Check for missing values
    print("\nMissing Values:")
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if len(missing_cols) > 0:
        print(missing_cols)
    else:
        print("No missing values found ✓")
    
    # Class distribution
    print("\n" + "-" * 40)
    print("CLASS DISTRIBUTION")
    print("-" * 40)
    
    class_counts = df['diagnosis'].value_counts()
    print(class_counts)
    
    benign_count = class_counts.get('B', 0)
    malignant_count = class_counts.get('M', 0)
    total = benign_count + malignant_count
    
    print(f"\nBenign (B): {benign_count} ({benign_count/total*100:.1f}%)")
    print(f"Malignant (M): {malignant_count} ({malignant_count/total*100:.1f}%)")
    
    # Feature overview
    print("\n" + "-" * 40)
    print("FEATURE OVERVIEW")
    print("-" * 40)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nFeature names (first 10):")
    print(df.columns[:10].tolist())
    
    # Statistics for key features
    print("\nStatistics for key features:")
    key_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
    if all(feat in df.columns for feat in key_features):
        print(df[key_features].describe())
    
    return df

# ==================== 2. PREPROCESS DATA ====================

def preprocess_data(df):
    """
    Preprocess the dataset for the UPA experiment
    """
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    # 1. Drop unnecessary columns
    df_clean = df.copy()
    
    # Remove 'id' column if exists
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop('id', axis=1)
    
    # Remove any unnamed columns
    unnamed_cols = [col for col in df_clean.columns if 'Unnamed' in col]
    if unnamed_cols:
        df_clean = df_clean.drop(unnamed_cols, axis=1)
        print(f"Removed unnamed columns: {unnamed_cols}")
    
    # 2. Encode target variable
    # B (Benign) = 0, M (Malignant) = 1
    le = LabelEncoder()
    df_clean['diagnosis_encoded'] = le.fit_transform(df_clean['diagnosis'])
    
    print(f"Target encoding: {list(le.classes_)} -> {list(le.transform(le.classes_))}")
    
    # 3. Separate features and target
    X = df_clean.drop(['diagnosis', 'diagnosis_encoded'], axis=1)
    y = df_clean['diagnosis_encoded']
    
    # 4. Convert to numpy arrays
    X_np = X.values
    y_np = y.values
    
    print(f"\nProcessed dataset shape: X={X_np.shape}, y={y_np.shape}")
    print(f"Number of features: {X_np.shape[1]}")
    print(f"Feature names: {list(X.columns[:5])}...")
    
    return X_np, y_np, X.columns.tolist(), le

# ==================== 3. UPA ALGORITHM IMPLEMENTATION ====================

class UnsupervisedPredictionAlignment:
    """
    Implementation of Unsupervised Prediction Alignment (UPA)
    for correcting performance drift due to acquisition shift
    """
    
    def __init__(self, n_bins=50, method='linear'):
        """
        Initialize UPA model
        
        Parameters:
        -----------
        n_bins : int
            Number of bins for histogram calculation
        method : str
            Interpolation method ('linear', 'nearest', 'cubic')
        """
        self.n_bins = n_bins
        self.method = method
        self.mapping_function = None
        self.reference_cdf = None
        self.target_cdf = None
        self.reference_bins = None
        self.target_bins = None

    def fit(self, reference_predictions, target_predictions):
        """
        Learn mapping function from target to reference domain
        
        Parameters:
        -----------
        reference_predictions : array-like
            Predictions from reference domain
        target_predictions : array-like
            Predictions from target domain for alignment
        """
        # Ensure predictions are numpy arrays
        ref_preds = np.array(reference_predictions).flatten()
        target_preds = np.array(target_predictions).flatten()
        
        # Clip predictions to [0, 1] range
        ref_preds = np.clip(ref_preds, 0, 1)
        target_preds = np.clip(target_preds, 0, 1)
        
        # Calculate histograms
        ref_hist, self.reference_bins = np.histogram(
            ref_preds, 
            bins=self.n_bins, 
            range=(0, 1), 
            density=True
        )
        target_hist, self.target_bins = np.histogram(
            target_preds, 
            bins=self.n_bins, 
            range=(0, 1), 
            density=True
        )
        
        # Calculate CDFs (Cumulative Distribution Functions)
        self.reference_cdf = np.cumsum(ref_hist) / np.sum(ref_hist)
        self.target_cdf = np.cumsum(target_hist) / np.sum(target_hist)
        
        # Ensure CDFs are strictly increasing (avoid plateaus)
        self.reference_cdf = np.maximum.accumulate(self.reference_cdf)
        self.target_cdf = np.maximum.accumulate(self.target_cdf)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        self.reference_cdf = np.clip(self.reference_cdf, eps, 1-eps)
        self.target_cdf = np.clip(self.target_cdf, eps, 1-eps)
        
        # Create mapping function
        self.mapping_function = interpolate.interp1d(
            self.target_cdf,
            self.reference_bins[:-1],
            kind=self.method,
            bounds_error=False,
            fill_value=(self.reference_bins[0], self.reference_bins[-2]),
            assume_sorted=True
        )
        
        # Store statistics
        self.wasserstein_before = wasserstein_distance(ref_preds, target_preds)
        self.ks_statistic, self.ks_pvalue = ks_2samp(ref_preds, target_preds)
        
        return self
    
    def transform(self, predictions):
        """
        Transform predictions using learned mapping
        
        Parameters:
        -----------
        predictions : array-like
            Predictions to align
            
        Returns:
        --------
        aligned_predictions : numpy array
            Aligned predictions
        """
        if self.mapping_function is None:
            raise ValueError("Model must be fitted before transformation")
        
        preds = np.array(predictions).flatten()
        preds = np.clip(preds, 0, 1)
        
        # For each prediction, find its CDF value and map it
        aligned = np.zeros_like(preds)
        
        # Calculate empirical CDF for input predictions
        hist, bins = np.histogram(preds, bins=self.n_bins, range=(0, 1), density=True)
        cdf = np.cumsum(hist) / np.sum(hist)
        cdf = np.maximum.accumulate(cdf)  # Ensure monotonic
        
        # Map using interpolation
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        for i, pred in enumerate(preds):
            # Find which bin this prediction falls into
            bin_idx = np.digitize(pred, bins) - 1
            bin_idx = np.clip(bin_idx, 0, len(cdf) - 1)
            
            # Get CDF value at this bin
            cdf_value = cdf[bin_idx]
            
            # Map using the learned function
            if cdf_value <= self.target_cdf[0]:
                aligned[i] = self.reference_bins[0]
            elif cdf_value >= self.target_cdf[-1]:
                aligned[i] = self.reference_bins[-2]
            else:
                aligned[i] = float(self.mapping_function(cdf_value))
        
        # Clip to valid range
        aligned = np.clip(aligned, 0, 1)
        
        # Calculate statistics after alignment
        self.wasserstein_after = wasserstein_distance(
            self.reference_bins[:-1], 
            np.interp(np.linspace(0, 1, len(aligned)), 
                     np.linspace(0, 1, len(bin_centers)), 
                     bin_centers)
        )
        
        return aligned
    
    def fit_transform(self, reference_predictions, target_predictions):
        """
        Fit and transform in one step
        """
        self.fit(reference_predictions, target_predictions)
        return self.transform(target_predictions)
    
    def get_alignment_statistics(self):
        """
        Get statistics about the alignment
        """
        stats = {
            'wasserstein_distance_before': self.wasserstein_before,
            'wasserstein_distance_after': getattr(self, 'wasserstein_after', None),
            'ks_statistic': self.ks_statistic,
            'ks_pvalue': self.ks_pvalue,
            'n_bins': self.n_bins,
            'method': self.method
        }
        
        if self.wasserstein_after is not None:
            stats['wasserstein_reduction_pct'] = (
                (self.wasserstein_before - self.wasserstein_after) / 
                self.wasserstein_before * 100
            )
        
        return stats

# ==================== 4. SIMULATE ACQUISITION SHIFT ====================

def simulate_acquisition_shift(X, y, shift_type='intensity', severity=0.3, random_state=42):
    """
    Simulate acquisition shift in medical imaging data
    
    Parameters:
    -----------
    X : numpy array
        Original features
    y : numpy array
        Labels
    shift_type : str
        Type of shift to simulate:
        - 'intensity': Change in image intensity/contrast
        - 'noise': Increased noise level
        - 'resolution': Change in resolution (affects texture features)
        - 'mixed': Combination of different shifts
    severity : float
        Severity of shift (0.0 to 1.0)
    random_state : int
        Random seed for reproducibility
    """
    np.random.seed(random_state)
    
    X_shifted = X.copy()
    n_samples, n_features = X.shape
    
    # Define which features are affected based on shift type
    if shift_type == 'intensity':
        # Intensity shift affects all features but especially mean values
        # Simulate different scanner calibration
        shift_vector = np.random.randn(n_features) * severity * 0.5 + severity * 0.3
        X_shifted = X_shifted * (1 + shift_vector)
        
    elif shift_type == 'noise':
        # Increased noise level
        noise_level = severity * 0.2
        X_shifted = X_shifted + np.random.randn(n_samples, n_features) * noise_level
        
    elif shift_type == 'resolution':
        # Resolution change affects texture and smoothness features
        # For simplicity, we'll apply to all features
        # In real scenario, would affect specific features
        resolution_factor = 1.0 + severity * 0.4
        X_shifted = X_shifted * resolution_factor
        
    elif shift_type == 'mixed':
        # Combined shift: intensity + noise
        # Intensity component
        intensity_shift = np.random.randn(n_features) * severity * 0.3 + severity * 0.2
        X_shifted = X_shifted * (1 + intensity_shift)
        
        # Noise component
        noise_level = severity * 0.15
        X_shifted = X_shifted + np.random.randn(n_samples, n_features) * noise_level
        
        # Different effect for malignant vs benign cases
        malignant_mask = y == 1
        if np.any(malignant_mask):
            # Malignant cases might show different shift characteristics
            X_shifted[malignant_mask] = X_shifted[malignant_mask] * (1 + severity * 0.1)
    
    # Add small random component to all shifts
    X_shifted = X_shifted + np.random.randn(*X_shifted.shape) * severity * 0.05
    
    shift_info = {
        'type': shift_type,
        'severity': severity,
        'samples_shifted': n_samples,
        'features_shifted': n_features
    }
    
    return X_shifted, shift_info

# ==================== 5. MODEL TRAINING AND EVALUATION ====================

def train_breast_cancer_model(X_train, y_train, model_type='rf'):
    """
    Train a model for breast cancer classification
    
    Parameters:
    -----------
    X_train : numpy array
        Training features
    y_train : numpy array
        Training labels
    model_type : str
        'rf' for Random Forest, 'lr' for Logistic Regression
        
    Returns:
    --------
    Trained model
    """
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    elif model_type == 'lr':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            C=1.0,
            solver='liblinear'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, threshold=0.5, set_name="Test"):
    """
    Evaluate model performance comprehensively
    
    Returns:
    --------
    Dictionary with all performance metrics
    """
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred_proba = model.predict(X)
    
    # Apply threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate all metrics
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'confusion_matrix': cm,
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'threshold': threshold
    }
    
    # Calculate derived metrics
    metrics['youden_index'] = metrics['sensitivity'] + metrics['specificity'] - 1
    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
    metrics['sen_spc_diff'] = metrics['sensitivity'] - metrics['specificity']
    
    # Print results
    print(f"\n{set_name} Set Performance:")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Youden's Index: {metrics['youden_index']:.4f}")
    print(f"  SEN-SPC Difference: {metrics['sen_spc_diff']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {tn:3d}, FP: {fp:3d}")
    print(f"    FN: {fn:3d}, TP: {tp:3d}")
    
    return metrics

def find_balanced_threshold(model, X_val, y_val):
    """
    Find threshold where sensitivity equals specificity
    """
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_pred_proba = model.predict(X_val)
    
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    
    # Find threshold where sensitivity equals specificity
    # i.e., where tpr = 1 - fpr
    diff = np.abs(tpr - (1 - fpr))
    optimal_idx = np.argmin(diff)
    optimal_threshold = thresholds[optimal_idx]
    
    # Ensure reasonable bounds
    optimal_threshold = np.clip(optimal_threshold, 0.1, 0.9)
    
    return optimal_threshold

# ==================== 6. MAIN EXPERIMENT ====================

def run_upa_experiment(shift_severity=0.4):
    """
    Run the complete UPA experiment
    """
    print("\n" + "=" * 80)
    print("UPA EXPERIMENT: PERFORMANCE DRIFT CORRECTION")
    print("=" * 80)
    
    # Step 1: Load and analyze data
    df = load_and_analyze_dataset()
    if df is None:
        return
    
    # Step 2: Preprocess data
    X, y, feature_names, label_encoder = preprocess_data(df)
    
    # Step 3: Split data into domains
    print("\n" + "-" * 80)
    print("DATA SPLITTING")
    print("-" * 80)
    
    # Split into source (training/validation) and target (test)
    # We'll use 70% for source, 30% for target
    X_source, X_target, y_source, y_target = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Split source into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_source, y_source, test_size=0.2, random_state=42, stratify=y_source
    )
    
    print(f"Training set (source): {X_train.shape[0]} samples")
    print(f"Validation set (source): {X_val.shape[0]} samples")
    print(f"Target set (before shift): {X_target.shape[0]} samples")
    
    # Step 4: Apply acquisition shift to target domain
    print("\n" + "-" * 80)
    print("APPLYING ACQUISITION SHIFT")
    print("-" * 80)
    
    X_target_shifted, shift_info = simulate_acquisition_shift(
        X_target, y_target, 
        shift_type='mixed', 
        severity=shift_severity
    )
    
    print(f"Shift type: {shift_info['type']}")
    print(f"Shift severity: {shift_info['severity']}")
    print(f"Samples shifted: {shift_info['samples_shifted']}")
    print(f"Features affected: {shift_info['features_shifted']}")
    
    # Step 5: Scale features
    print("\n" + "-" * 80)
    print("FEATURE SCALING")
    print("-" * 80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_target_scaled = scaler.transform(X_target_shifted)
    
    print("Features scaled using StandardScaler")
    
    # Step 6: Train model on source domain
    print("\n" + "-" * 80)
    print("TRAINING MODEL ON SOURCE DOMAIN")
    print("-" * 80)
    
    model = train_breast_cancer_model(X_train_scaled, y_train, model_type='rf')
    
    # Evaluate on source validation
    print("\nEvaluating on source validation set...")
    source_metrics = evaluate_model(model, X_val_scaled, y_val, set_name="Source Validation")
    
    # Find optimal threshold (where sensitivity = specificity)
    optimal_threshold = find_balanced_threshold(model, X_val_scaled, y_val)
    print(f"\nOptimal threshold (SEN = SPC): {optimal_threshold:.4f}")
    
    # Re-evaluate with optimal threshold
    source_metrics_optimal = evaluate_model(
        model, X_val_scaled, y_val, 
        threshold=optimal_threshold,
        set_name="Source Validation (Optimal Threshold)"
    )
    
    # Step 7: Evaluate on shifted target domain (BEFORE UPA)
    print("\n" + "-" * 80)
    print("EVALUATION ON SHIFTED TARGET DOMAIN (BEFORE UPA)")
    print("-" * 80)
    
    target_metrics_before = evaluate_model(
        model, X_target_scaled, y_target,
        threshold=optimal_threshold,
        set_name="Target Domain (Before UPA)"
    )
    
    # Calculate performance drift
    drift_sensitivity = target_metrics_before['sensitivity'] - source_metrics_optimal['sensitivity']
    drift_specificity = target_metrics_before['specificity'] - source_metrics_optimal['specificity']
    drift_balance = target_metrics_before['sen_spc_diff'] - source_metrics_optimal['sen_spc_diff']
    
    print(f"\nPerformance Drift (Target - Source):")
    print(f"  Δ Sensitivity: {drift_sensitivity:+.4f}")
    print(f"  Δ Specificity: {drift_specificity:+.4f}")
    print(f"  Δ SEN-SPC Balance: {drift_balance:+.4f}")
    
    # Step 8: Apply UPA for correction
    print("\n" + "-" * 80)
    print("APPLYING UNSUPERVISED PREDICTION ALIGNMENT (UPA)")
    print("-" * 80)
    
    # We need some data from target domain for alignment (without labels)
    # Let's split the target data into alignment and test sets
    X_target_align, X_target_test, y_target_align, y_target_test = train_test_split(
        X_target_scaled, y_target, test_size=0.7, random_state=42, stratify=y_target
    )
    
    print(f"Alignment set: {X_target_align.shape[0]} samples (unlabeled)")
    print(f"Test set: {X_target_test.shape[0]} samples")
    
    # Get predictions for alignment
    source_val_predictions = model.predict_proba(X_val_scaled)[:, 1]
    target_align_predictions = model.predict_proba(X_target_align)[:, 1]
    target_test_predictions = model.predict_proba(X_target_test)[:, 1]
    
    # Apply UPA
    print("\nTraining UPA model...")
    upa = UnsupervisedPredictionAlignment(n_bins=50, method='linear')
    upa.fit(source_val_predictions, target_align_predictions)
    
    # Get alignment statistics
    alignment_stats = upa.get_alignment_statistics()
    print(f"\nAlignment Statistics:")
    print(f"  Wasserstein distance before: {alignment_stats['wasserstein_distance_before']:.6f}")
    if 'wasserstein_reduction_pct' in alignment_stats:
        print(f"  Wasserstein reduction: {alignment_stats['wasserstein_reduction_pct']:.1f}%")
    print(f"  KS test statistic: {alignment_stats['ks_statistic']:.4f}")
    print(f"  KS test p-value: {alignment_stats['ks_pvalue']:.4f}")
    
    # Align test predictions
    print("\nAligning test predictions...")
    target_test_aligned = upa.transform(target_test_predictions)
    
    # Step 9: Evaluate on target domain (AFTER UPA)
    print("\n" + "-" * 80)
    print("EVALUATION ON TARGET DOMAIN (AFTER UPA)")
    print("-" * 80)
    
    # Create new metrics with aligned predictions
    y_pred_aligned = (target_test_aligned >= optimal_threshold).astype(int)
    
    cm_aligned = confusion_matrix(y_target_test, y_pred_aligned)
    tn, fp, fn, tp = cm_aligned.ravel()
    
    target_metrics_after = {
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'accuracy': accuracy_score(y_target_test, y_pred_aligned),
        'precision': precision_score(y_target_test, y_pred_aligned, zero_division=0),
        'recall': recall_score(y_target_test, y_pred_aligned, zero_division=0),
        'f1': f1_score(y_target_test, y_pred_aligned, zero_division=0),
        'roc_auc': roc_auc_score(y_target_test, target_test_aligned),
        'confusion_matrix': cm_aligned,
        'y_true': y_target_test,
        'y_pred': y_pred_aligned,
        'y_pred_proba': target_test_aligned,
        'threshold': optimal_threshold
    }
    
    target_metrics_after['youden_index'] = (
        target_metrics_after['sensitivity'] + target_metrics_after['specificity'] - 1
    )
    target_metrics_after['balanced_accuracy'] = (
        target_metrics_after['sensitivity'] + target_metrics_after['specificity']
    ) / 2
    target_metrics_after['sen_spc_diff'] = (
        target_metrics_after['sensitivity'] - target_metrics_after['specificity']
    )
    
    print("\nPerformance after UPA alignment:")
    print(f"  Sensitivity: {target_metrics_after['sensitivity']:.4f}")
    print(f"  Specificity: {target_metrics_after['specificity']:.4f}")
    print(f"  Accuracy: {target_metrics_after['accuracy']:.4f}")
    print(f"  F1-Score: {target_metrics_after['f1']:.4f}")
    print(f"  ROC-AUC: {target_metrics_after['roc_auc']:.4f}")
    print(f"  Youden's Index: {target_metrics_after['youden_index']:.4f}")
    print(f"  SEN-SPC Difference: {target_metrics_after['sen_spc_diff']:.4f}")
    
    # Step 10: Calculate improvements
    print("\n" + "-" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("-" * 80)
    
    improvements = {
        'sensitivity': target_metrics_after['sensitivity'] - target_metrics_before['sensitivity'],
        'specificity': target_metrics_after['specificity'] - target_metrics_before['specificity'],
        'accuracy': target_metrics_after['accuracy'] - target_metrics_before['accuracy'],
        'f1': target_metrics_after['f1'] - target_metrics_before['f1'],
        'youden_index': target_metrics_after['youden_index'] - target_metrics_before['youden_index'],
        'sen_spc_balance': (
            abs(target_metrics_after['sen_spc_diff']) - 
            abs(target_metrics_before['sen_spc_diff'])
        )
    }
    
    print(f"Sensitivity improvement: {improvements['sensitivity']:+.4f}")
    print(f"Specificity improvement: {improvements['specificity']:+.4f}")
    print(f"Accuracy improvement: {improvements['accuracy']:+.4f}")
    print(f"F1-Score improvement: {improvements['f1']:+.4f}")
    print(f"Youden's Index improvement: {improvements['youden_index']:+.4f}")
    print(f"SEN-SPC balance improvement: {improvements['sen_spc_balance']:+.4f} (negative is better)")
    
    # Calculate recovery percentages
    print(f"\nRecovery from drift:")
    sens_recovery = improvements['sensitivity'] / abs(drift_sensitivity) * 100 if drift_sensitivity != 0 else 0
    spec_recovery = improvements['specificity'] / abs(drift_specificity) * 100 if drift_specificity != 0 else 0
    balance_recovery = improvements['sen_spc_balance'] / abs(drift_balance) * 100 if drift_balance != 0 else 0
    
    print(f"  Sensitivity drift recovery: {sens_recovery:.1f}%")
    print(f"  Specificity drift recovery: {spec_recovery:.1f}%")
    print(f"  Balance drift recovery: {balance_recovery:.1f}%")
    
    # Return all results for visualization
    results = {
        'source_metrics': source_metrics_optimal,
        'target_before': target_metrics_before,
        'target_after': target_metrics_after,
        'improvements': improvements,
        'alignment_stats': alignment_stats,
        'optimal_threshold': optimal_threshold,
        'shift_severity': shift_severity,
        'model': model,
        'upa': upa,
        'data': {
            'y_target_test': y_target_test,
            'predictions_before': target_test_predictions,
            'predictions_after': target_test_aligned,
            'feature_names': feature_names,
            'label_encoder': label_encoder
        }
    }
    
    return results

# ==================== 7. VISUALIZATION ====================

def create_visualizations(results, save_prefix='upa_results'):
    """
    Create comprehensive visualizations of results
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Extract data
    source = results['source_metrics']
    target_before = results['target_before']
    target_after = results['target_after']
    improvements = results['improvements']
    threshold = results['optimal_threshold']
    shift_severity = results['shift_severity']
    
    y_test = results['data']['y_target_test']
    preds_before = results['data']['predictions_before']
    preds_after = results['data']['predictions_after']
    
    # Create a figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f'Unsupervised Prediction Alignment (UPA) - Breast Cancer Classification\n'
        f'Acquisition Shift Severity: {shift_severity}',
        fontsize=16,
        fontweight='bold'
    )
    
    # 1. Prediction Distributions Before UPA
    ax1 = axes[0, 0]
    benign_mask = y_test == 0
    malignant_mask = y_test == 1
    
    ax1.hist(preds_before[benign_mask], bins=30, alpha=0.6, 
             label='Benign', density=True, color='blue')
    ax1.hist(preds_before[malignant_mask], bins=30, alpha=0.6, 
             label='Malignant', density=True, color='red')
    ax1.axvline(x=threshold, color='black', linestyle='--', 
                linewidth=2, label=f'Threshold ({threshold:.2f})')
    ax1.set_xlabel('Prediction Probability', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title('Predictions Before UPA\n(Target Domain with Shift)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction Distributions After UPA
    ax2 = axes[0, 1]
    ax2.hist(preds_after[benign_mask], bins=30, alpha=0.6, 
             label='Benign', density=True, color='green')
    ax2.hist(preds_after[malignant_mask], bins=30, alpha=0.6, 
             label='Malignant', density=True, color='orange')
    ax2.axvline(x=threshold, color='black', linestyle='--', 
                linewidth=2, label=f'Threshold ({threshold:.2f})')
    ax2.set_xlabel('Prediction Probability', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title('Predictions After UPA\n(Aligned to Source Domain)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Sensitivity and Specificity Comparison
    ax3 = axes[0, 2]
    metrics = ['Sensitivity', 'Specificity']
    source_vals = [source['sensitivity'], source['specificity']]
    before_vals = [target_before['sensitivity'], target_before['specificity']]
    after_vals = [target_after['sensitivity'], target_after['specificity']]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    ax3.bar(x - width, source_vals, width, label='Source Domain', color='gray', alpha=0.7)
    ax3.bar(x, before_vals, width, label='Target (Before UPA)', color='red', alpha=0.7)
    ax3.bar(x + width, after_vals, width, label='Target (After UPA)', color='green', alpha=0.7)
    
    ax3.set_xlabel('Performance Metric', fontsize=10)
    ax3.set_ylabel('Value', fontsize=10)
    ax3.set_title('Sensitivity and Specificity Comparison', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (src, before, after) in enumerate(zip(source_vals, before_vals, after_vals)):
        ax3.text(i - width, src + 0.01, f'{src:.3f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i, before + 0.01, f'{before:.3f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i + width, after + 0.01, f'{after:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. ROC Curves
    ax4 = axes[1, 0]
    
    # Calculate ROC curves
    fpr_source, tpr_source, _ = roc_curve(source['y_true'], source['y_pred_proba'])
    fpr_before, tpr_before, _ = roc_curve(y_test, preds_before)
    fpr_after, tpr_after, _ = roc_curve(y_test, preds_after)
    
    ax4.plot(fpr_source, tpr_source, label=f'Source (AUC={source["roc_auc"]:.3f})', 
             linewidth=2, color='gray')
    ax4.plot(fpr_before, tpr_before, label=f'Target Before (AUC={target_before["roc_auc"]:.3f})', 
             linewidth=2, color='red')
    ax4.plot(fpr_after, tpr_after, label=f'Target After (AUC={target_after["roc_auc"]:.3f})', 
             linewidth=2, color='green')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    ax4.set_xlabel('False Positive Rate', fontsize=10)
    ax4.set_ylabel('True Positive Rate', fontsize=10)
    ax4.set_title('ROC Curves', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. SEN-SPC Balance
    ax5 = axes[1, 1]
    
    balance_values = [
        source['sen_spc_diff'],
        target_before['sen_spc_diff'],
        target_after['sen_spc_diff']
    ]
    labels = ['Source', 'Target\n(Before)', 'Target\n(After)']
    colors = ['gray', 'red', 'green']
    
    bars = ax5.bar(labels, balance_values, color=colors, alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax5.set_ylabel('Sensitivity - Specificity', fontsize=10)
    ax5.set_title('Sensitivity-Specificity Balance\n(Closer to 0 is Better)', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, balance_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # 6. Improvement Summary
    ax6 = axes[1, 2]
    
    improvement_metrics = ['Sensitivity', 'Specificity', 'F1-Score', 'Youden Index']
    improvement_values = [
        improvements['sensitivity'],
        improvements['specificity'],
        improvements['f1'],
        improvements['youden_index']
    ]
    
    colors_improve = ['green' if val >= 0 else 'red' for val in improvement_values]
    
    bars = ax6.bar(improvement_metrics, improvement_values, color=colors_improve, alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax6.set_ylabel('Improvement (After - Before)', fontsize=10)
    ax6.set_title('Performance Improvements from UPA\n(Positive is Better)', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, improvement_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                f'{val:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = f'{save_prefix}_main.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Main visualization saved to: {output_file}")
    
    # Create detailed metrics comparison
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    metrics_names = [
        'Sensitivity', 'Specificity', 'Accuracy', 
        'Precision', 'F1-Score', 'ROC-AUC', 'Youden Index'
    ]
    
    source_values = [
        source['sensitivity'], source['specificity'], source['accuracy'],
        source['precision'], source['f1'], source['roc_auc'], source['youden_index']
    ]
    
    target_before_values = [
        target_before['sensitivity'], target_before['specificity'], target_before['accuracy'],
        target_before['precision'], target_before['f1'], target_before['roc_auc'], target_before['youden_index']
    ]
    
    target_after_values = [
        target_after['sensitivity'], target_after['specificity'], target_after['accuracy'],
        target_after['precision'], target_after['f1'], target_after['roc_auc'], target_after['youden_index']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    ax.bar(x - width, source_values, width, label='Source Domain', color='gray', alpha=0.7)
    ax.bar(x, target_before_values, width, label='Target (Before UPA)', color='red', alpha=0.7)
    ax.bar(x + width, target_after_values, width, label='Target (After UPA)', color='green', alpha=0.7)
    
    ax.set_xlabel('Performance Metrics', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Comprehensive Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file2 = f'{save_prefix}_detailed.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed metrics saved to: {output_file2}")
    
    plt.show()
    
    return fig, fig2

# ==================== 8. RUN MULTIPLE EXPERIMENTS ====================

def run_multiple_scenarios():
    """
    Run UPA experiment with different shift severities
    """
    print("\n" + "=" * 80)
    print("MULTIPLE SCENARIO ANALYSIS")
    print("=" * 80)
    
    shift_levels = [0.2, 0.4, 0.6, 0.8]
    all_results = []
    
    for severity in shift_levels:
        print(f"\n>>> Running experiment with shift severity: {severity}")
        print("-" * 60)
        
        results = run_upa_experiment(shift_severity=severity)
        if results:
            all_results.append((severity, results))
            
            # Create visualizations for this scenario
            create_visualizations(results, save_prefix=f'upa_shift_{severity}')
    
    if not all_results:
        print("No experiments completed successfully.")
        return
    
    # Create comparison plot across shift levels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('UPA Effectiveness Across Different Shift Severities', 
                 fontsize=16, fontweight='bold')
    
    severities = [s for s, _ in all_results]
    
    # Extract data
    sens_improvements = []
    spec_improvements = []
    balance_improvements = []
    youden_improvements = []
    
    for severity, results in all_results:
        sens_improvements.append(results['improvements']['sensitivity'])
        spec_improvements.append(results['improvements']['specificity'])
        balance_improvements.append(results['improvements']['sen_spc_balance'])
        youden_improvements.append(results['improvements']['youden_index'])
    
    # Plot 1: Sensitivity improvements
    axes[0, 0].plot(severities, sens_improvements, 'o-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Shift Severity')
    axes[0, 0].set_ylabel('Sensitivity Improvement')
    axes[0, 0].set_title('Sensitivity Recovery')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Specificity improvements
    axes[0, 1].plot(severities, spec_improvements, 's-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Shift Severity')
    axes[0, 1].set_ylabel('Specificity Improvement')
    axes[0, 1].set_title('Specificity Recovery')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Balance improvements
    axes[1, 0].plot(severities, balance_improvements, '^-', linewidth=2, markersize=8, color='green')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Shift Severity')
    axes[1, 0].set_ylabel('Balance Improvement')
    axes[1, 0].set_title('SEN-SPC Balance Recovery\n(Negative is Better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Youden's Index improvements
    axes[1, 1].plot(severities, youden_improvements, 'd-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Shift Severity')
    axes[1, 1].set_ylabel('Youden Index Improvement')
    axes[1, 1].set_title('Overall Performance (Youden Index)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('upa_multiple_scenarios.png', dpi=300, bbox_inches='tight')
    print("\n✓ Multiple scenario comparison saved to: upa_multiple_scenarios.png")
    
    plt.show()

# ==================== 9. MAIN EXECUTION ====================

def main():
    """
    Main function to run the UPA project
    """
    print("=" * 80)
    print("UPA PROJECT: Automatic Correction of Performance Drift")
    print("Based on: Nature Communications (2023) 14:6608")
    print("=" * 80)
    
    # Check for required packages
    # required_packages = ['pandas', 'numpy', 'scikit-learn', 'scipy', 'matplotlib', 'seaborn']
    # missing_packages = []
    #
    # for package in required_packages:
    #     try:
    #         __import__(package)
    #     except ImportError:
    #         missing_packages.append(package)
    #
    # if missing_packages:
    #     print(f"\nMissing packages: {missing_packages}")
    #     print("Please install them using: pip install " + " ".join(missing_packages))
    #     return
    
    print("\n✓ All required packages are installed")
    
    # Run single experiment
    print("\nRunning single experiment with moderate shift...")
    results = run_upa_experiment(shift_severity=0.4)
    
    if results:
        # Create visualizations
        create_visualizations(results)
        
        # Print final summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        
        print(f"\nDataset: Breast Cancer Wisconsin (Diagnostic)")
        print(f"Samples: 569, Features: 30")
        print(f"Shift Severity: {results['shift_severity']}")
        print(f"Optimal Threshold: {results['optimal_threshold']:.4f}")
        
        print("\n" + "-" * 80)
        print(f"{'Metric':<20} {'Source':<10} {'Target (Before)':<15} {'Target (After)':<15} {'Improvement':<12}")
        print("-" * 80)
        
        metrics_to_show = [
            ('Sensitivity', results['source_metrics']['sensitivity'], 
             results['target_before']['sensitivity'], results['target_after']['sensitivity']),
            ('Specificity', results['source_metrics']['specificity'], 
             results['target_before']['specificity'], results['target_after']['specificity']),
            ('Accuracy', results['source_metrics']['accuracy'], 
             results['target_before']['accuracy'], results['target_after']['accuracy']),
            ('F1-Score', results['source_metrics']['f1'], 
             results['target_before']['f1'], results['target_after']['f1']),
            ('ROC-AUC', results['source_metrics']['roc_auc'], 
             results['target_before']['roc_auc'], results['target_after']['roc_auc']),
        ]
        
        for name, src, before, after in metrics_to_show:
            improvement = after - before
            print(f"{name:<20} {src:<10.4f} {before:<15.4f} {after:<15.4f} {improvement:+.4f}")
        
        print("-" * 80)
        
        print("\n✓ Experiment completed successfully!")
        print("✓ Check the generated PNG files for visualizations")
        
        # Ask if user wants to run multiple scenarios
        response = input("\nRun multiple scenarios with different shift levels? (y/n): ")
        if response.lower() == 'y':
            run_multiple_scenarios()
    
    else:
        print("\n✗ Experiment failed. Please check the error messages above.")

# ==================== 10. RUN THE PROJECT ====================

if __name__ == "__main__":
    main()