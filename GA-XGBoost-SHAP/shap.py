import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib
import shap
import warnings
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')


# =========================== Configuration ===========================
class Config:
    """Configuration settings."""

    # Model and data
    MODEL_PATH = 'model/GA-XGBoost/best_xgboost_model.pkl'
    PARAMS_PATH = 'model/GA-XGBoost/best_params.json'
    DATA_DIR = 'model/GA-XGBoost'
    ORIGINAL_DATA_PATH = 'dataset2.csv'

    # Data file names
    X_TRAIN_FILE = 'X_train.npy'
    Y_TRAIN_FILE = 'y_train.npy'
    X_TEST_FILE = 'X_test.npy'
    Y_TEST_FILE = 'y_test.npy'

    # Output
    OUTPUT_DIR = 'model/GA-xgboost-shap'
    FIGURE_FORMAT = 'png'
    FIGURE_DPI = 300

    # SHAP
    SHAP_SAMPLE_SIZE = None  # all test
    BACKGROUND_SAMPLE_SIZE = None  # all train
    TOP_N_FEATURES = 12
    DEPENDENCE_PLOT_FEATURES = 5

    # Colors
    COLOR_SCHEME = {
        'scatter': 'steelblue',
        'line': 'crimson',
        'hist': 'forestgreen',
        'bar': 'darkorange',
        'residual': 'mediumpurple'
    }

    # Font settings
    FONT_SETTINGS = {
        'family': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'size': 12
    }

    # Target variable
    TARGET_COLUMN = None


# ===============================================================


class XGBoostVisualizer:
    """XGBoost model visualizer."""

    def __init__(self, config=None):
        self.config = config if config else Config
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.shap_values = None
        self.explainer = None
        self.predictions = None
        self.X_shap_background = None
        self.X_shap_explain = None

        # matplotlib
        self._setup_matplotlib_fonts()

        # Output
        self.output_path = Path(self.config.OUTPUT_DIR)
        self.output_path.mkdir(exist_ok=True)

    def _setup_matplotlib_fonts(self):
        """Set matplotlib fonts."""
        plt.rcParams['font.sans-serif'] = self.config.FONT_SETTINGS['family']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = self.config.FONT_SETTINGS['size']

    def load_model_and_data(self):
        """Load model and data."""
        print("=" * 60)
        print("Loading model and data...")
        print("=" * 60)

        # Model
        model_path = Path(self.config.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded: {type(self.model)}")

        # CSV feature names
        self._load_feature_names_from_csv()

        # Train and test data
        data_dir = Path(self.config.DATA_DIR)

        # Train data
        X_train_path = data_dir / self.config.X_TRAIN_FILE
        if X_train_path.exists():
            self.X_train = np.load(X_train_path)
            print(f"✓ Training data: X_train={self.X_train.shape}")
        else:
            raise FileNotFoundError(f"Training data not found: {X_train_path}")

        # Test data
        X_test_path = data_dir / self.config.X_TEST_FILE
        y_test_path = data_dir / self.config.Y_TEST_FILE

        if X_test_path.exists() and y_test_path.exists():
            self.X_test = np.load(X_test_path)
            self.y_test = np.load(y_test_path)
            print(f"✓ Test data: X_test={self.X_test.shape}, y_test={self.y_test.shape}")

            # Check feature name count
            if self.feature_names and len(self.feature_names) != self.X_test.shape[1]:
                print(f"⚠ Feature name count ({len(self.feature_names)}) does not match data columns ({self.X_test.shape[1]})!")
                print("  Using generic feature names instead.")
                self.feature_names = [f'feature_{i}' for i in range(self.X_test.shape[1])]
        else:
            raise FileNotFoundError(f"Test data not found: {data_dir}")

        # Print feature names
        if self.feature_names:
            print(f"✓ Feature names loaded ({len(self.feature_names)}):")
            for i, name in enumerate(self.feature_names[:5]):
                print(f"  {i}: {name}")
            if len(self.feature_names) > 5:
                print(f"  ... and {len(self.feature_names) - 5} more features")

        return True

    def _load_feature_names_from_csv(self):
        """Load feature names from the original CSV file."""
        csv_path = Path(self.config.ORIGINAL_DATA_PATH)
        if not csv_path.exists():
            print(f"⚠ Original CSV not found: {csv_path}")
            return

        try:
            # Read first row
            df = pd.read_csv(csv_path, encoding='gbk', nrows=1)

            target_column = self.config.TARGET_COLUMN

            if target_column and target_column in df.columns:
                self.feature_names = [col for col in df.columns if col != target_column]
                print(f"✓ Feature names loaded (excluding target column '{target_column}')")
            elif len(df.columns) > 1:
                self.feature_names = df.columns[:-1].tolist()
                print("✓ Feature names loaded (using all columns except the last one)")
            else:
                print("⚠ Could not infer feature names from CSV.")

        except Exception as e:
            print(f"⚠ Failed to read CSV for feature names: {e}")

        if not self.feature_names:
            print("⚠ Feature names are not available. Generic names will be used.")

    def prepare_shap_data(self):
        """Prepare data for SHAP analysis."""
        print("\nPreparing SHAP data...")

        # 1. Background data
        if self.config.BACKGROUND_SAMPLE_SIZE is None:
            self.X_shap_background = self.X_train
            print(f"✓ Using all training data as background ({len(self.X_train)} samples)")
        else:
            sample_size = min(self.config.BACKGROUND_SAMPLE_SIZE, len(self.X_train))
            np.random.seed(42)
            indices = np.random.choice(len(self.X_train), sample_size, replace=False)
            self.X_shap_background = self.X_train[indices]
            print(f"✓ Using sampled training data as background ({len(self.X_shap_background)} samples)")

        # 2. Explain data (test)
        if self.config.SHAP_SAMPLE_SIZE is None:
            self.X_shap_explain = self.X_test
            print(f"✓ Using all test data for explanation ({len(self.X_test)} samples)")
        else:
            sample_size = min(self.config.SHAP_SAMPLE_SIZE, len(self.X_test))
            np.random.seed(42)
            indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            self.X_shap_explain = self.X_test[indices]
            print(f"✓ Using sampled test data for explanation ({len(self.X_shap_explain)} samples)")

        return True

    def calculate_predictions_and_shap(self):
        """Calculate predictions and SHAP values."""
        print("\nCalculating predictions and SHAP values...")

        # 1. Predictions
        self.predictions = self.model.predict(self.X_test)

        # 2. Performance metrics
        self.calculate_performance_metrics()

        # 3. Prepare SHAP data
        self.prepare_shap_data()

        # 4. SHAP values
        print("\nCalculating SHAP values...")

        # XGBoost uses TreeExplainer
        print(f"  Creating SHAP explainer...")

        if len(self.X_shap_background) > 7000:
            background_for_explainer = self.X_shap_background[:7000]
            print(f"  Using first 7000 background samples (original: {len(self.X_shap_background)})")
        else:
            background_for_explainer = self.X_shap_background

        # Create explainer
        self.explainer = shap.TreeExplainer(self.model, background_for_explainer)

        # Calculate SHAP values
        print(f"  Calculating SHAP values for {len(self.X_shap_explain)} samples...")
        self.shap_values = self.explainer(self.X_shap_explain)

        print(f"✓ SHAP values calculated: {self.shap_values.shape}")
        print(f"  - Samples: {self.shap_values.shape[0]}")
        print(f"  - Features: {self.shap_values.shape[1]}")

        return True

    def calculate_performance_metrics(self):
        """Calculate and save performance metrics."""
        self.r2 = r2_score(self.y_test, self.predictions)
        self.rmse = np.sqrt(mean_squared_error(self.y_test, self.predictions))
        self.mae = mean_absolute_error(self.y_test, self.predictions)
        self.residuals = self.y_test - self.predictions

        print(f"Performance metrics:")
        print(f"  R²: {self.r2:.6f}")
        print(f"  RMSE: {self.rmse:.6f}")
        print(f"  MAE: {self.mae:.6f}")

        metrics_df = pd.DataFrame({
            'Metric': ['R² Score', 'RMSE', 'MAE', 'MSE'],
            'Value': [self.r2, self.rmse, self.mae, self.rmse ** 2],
            'Description': ['Higher is better, max 1', 'Lower is better', 'Lower is better', 'Lower is better']
        })

        metrics_path = self.output_path / 'performance_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"✓ Performance metrics saved to {metrics_path}")

        return metrics_df

    def plot_shap_summary(self):
        """Plot SHAP summary plot."""
        print("\nGenerating SHAP summary plot...")

        plt.figure(figsize=(14, 8))

        # Data
        shap_data = self.shap_values.data if hasattr(self.shap_values, 'data') else self.X_shap_explain

        # Feature names
        if self.feature_names and len(self.feature_names) == self.shap_values.shape[1]:
            feature_names = self.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(self.shap_values.shape[1])]

        # SHAP summary plot
        shap.summary_plot(
            self.shap_values,
            shap_data,
            feature_names=feature_names,
            max_display=self.config.TOP_N_FEATURES,
            show=False,
            plot_size=(14, 8)
        )

        plt.title(f'SHAP Summary Plot (Top {self.config.TOP_N_FEATURES})',
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        save_path = self.output_path / f'shap_summary.{self.config.FIGURE_FORMAT}'
        plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        print(f"✓ SHAP summary plot saved to {save_path}")
        return save_path

    def plot_shap_importance(self):
        """Plot SHAP feature importance."""
        print("\nGenerating SHAP feature importance plot...")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Calculate SHAP importance
        shap_importance = np.abs(self.shap_values.values).mean(axis=0)

        # Feature names
        if self.feature_names and len(self.feature_names) == len(shap_importance):
            all_features = self.feature_names
        else:
            all_features = [f'feature_{i}' for i in range(len(shap_importance))]

        # Top N features
        top_n = min(self.config.TOP_N_FEATURES, len(all_features))
        top_indices = np.argsort(shap_importance)[-top_n:][::-1]
        top_features = [all_features[i] for i in top_indices]
        top_importance = shap_importance[top_indices]

        # Bar plot
        bars = ax.barh(range(top_n), top_importance,
                       color=self.config.COLOR_SCHEME['bar'],
                       edgecolor='black',
                       height=0.7)

        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_importance)):
            ax.text(importance * 1.005, bar.get_y() + bar.get_height() / 2,
                    f'{importance:.4f}',
                    va='center',
                    fontsize=10)

        # Labels and title
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features, fontsize=11)
        ax.set_xlabel('Mean |SHAP value| (average impact on model output)', fontsize=12)
        ax.set_title(f'SHAP Feature Importance (Top {top_n} Features)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()

        # Save figure
        save_path = self.output_path / f'shap_feature_importance.{self.config.FIGURE_FORMAT}'
        plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        # Save importance data
        importance_df = pd.DataFrame({
            'Feature': all_features,
            'SHAP Importance': shap_importance
        }).sort_values('SHAP Importance', ascending=False)

        importance_path = self.output_path / 'shap_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')

        print(f"✓ SHAP feature importance plot saved to {save_path}")
        print(f"✓ SHAP feature importance data saved to {importance_path}")

        return save_path, importance_df

    def plot_shap_dependence(self):
        """Plot SHAP dependence plots."""
        print("\nGenerating SHAP dependence plots...")

        # Calculate feature importance
        shap_importance = np.abs(self.shap_values.values).mean(axis=0)

        # Feature names
        if self.feature_names and len(self.feature_names) == len(shap_importance):
            all_features = self.feature_names
        else:
            all_features = [f'feature_{i}' for i in range(len(shap_importance))]

        # Top N features
        top_n = min(self.config.DEPENDENCE_PLOT_FEATURES, len(all_features))
        top_indices = np.argsort(shap_importance)[-top_n:][::-1]

        save_paths = []

        for i, feature_idx in enumerate(top_indices):
            feature_name = all_features[feature_idx]
            print(f"  Plotting dependence for {feature_name} ({i + 1}/{top_n})...")

            fig, ax = plt.subplots(figsize=(10, 7))
            shap_data = self.shap_values.data if hasattr(self.shap_values, 'data') else self.X_shap_explain

            shap.dependence_plot(
                feature_idx,
                self.shap_values.values,
                shap_data,
                feature_names=all_features,
                ax=ax,
                show=False,
                dot_size=16,
                alpha=0.6
            )

            plt.title(f'SHAP Dependence Plot: {feature_name}', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(feature_name, fontsize=12)
            plt.ylabel('SHAP value', fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            # Ensure valid file name
            safe_feature_name = "".join(c for c in feature_name if c.isalnum() or c in (' ', '_')).rstrip()
            save_path = self.output_path / f'shap_dependence_{safe_feature_name}.{self.config.FIGURE_FORMAT}'
            plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
            plt.close()

            save_paths.append(save_path)

        print(f"✓ {len(save_paths)} SHAP dependence plots saved")
        return save_paths

    def plot_performance_scatter(self):
        """Plot performance scatter plots."""
        print("\nGenerating performance scatter plots...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Subplot 1: Actual vs Predicted
        scatter = ax1.scatter(self.y_test, self.predictions,
                              alpha=0.7, s=40,
                              c=self.config.COLOR_SCHEME['scatter'],
                              edgecolor='white', linewidth=0.5)

        # Range
        min_val = min(self.y_test.min(), self.predictions.min())
        max_val = max(self.y_test.max(), self.predictions.max())
        margin = (max_val - min_val) * 0.05
        min_val -= margin
        max_val += margin

        # Perfect prediction line
        ax1.plot([min_val, max_val], [min_val, max_val],
                 color=self.config.COLOR_SCHEME['line'],
                 linestyle='--', linewidth=2,
                 label='Perfect prediction line (y=x)')

        # Regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.y_test, self.predictions)
        regression_line = slope * np.array([min_val, max_val]) + intercept
        ax1.plot([min_val, max_val], regression_line,
                 color='darkgreen', linestyle='-', linewidth=2,
                 label=f'Regression line (R²={self.r2:.3f})')

        ax1.set_xlabel('Actual values', fontsize=12)
        ax1.set_ylabel('Predicted values', fontsize=12)
        ax1.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_xlim(min_val, max_val)
        ax1.set_ylim(min_val, max_val)

        # Subplot 2: Residual plot
        ax2.scatter(self.predictions, self.residuals,
                    alpha=0.7, s=40,
                    c=self.config.COLOR_SCHEME['residual'],
                    edgecolor='white', linewidth=0.5)

        ax2.axhline(y=0, color=self.config.COLOR_SCHEME['line'],
                    linestyle='--', linewidth=2)

        ax2.set_xlabel('Predicted values', fontsize=12)
        ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
        ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3, linestyle='--')

        # Performance metrics text box
        textstr = '\n'.join((
            f'R² Score: {self.r2:.4f}',
            f'RMSE: {self.rmse:.4f}',
            f'MAE: {self.mae:.4f}',
            f'Samples: {len(self.y_test)}'
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)

        plt.tight_layout()

        save_path = self.output_path / f'performance_scatter.{self.config.FIGURE_FORMAT}'
        plt.savefig(save_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        print(f"✓ Performance scatter plot saved to {save_path}")
        return save_path

    def run_analysis(self):
        """Run the complete analysis."""
        print("=" * 60)
        print("XGBoost Model Visualization Analysis")
        print("=" * 60)

        try:
            # 1. Load model and data
            self.load_model_and_data()

            # 2. Calculate predictions and SHAP values
            self.calculate_predictions_and_shap()

            # 3. SHAP visualizations
            print("\n" + "-" * 50)
            print("Generating SHAP visualizations")
            print("-" * 50)

            self.plot_shap_summary()
            _, importance_df = self.plot_shap_importance()
            self.plot_shap_dependence()

            # 4. Performance visualizations
            print("\n" + "-" * 50)
            print("Generating performance visualizations")
            print("-" * 50)

            self.plot_performance_scatter()

            # 5. Report
            print("\n" + "=" * 60)
            print("Analysis completed!")
            print("=" * 60)
            print(f"All results saved to: {self.output_path}")
            print(f"\nGenerated files:")
            print(f"  1. SHAP summary plot: shap_summary.png")
            print(f"  2. SHAP feature importance: shap_feature_importance.png")
            print(f"  3. SHAP dependence plots: shap_dependence_*.png (total {self.config.DEPENDENCE_PLOT_FEATURES})")
            print(f"  4. Performance scatter plot: performance_scatter.png")
            print(f"  5. Performance metrics: performance_metrics.csv")
            print(f"  6. Feature importance data: shap_feature_importance.csv")

            # Show top features
            print(f"\nTop {min(5, self.config.TOP_N_FEATURES)} most important features:")
            top_features = importance_df.head(min(5, self.config.TOP_N_FEATURES))
            for idx, row in top_features.iterrows():
                print(f"  {row['Feature']}: {row['SHAP Importance']:.6f}")

            return True

        except Exception as e:
            print(f"\nError during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function."""
    # Create visualizer
    visualizer = XGBoostVisualizer()

    # ====================================================
    # Custom configuration (modify as needed)
    # ====================================================

    # 1. If you know the target column name, specify it
    # visualizer.config.TARGET_COLUMN = "mean_average_value"

    # 2. SHAP calculation settings
    # visualizer.config.SHAP_SAMPLE_SIZE = 1000  # Number of test samples for SHAP explanation
    # visualizer.config.BACKGROUND_SAMPLE_SIZE = 2000  # Number of training samples for background

    # 3. Display settings
    # visualizer.config.TOP_N_FEATURES = 15

    # Run analysis
    success = visualizer.run_analysis()

    if success:
        print("\nCustom configuration notes:")
        print("1. SHAP_SAMPLE_SIZE: controls the number of test samples used for SHAP values")
        print("2. BACKGROUND_SAMPLE_SIZE: controls the number of training samples used as background")
        print("3. When both are None, all data is used")

    return success


if __name__ == "__main__":
    main()