"""
Reproducibility: seed management, experiment tracking and result export.
"""

import os
import json
import hashlib
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeedManager:
    """
    Manages the set of random seeds used across runs.
    """
    
    def __init__(self, base_seed: int = 42, n_seeds: int = 5):
        """
        Args:
            base_seed: starting seed
            n_seeds: how many seeds to derive (5 or more)
        """
        self.base_seed = base_seed
        self.n_seeds = max(n_seeds, 5)
        self.seed_list = self._generate_seeds()
        
    def _generate_seeds(self) -> List[int]:
        """Derive the list of seeds."""
        np.random.seed(self.base_seed)
        seeds = [self.base_seed] + [int(s) for s in np.random.randint(0, 10000, size=self.n_seeds - 1)]
        logger.info(f"Generated {self.n_seeds} seeds: {seeds}")
        return seeds
    
    def set_seed(self, seed: int):
        """
        Seed every random number generator in use.
        
        Args:
            seed: the seed to apply
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # PyTorch
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
        # TensorFlow
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        logger.info(f"Random seed set: {seed}")
    
    def get_seeds(self) -> List[int]:
        """Return the seed list."""
        return self.seed_list


class DataVersionManager:
    """
    Records a content hash so a dataset version can be identified later.
    """
    
    @staticmethod
    def compute_data_hash(df: pd.DataFrame) -> str:
        """
        Hash the contents of a dataframe.
        
        Args:
            df: the dataframe
            
        Returns:
            the SHA256 digest
        """
        # Serialise the frame, then hash it
        data_str = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        hash_value = hashlib.sha256(data_str).hexdigest()
        logger.info(f"Data hash: {hash_value[:16]}...")
        return hash_value
    
    @staticmethod
    def save_data_info(
        df: pd.DataFrame,
        save_path: str,
        additional_info: Optional[Dict] = None
    ):
        """
        Write the data version record.
        
        Args:
            df: the dataframe
            save_path: destination path
            additional_info: anything else worth recording
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'hash': DataVersionManager.compute_data_hash(df),
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            info.update(additional_info)
        
        with open(save_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Data record saved: {save_path}")


class ExperimentTracker:
    """
    Tracks a full training and evaluation run.
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "../../experiments"):
        """
        Args:
            experiment_name: name of the experiment
            output_dir: output directory
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'data_info': {},
            'split_indices': {},
            'preprocessing': {},
            'model': {},
            'metrics': {},
            'calibration': {},
            'interpretability': {}
        }
        
        logger.info(f"Experiment tracker ready: {self.experiment_dir}")
    
    def log_config(self, config: Dict):
        """Record the configuration."""
        self.metadata['config'] = config
        logger.info("Configuration recorded")
    
    def log_data_info(self, data_hash: str, data_shape: tuple, feature_list: List[str]):
        """Record the dataset details."""
        self.metadata['data_info'] = {
            'hash': data_hash,
            'shape': data_shape,
            'n_features': len(feature_list),
            'features': feature_list
        }
        logger.info("Dataset details recorded")
    
    def log_split_indices(self, split_indices: Dict[str, List[int]]):
        """Record split membership."""
        self.metadata['split_indices'] = split_indices
        
        # Written to its own file
        split_path = os.path.join(self.experiment_dir, 'split_indices.json')
        with open(split_path, 'w') as f:
            json.dump(split_indices, f, indent=2)
        
        logger.info(f"Split membership recorded: {split_path}")
    
    def log_preprocessing(self, preprocessing_info: Dict):
        """Record the preprocessing settings."""
        self.metadata['preprocessing'] = preprocessing_info
        logger.info("Preprocessing settings recorded")
    
    def log_model(self, model_info: Dict):
        """Record the model details."""
        self.metadata['model'] = model_info
        logger.info("Model details recorded")
    
    def log_metrics(self, split: str, metrics: Dict, before_calibration: bool = True):
        """
        Record evaluation metrics.
        
        Args:
            split: 'train', 'val', 'test'
            metrics: metric name -> value
            before_calibration: True for pre-calibration metrics
        """
        key = f"{split}_{'before' if before_calibration else 'after'}_calibration"
        if split not in self.metadata['metrics']:
            self.metadata['metrics'][split] = {}
        
        self.metadata['metrics'][split][key] = metrics
        logger.info(f"Recorded {split} metrics ({'before' if before_calibration else 'after'} calibration)")
    
    def log_calibration(self, calibration_info: Dict):
        """Record calibration details."""
        self.metadata['calibration'] = calibration_info
        logger.info("Calibration details recorded")
    
    def log_interpretability(self, interpretability_info: Dict):
        """Record interpretability output."""
        self.metadata['interpretability'] = interpretability_info
        logger.info("Interpretability output recorded")
    
    def save_metadata(self):
        """Write the metadata file."""
        self.metadata['end_time'] = datetime.now().isoformat()
        
        metadata_path = os.path.join(self.experiment_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_path}")
    
    def save_artifact(self, obj: Any, filename: str):
        """
        Save an experiment artifact.
        
        Args:
            obj: the object to write
            filename: destination file name
        """
        filepath = os.path.join(self.experiment_dir, filename)
        
        if filename.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        elif filename.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(obj, f, indent=2)
        elif filename.endswith('.csv'):
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(filepath, index=False)
            else:
                raise ValueError("CSV output requires a DataFrame")
        else:
            raise ValueError(f"unsupported file format: {filename}")
        
        logger.info(f"Artifact saved: {filepath}")
    
    def get_experiment_dir(self) -> str:
        """Return the experiment directory."""
        return self.experiment_dir


class ResultsExporter:
    """
    Exports one self-contained result pack per split protocol.
    """
    
    def __init__(self, output_dir: str = "../../results"):
        """
        Args:
            output_dir: output directory
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Result exporter ready: {output_dir}")
    
    def export_protocol_results(
        self,
        protocol_name: str,
        version: str,
        metadata: Dict,
        metrics: Dict,
        predictions: Optional[pd.DataFrame] = None,
        curves_dir: Optional[str] = None,
        interpretability_dir: Optional[str] = None
    ):
        """
        Export the result pack for one protocol.
        
        Args:
            protocol_name: 'patient_level', 'temporal' or 'site'
            version: version string
            metadata: run metadata
            metrics: evaluation metrics
            predictions: per-row predictions
            curves_dir: directory holding the curve plots
            interpretability_dir: directory holding interpretability output
        """
        # One directory per protocol
        protocol_dir = os.path.join(
            self.output_dir,
            f"{protocol_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(protocol_dir, exist_ok=True)
        
        # Metadata
        metadata_path = os.path.join(protocol_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Metrics
        metrics_path = os.path.join(protocol_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Predictions
        if predictions is not None:
            predictions_path = os.path.join(protocol_dir, 'predictions.csv')
            predictions.to_csv(predictions_path, index=False)
        
        # Copy the curve plots
        if curves_dir and os.path.exists(curves_dir):
            import shutil
            curves_target = os.path.join(protocol_dir, 'curves')
            shutil.copytree(curves_dir, curves_target, dirs_exist_ok=True)
        
        # Copy the interpretability output
        if interpretability_dir and os.path.exists(interpretability_dir):
            import shutil
            interp_target = os.path.join(protocol_dir, 'interpretability')
            shutil.copytree(interpretability_dir, interp_target, dirs_exist_ok=True)
        
        logger.info(f"Protocol results exported: {protocol_dir}")
        
        # README
        self._create_readme(protocol_dir, protocol_name, version, metadata, metrics)
        
        return protocol_dir
    
    def _create_readme(
        self,
        protocol_dir: str,
        protocol_name: str,
        version: str,
        metadata: Dict,
        metrics: Dict
    ):
        """Write the result pack README."""
        readme_lines = [
            f"# {protocol_name.upper()} Protocol Results",
            f"",
            f"**Version:** {version}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Data Info",
            f"- Data Hash: {metadata.get('data_info', {}).get('hash', 'N/A')}",
            f"- Data Shape: {metadata.get('data_info', {}).get('shape', 'N/A')}",
            f"- N Features: {metadata.get('data_info', {}).get('n_features', 'N/A')}",
            f"",
            f"## Model Info",
            f"- Model Type: {metadata.get('model', {}).get('type', 'N/A')}",
            f"- Random Seed: {metadata.get('config', {}).get('random_seed', 'N/A')}",
            f"",
            f"## Test Metrics (Before Calibration)",
        ]
        
        test_metrics_before = metrics.get('test', {}).get('test_before_calibration', {})
        for key, value in test_metrics_before.items():
            if isinstance(value, float):
                readme_lines.append(f"- {key.upper()}: {value:.4f}")
            else:
                readme_lines.append(f"- {key.upper()}: {value}")
        
        readme_lines.extend([
            f"",
            f"## Test Metrics (After Calibration)",
        ])
        
        test_metrics_after = metrics.get('test', {}).get('test_after_calibration', {})
        for key, value in test_metrics_after.items():
            if isinstance(value, float):
                readme_lines.append(f"- {key.upper()}: {value:.4f}")
            else:
                readme_lines.append(f"- {key.upper()}: {value}")
        
        readme_lines.extend([
            f"",
            f"## Files",
            f"- `metadata.json`: full run metadata",
            f"- `metrics.json`: all evaluation metrics",
            f"- `predictions.csv`: per-row predictions",
            f"- `curves/`: ROC, PR and calibration plots",
            f"- `interpretability/`: SHAP output",
            f""
        ])
        
        readme_path = os.path.join(protocol_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write('\n'.join(readme_lines))
        
        logger.info(f"README written: {readme_path}")
    
    def create_summary_report(self, all_protocols_results: Dict[str, Dict]):
        """
        Write a summary report covering every protocol.
        
        Args:
            all_protocols_results: {protocol_name: results_dict}
        """
        summary_path = os.path.join(self.output_dir, 'summary_report.json')
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'protocols': {}
        }
        
        for protocol_name, results in all_protocols_results.items():
            summary['protocols'][protocol_name] = {
                'test_metrics_before': results.get('metrics', {}).get('test', {}).get('test_before_calibration', {}),
                'test_metrics_after': results.get('metrics', {}).get('test', {}).get('test_after_calibration', {})
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report written: {summary_path}")


class MLflowTracker:
    """
    Optional MLflow-backed experiment tracking.
    """
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Args:
            experiment_name: name of the experiment
            tracking_uri: MLflow tracking server URI
        """
        try:
            import mlflow
            self.mlflow = mlflow
            self.enabled = True
        except ImportError:
            logger.warning("MLflow is not installed; skipping MLflow tracking")
            self.enabled = False
            return
        
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        
        self.mlflow.set_experiment(experiment_name)
        self.run = None
        logger.info(f"MLflow tracker ready: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None):
        """Start an MLflow run."""
        if not self.enabled:
            return
        
        self.run = self.mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run started: {self.run.info.run_id}")
    
    def log_params(self, params: Dict):
        """Log parameters."""
        if not self.enabled:
            return
        
        self.mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics."""
        if not self.enabled:
            return
        
        self.mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, filepath: str):
        """Log an artifact."""
        if not self.enabled:
            return
        
        self.mlflow.log_artifact(filepath)
    
    def end_run(self):
        """End the MLflow run."""
        if not self.enabled:
            return
        
        self.mlflow.end_run()
        logger.info("MLflow run ended")

