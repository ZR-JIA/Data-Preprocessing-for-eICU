"""
可复现性模块
包括种子管理、实验跟踪、结果导出
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
    随机种子管理器
    管理多个随机种子集合，确保可复现性
    """
    
    def __init__(self, base_seed: int = 42, n_seeds: int = 5):
        """
        Args:
            base_seed: 基础种子
            n_seeds: 种子数量（≥5个）
        """
        self.base_seed = base_seed
        self.n_seeds = max(n_seeds, 5)
        self.seed_list = self._generate_seeds()
        
    def _generate_seeds(self) -> List[int]:
        """生成种子列表"""
        np.random.seed(self.base_seed)
        seeds = [self.base_seed] + [int(s) for s in np.random.randint(0, 10000, size=self.n_seeds - 1)]
        logger.info(f"生成 {self.n_seeds} 个随机种子: {seeds}")
        return seeds
    
    def set_seed(self, seed: int):
        """
        设置所有随机种子
        
        Args:
            seed: 随机种子
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
        
        logger.info(f"已设置随机种子: {seed}")
    
    def get_seeds(self) -> List[int]:
        """获取种子列表"""
        return self.seed_list


class DataVersionManager:
    """
    数据版本管理器
    记录数据版本哈希
    """
    
    @staticmethod
    def compute_data_hash(df: pd.DataFrame) -> str:
        """
        计算数据的哈希值
        
        Args:
            df: 数据框
            
        Returns:
            SHA256 哈希值
        """
        # 将数据框转为字符串并计算哈希
        data_str = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        hash_value = hashlib.sha256(data_str).hexdigest()
        logger.info(f"数据哈希值: {hash_value[:16]}...")
        return hash_value
    
    @staticmethod
    def save_data_info(
        df: pd.DataFrame,
        save_path: str,
        additional_info: Optional[Dict] = None
    ):
        """
        保存数据信息
        
        Args:
            df: 数据框
            save_path: 保存路径
            additional_info: 额外信息
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
        
        logger.info(f"数据信息已保存: {save_path}")


class ExperimentTracker:
    """
    实验跟踪器
    记录训练与评估全流程
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "../../experiments"):
        """
        Args:
            experiment_name: 实验名称
            output_dir: 输出目录
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
        
        logger.info(f"实验跟踪器已初始化: {self.experiment_dir}")
    
    def log_config(self, config: Dict):
        """记录配置"""
        self.metadata['config'] = config
        logger.info("已记录配置")
    
    def log_data_info(self, data_hash: str, data_shape: tuple, feature_list: List[str]):
        """记录数据信息"""
        self.metadata['data_info'] = {
            'hash': data_hash,
            'shape': data_shape,
            'n_features': len(feature_list),
            'features': feature_list
        }
        logger.info("已记录数据信息")
    
    def log_split_indices(self, split_indices: Dict[str, List[int]]):
        """记录切分索引"""
        self.metadata['split_indices'] = split_indices
        
        # 保存为单独文件
        split_path = os.path.join(self.experiment_dir, 'split_indices.json')
        with open(split_path, 'w') as f:
            json.dump(split_indices, f, indent=2)
        
        logger.info(f"已记录切分索引: {split_path}")
    
    def log_preprocessing(self, preprocessing_info: Dict):
        """记录预处理信息"""
        self.metadata['preprocessing'] = preprocessing_info
        logger.info("已记录预处理信息")
    
    def log_model(self, model_info: Dict):
        """记录模型信息"""
        self.metadata['model'] = model_info
        logger.info("已记录模型信息")
    
    def log_metrics(self, split: str, metrics: Dict, before_calibration: bool = True):
        """
        记录评估指标
        
        Args:
            split: 'train', 'val', 'test'
            metrics: 指标字典
            before_calibration: 是否校准前
        """
        key = f"{split}_{'before' if before_calibration else 'after'}_calibration"
        if split not in self.metadata['metrics']:
            self.metadata['metrics'][split] = {}
        
        self.metadata['metrics'][split][key] = metrics
        logger.info(f"已记录 {split} 指标 ({'校准前' if before_calibration else '校准后'})")
    
    def log_calibration(self, calibration_info: Dict):
        """记录校准信息"""
        self.metadata['calibration'] = calibration_info
        logger.info("已记录校准信息")
    
    def log_interpretability(self, interpretability_info: Dict):
        """记录可解释性信息"""
        self.metadata['interpretability'] = interpretability_info
        logger.info("已记录可解释性信息")
    
    def save_metadata(self):
        """保存元数据"""
        self.metadata['end_time'] = datetime.now().isoformat()
        
        metadata_path = os.path.join(self.experiment_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"元数据已保存: {metadata_path}")
    
    def save_artifact(self, obj: Any, filename: str):
        """
        保存实验产物
        
        Args:
            obj: 要保存的对象
            filename: 文件名
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
                raise ValueError("CSV 格式要求输入 DataFrame")
        else:
            raise ValueError(f"不支持的文件格式: {filename}")
        
        logger.info(f"产物已保存: {filepath}")
    
    def get_experiment_dir(self) -> str:
        """获取实验目录"""
        return self.experiment_dir


class ResultsExporter:
    """
    结果导出器
    导出三套协议的独立结果包
    """
    
    def __init__(self, output_dir: str = "../../results"):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"结果导出器已初始化: {output_dir}")
    
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
        导出单个协议的结果包
        
        Args:
            protocol_name: 协议名称 ('patient_level', 'temporal', 'site')
            version: 版本号
            metadata: 元数据
            metrics: 评估指标
            predictions: 预测结果
            curves_dir: 曲线图目录
            interpretability_dir: 可解释性结果目录
        """
        # 创建协议专用目录
        protocol_dir = os.path.join(
            self.output_dir,
            f"{protocol_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(protocol_dir, exist_ok=True)
        
        # 保存元数据
        metadata_path = os.path.join(protocol_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 保存指标
        metrics_path = os.path.join(protocol_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 保存预测结果
        if predictions is not None:
            predictions_path = os.path.join(protocol_dir, 'predictions.csv')
            predictions.to_csv(predictions_path, index=False)
        
        # 复制曲线图
        if curves_dir and os.path.exists(curves_dir):
            import shutil
            curves_target = os.path.join(protocol_dir, 'curves')
            shutil.copytree(curves_dir, curves_target, dirs_exist_ok=True)
        
        # 复制可解释性结果
        if interpretability_dir and os.path.exists(interpretability_dir):
            import shutil
            interp_target = os.path.join(protocol_dir, 'interpretability')
            shutil.copytree(interpretability_dir, interp_target, dirs_exist_ok=True)
        
        logger.info(f"协议结果已导出: {protocol_dir}")
        
        # 创建 README
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
        """创建 README 文件"""
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
            f"- `metadata.json`: 完整元数据",
            f"- `metrics.json`: 所有评估指标",
            f"- `predictions.csv`: 预测结果",
            f"- `curves/`: ROC、PR、Calibration 曲线图",
            f"- `interpretability/`: SHAP 解释结果",
            f""
        ])
        
        readme_path = os.path.join(protocol_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write('\n'.join(readme_lines))
        
        logger.info(f"README 已创建: {readme_path}")
    
    def create_summary_report(self, all_protocols_results: Dict[str, Dict]):
        """
        创建所有协议的汇总报告
        
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
        
        logger.info(f"汇总报告已创建: {summary_path}")


class MLflowTracker:
    """
    MLflow 跟踪器
    （可选）使用 MLflow 进行实验跟踪
    """
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Args:
            experiment_name: 实验名称
            tracking_uri: MLflow 跟踪服务器 URI
        """
        try:
            import mlflow
            self.mlflow = mlflow
            self.enabled = True
        except ImportError:
            logger.warning("MLflow 未安装，跳过 MLflow 跟踪")
            self.enabled = False
            return
        
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        
        self.mlflow.set_experiment(experiment_name)
        self.run = None
        logger.info(f"MLflow 跟踪器已初始化: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None):
        """开始 MLflow run"""
        if not self.enabled:
            return
        
        self.run = self.mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run 已开始: {self.run.info.run_id}")
    
    def log_params(self, params: Dict):
        """记录参数"""
        if not self.enabled:
            return
        
        self.mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """记录指标"""
        if not self.enabled:
            return
        
        self.mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, filepath: str):
        """记录产物"""
        if not self.enabled:
            return
        
        self.mlflow.log_artifact(filepath)
    
    def end_run(self):
        """结束 MLflow run"""
        if not self.enabled:
            return
        
        self.mlflow.end_run()
        logger.info("MLflow run 已结束")

