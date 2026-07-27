"""
Model versioning and registry system.
Tracks model versions, metadata, and enables rollback/comparison.
"""

import json
import os
import hashlib
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import joblib
import pandas as pd

from config.settings import MODELS_DIR
from utils.logging_setup import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    """Registry for tracking and managing model versions."""
    
    def __init__(self):
        self.registry_file = os.path.join(MODELS_DIR, "model_registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing registry or create new one."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load registry: {e}. Creating new registry.")
        return {}
    
    def _save_registry(self):
        """Save registry to disk."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=4, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_model(self, symbol: str, interval: str, model_data: Dict, metrics: Dict) -> str:
        """
        Register a new model version.
        
        Args:
            symbol: Asset symbol
            interval: Timeframe (1d, 1h, 5m)
            model_data: Dict with 'models' and 'meta_stacker'
            metrics: Dict with validation metrics
        
        Returns:
            version_id: Unique version identifier
        """
        key = f"{symbol}_{interval}"
        version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if key not in self.registry:
            self.registry[key] = {"versions": []}
        
        # Create version metadata
        version_meta = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "interval": interval,
            "metrics": metrics,
            "model_hash": self._compute_model_hash(model_data),
            "status": "active"
        }
        
        self.registry[key]["versions"].append(version_meta)
        self.registry[key]["latest"] = version_id
        
        logger.info(f"Registered model {symbol}_{interval} as {version_id}")
        self._save_registry()
        
        return version_id
    
    def _compute_model_hash(self, model_data: Dict) -> str:
        """Compute hash of model for deduplication."""
        try:
            data_str = json.dumps(model_data, default=str, sort_keys=True)
            return hashlib.md5(data_str.encode()).hexdigest()[:8]
        except:
            return "unknown"
    
    def get_model_info(self, symbol: str, interval: str, version_id: Optional[str] = None) -> Optional[Dict]:
        """Retrieve model metadata."""
        key = f"{symbol}_{interval}"
        if key not in self.registry:
            return None
        
        if version_id is None:
            version_id = self.registry[key].get("latest")
        
        for version in self.registry[key]["versions"]:
            if version["version_id"] == version_id:
                return version
        
        return None
    
    def list_versions(self, symbol: str, interval: str) -> List[Dict]:
        """List all versions for a symbol-interval pair."""
        key = f"{symbol}_{interval}"
        if key not in self.registry:
            return []
        return self.registry[key]["versions"]
    
    def archive_old_versions(self, symbol: str, interval: str, keep_count: int = 5):
        """Archive old versions, keeping only the latest N."""
        key = f"{symbol}_{interval}"
        if key not in self.registry:
            return
        
        versions = self.registry[key]["versions"]
        if len(versions) > keep_count:
            to_archive = versions[:-keep_count]
            for version in to_archive:
                version["status"] = "archived"
            logger.info(f"Archived {len(to_archive)} versions for {key}")
            self._save_registry()


class ModelStorage:
    """Handle model persistence with versioning."""
    
    def __init__(self):
        self.registry = ModelRegistry()
    
    def save_model(self, symbol: str, interval: str, model_data: Dict, metrics: Dict) -> str:
        """
        Save model with versioning.
        
        Returns:
            version_id: The version ID of saved model
        """
        version_id = self.registry.register_model(symbol, interval, model_data, metrics)
        
        safe_symbol = symbol.replace("=", "_").replace("^", "_")
        version_dir = os.path.join(MODELS_DIR, f"{safe_symbol}_{interval}_{version_id}")
        os.makedirs(version_dir, exist_ok=True)
        
        # Save models
        model_path = os.path.join(version_dir, "models.pkl")
        joblib.dump(model_data, model_path)
        
        # Save metrics
        metrics_path = os.path.join(version_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        
        logger.info(f"Saved model {symbol}_{interval} to {version_dir}")
        return version_id
    
    def load_model(self, symbol: str, interval: str, version_id: Optional[str] = None) -> Optional[Dict]:
        """Load model by version."""
        if version_id is None:
            info = self.registry.get_model_info(symbol, interval)
            if info:
                version_id = info["version_id"]
            else:
                logger.warning(f"No model found for {symbol}_{interval}")
                return None
        
        safe_symbol = symbol.replace("=", "_").replace("^", "_")
        version_dir = os.path.join(MODELS_DIR, f"{safe_symbol}_{interval}_{version_id}")
        model_path = os.path.join(version_dir, "models.pkl")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            model_data = joblib.load(model_path)
            logger.info(f"Loaded model {symbol}_{interval} from {version_id}")
            return model_data
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def compare_versions(self, symbol: str, interval: str, version_ids: List[str]) -> pd.DataFrame:
        """Compare metrics across multiple model versions."""
        comparison = []
        for version_id in version_ids:
            info = self.registry.get_model_info(symbol, interval, version_id)
            if info:
                row = {
                    "version_id": version_id,
                    "timestamp": info.get("timestamp"),
                    **info.get("metrics", {})
                }
                comparison.append(row)
        
        return pd.DataFrame(comparison) if comparison else pd.DataFrame()
