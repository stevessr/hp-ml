"""模型集成学习框架：Stacking 和 Blending"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """Stacking 集成学习器"""

    def __init__(
        self,
        base_models: list[Any],
        meta_model: Any | None = None,
        cv_folds: int = 5,
        use_original_features: bool = False,
        random_state: int = 42,
    ):
        """
        Args:
            base_models: 基础模型列表
            meta_model: 元模型（默认使用 Ridge）
            cv_folds: 交叉验证折数
            use_original_features: 是否在元模型中使用原始特征
            random_state: 随机种子
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.cv_folds = cv_folds
        self.use_original_features = use_original_features
        self.random_state = random_state
        self.base_models_fitted_: list[Any] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> StackingEnsemble:
        """训练 Stacking 模型"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        n_samples = X.shape[0]
        n_base_models = len(self.base_models)

        # 生成元特征
        meta_features = np.zeros((n_samples, n_base_models))

        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for i, model in enumerate(self.base_models):
            print(f"训练基础模型 {i+1}/{n_base_models}...")

            for train_idx, val_idx in kfold.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]

                # 训练基础模型
                model_clone = self._clone_model(model)
                model_clone.fit(X_train_fold, y_train_fold)

                # 生成验证集预测作为元特征
                meta_features[val_idx, i] = model_clone.predict(X_val_fold)

        # 在全数据上训练所有基础模型
        self.base_models_fitted_ = []
        for i, model in enumerate(self.base_models):
            print(f"在全数据上训练基础模型 {i+1}/{n_base_models}...")
            model_clone = self._clone_model(model)
            model_clone.fit(X, y)
            self.base_models_fitted_.append(model_clone)

        # 训练元模型
        print("训练元模型...")
        if self.use_original_features:
            meta_X = np.hstack([meta_features, X])
        else:
            meta_X = meta_features

        self.meta_model.fit(meta_X, y)

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]
        n_base_models = len(self.base_models_fitted_)

        # 生成基础模型预测
        meta_features = np.zeros((n_samples, n_base_models))
        for i, model in enumerate(self.base_models_fitted_):
            meta_features[:, i] = model.predict(X)

        # 元模型预测
        if self.use_original_features:
            meta_X = np.hstack([meta_features, X])
        else:
            meta_X = meta_features

        return self.meta_model.predict(meta_X)

    def _clone_model(self, model: Any) -> Any:
        """克隆模型"""
        from sklearn.base import clone
        try:
            return clone(model)
        except Exception:
            # 对于不支持 clone 的模型，尝试使用 copy
            import copy
            return copy.deepcopy(model)


class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """Blending 集成学习器"""

    def __init__(
        self,
        base_models: list[Any],
        meta_model: Any | None = None,
        holdout_ratio: float = 0.2,
        use_original_features: bool = False,
        random_state: int = 42,
    ):
        """
        Args:
            base_models: 基础模型列表
            meta_model: 元模型（默认使用 Ridge）
            holdout_ratio: holdout 集比例
            use_original_features: 是否在元模型中使用原始特征
            random_state: 随机种子
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.holdout_ratio = holdout_ratio
        self.use_original_features = use_original_features
        self.random_state = random_state
        self.base_models_fitted_: list[Any] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> BlendingEnsemble:
        """训练 Blending 模型"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # 划分 train 和 holdout
        n_samples = X.shape[0]
        n_holdout = int(n_samples * self.holdout_ratio)

        np.random.seed(self.random_state)
        indices = np.random.permutation(n_samples)
        train_idx = indices[n_holdout:]
        holdout_idx = indices[:n_holdout]

        X_train, X_holdout = X[train_idx], X[holdout_idx]
        y_train, y_holdout = y[train_idx], y[holdout_idx]

        # 训练基础模型
        n_base_models = len(self.base_models)
        meta_features = np.zeros((len(holdout_idx), n_base_models))

        for i, model in enumerate(self.base_models):
            print(f"训练基础模型 {i+1}/{n_base_models}...")
            model.fit(X_train, y_train)
            self.base_models_fitted_.append(model)

            # 在 holdout 上预测
            meta_features[:, i] = model.predict(X_holdout)

        # 训练元模型
        print("训练元模型...")
        if self.use_original_features:
            meta_X = np.hstack([meta_features, X_holdout])
        else:
            meta_X = meta_features

        self.meta_model.fit(meta_X, y_holdout)

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]
        n_base_models = len(self.base_models_fitted_)

        # 生成基础模型预测
        meta_features = np.zeros((n_samples, n_base_models))
        for i, model in enumerate(self.base_models_fitted_):
            meta_features[:, i] = model.predict(X)

        # 元模型预测
        if self.use_original_features:
            meta_X = np.hstack([meta_features, X])
        else:
            meta_X = meta_features

        return self.meta_model.predict(meta_X)


class WeightedAverageEnsemble(BaseEstimator, RegressorMixin):
    """加权平均集成"""

    def __init__(self, models: list[Any], weights: list[float] | None = None):
        """
        Args:
            models: 模型列表
            weights: 权重列表（默认等权）
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("权重数量必须与模型数量一致")

        # 归一化权重
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> WeightedAverageEnsemble:
        """训练所有模型"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        for i, model in enumerate(self.models):
            print(f"训练模型 {i+1}/{len(self.models)}...")
            model.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """加权平均预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = np.zeros(len(X))

        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X)

        return predictions


def create_ensemble(
    ensemble_type: str,
    base_models: list[Any],
    meta_model: Any | None = None,
    **kwargs: Any,
) -> BaseEstimator:
    """
    创建集成模型

    Args:
        ensemble_type: 'stacking', 'blending', 'weighted_average'
        base_models: 基础模型列表
        meta_model: 元模型（Stacking/Blending 使用）
        **kwargs: 其他参数

    Returns:
        集成模型实例
    """
    ensemble_type = ensemble_type.lower()

    if ensemble_type == "stacking":
        return StackingEnsemble(base_models, meta_model, **kwargs)

    elif ensemble_type == "blending":
        return BlendingEnsemble(base_models, meta_model, **kwargs)

    elif ensemble_type in {"weighted_average", "weighted"}:
        return WeightedAverageEnsemble(base_models, **kwargs)

    else:
        raise ValueError(f"不支持的集成类型：{ensemble_type}")
