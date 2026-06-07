"""自动超参数调优模块"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """超参数自动调优器"""

    def __init__(
        self,
        model_type: str,
        n_trials: int = 50,
        cv_folds: int = 3,
        scoring: str = "neg_mean_squared_error",
        random_state: int = 42,
    ):
        """
        Args:
            model_type: 模型类型 ('ridge', 'hgb', 'rf', 'xgb', 'lgb')
            n_trials: 优化试验次数
            cv_folds: 交叉验证折数
            scoring: 评分指标
            random_state: 随机种子
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna 未安装，请运行: pip install optuna")

        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = 0.0
        self.study_: optuna.Study | None = None

    def _objective_ridge(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Ridge 回归超参数空间"""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
        solver = trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sag"])

        model = Ridge(alpha=alpha, solver=solver, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1)
        return scores.mean()

    def _objective_hgb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """HistGradientBoosting 超参数空间"""
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_score

        max_iter = trial.suggest_int("max_iter", 100, 500)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 63)
        l2_regularization = trial.suggest_float("l2_regularization", 0.0, 10.0)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 50)

        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
        )
        scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1)
        return scores.mean()

    def _objective_rf(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """RandomForest 超参数空间"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 5, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=self.random_state,
        )
        scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1)
        return scores.mean()

    def _objective_xgb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """XGBoost 超参数空间"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost 未安装，请运行: pip install xgboost")

        from sklearn.model_selection import cross_val_score

        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
        )
        scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1)
        return scores.mean()

    def _objective_lgb(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """LightGBM 超参数空间"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")

        from sklearn.model_selection import cross_val_score

        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        num_leaves = trial.suggest_int("num_leaves", 20, 100)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1)
        return scores.mean()

    def optimize(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> dict[str, Any]:
        """
        执行超参数优化

        Args:
            X: 特征数据
            y: 目标变量

        Returns:
            最佳超参数字典
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # 选择目标函数
        objective_map = {
            "ridge": self._objective_ridge,
            "hgb": self._objective_hgb,
            "rf": self._objective_rf,
            "random_forest": self._objective_rf,
            "xgb": self._objective_xgb,
            "xgboost": self._objective_xgb,
            "lgb": self._objective_lgb,
            "lightgbm": self._objective_lgb,
        }

        if self.model_type not in objective_map:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        objective_func = objective_map[self.model_type]

        # 创建 study
        self.study_ = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # 执行优化
        self.study_.optimize(
            lambda trial: objective_func(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value

        return self.best_params_

    def get_optimization_history(self) -> pd.DataFrame:
        """获取优化历史"""
        if self.study_ is None:
            raise RuntimeError("尚未执行优化")

        trials = self.study_.trials
        history = []

        for trial in trials:
            record = {
                "trial": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                **trial.params,
            }
            history.append(record)

        return pd.DataFrame(history)


def auto_tune_model(
    model_type: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    n_trials: int = 50,
    cv_folds: int = 3,
) -> tuple[dict[str, Any], float]:
    """
    自动调优模型超参数（便捷函数）

    Args:
        model_type: 模型类型
        X_train: 训练特征
        y_train: 训练目标
        n_trials: 试验次数
        cv_folds: 交叉验证折数

    Returns:
        (最佳参数, 最佳得分)
    """
    tuner = HyperparameterTuner(
        model_type=model_type,
        n_trials=n_trials,
        cv_folds=cv_folds,
    )

    best_params = tuner.optimize(X_train, y_train)
    best_score = tuner.best_score_

    return best_params, best_score
