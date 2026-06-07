"""模型预测报告生成器：预测与实际对比可视化"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class PredictionReportGenerator:
    """预测报告生成器"""

    def __init__(self, output_dir: Path | str = "reports"):
        """
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        predictions_df: pd.DataFrame,
        prediction_col: str,
        actual_col: str,
        model_name: str = "model",
        date_col: str | None = None,
        code_col: str | None = None,
    ) -> dict[str, Path]:
        """
        生成完整预测报告

        Args:
            predictions_df: 预测数据框
            prediction_col: 预测值列名
            actual_col: 实际值列名
            model_name: 模型名称
            date_col: 日期列名
            code_col: 股票代码列名

        Returns:
            生成的文件路径字典
        """
        output_files = {}

        print(f"生成 {model_name} 预测报告...")

        # 1. 预测 vs 实际时间序列图
        print("  生成时间序列对比图...")
        ts_path = self._plot_prediction_vs_actual_timeseries(
            predictions_df, prediction_col, actual_col, model_name, date_col, code_col
        )
        output_files["timeseries"] = ts_path

        # 2. 散点图和回归线
        print("  生成散点图...")
        scatter_path = self._plot_prediction_scatter(
            predictions_df, prediction_col, actual_col, model_name
        )
        output_files["scatter"] = scatter_path

        # 3. 误差分布图
        print("  生成误差分析图...")
        error_path = self._plot_error_distribution(
            predictions_df, prediction_col, actual_col, model_name
        )
        output_files["error"] = error_path

        # 4. 分股票对比图（如果有 code_col）
        if code_col and code_col in predictions_df.columns:
            print("  生成分股票对比图...")
            by_code_path = self._plot_by_code(
                predictions_df, prediction_col, actual_col, model_name, code_col
            )
            output_files["by_code"] = by_code_path

        # 5. 综合仪表板
        print("  生成综合仪表板...")
        dashboard_path = self._plot_dashboard(
            predictions_df, prediction_col, actual_col, model_name
        )
        output_files["dashboard"] = dashboard_path

        # 6. 生成 Markdown 报告
        print("  生成 Markdown 报告...")
        report_path = self._generate_markdown_report(
            predictions_df, prediction_col, actual_col, model_name, output_files
        )
        output_files["report"] = report_path

        print(f"✓ 报告生成完成，保存至: {self.output_dir}")
        return output_files

    def _plot_prediction_vs_actual_timeseries(
        self,
        df: pd.DataFrame,
        pred_col: str,
        actual_col: str,
        model_name: str,
        date_col: str | None,
        code_col: str | None,
    ) -> Path:
        """绘制预测vs实际时间序列"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 如果有多只股票，随机选择一只展示
        if code_col and code_col in df.columns:
            unique_codes = df[code_col].unique()
            sample_code = unique_codes[0] if len(unique_codes) > 0 else None
            df_plot = df[df[code_col] == sample_code].copy() if sample_code else df.copy()
            title_suffix = f" (股票: {sample_code})" if sample_code else ""
        else:
            df_plot = df.copy()
            title_suffix = ""

        # 排序
        if date_col and date_col in df_plot.columns:
            df_plot = df_plot.sort_values(date_col).reset_index(drop=True)
            x = df_plot[date_col]
            xlabel = "日期"
        else:
            df_plot = df_plot.reset_index(drop=True)
            x = df_plot.index
            xlabel = "样本序号"

        # 上图：预测值 vs 实际值
        axes[0].plot(x, df_plot[actual_col], label="实际值", color="#2E86AB", linewidth=2, alpha=0.8)
        axes[0].plot(x, df_plot[pred_col], label="预测值", color="#A23B72", linewidth=2, alpha=0.8)
        axes[0].fill_between(x, df_plot[actual_col], df_plot[pred_col], alpha=0.2, color="gray")
        axes[0].set_title(f"{model_name} - 预测 vs 实际{title_suffix}", fontsize=14, fontweight="bold")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel("收益率")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        # 下图：预测误差
        error = df_plot[pred_col] - df_plot[actual_col]
        axes[1].plot(x, error, color="#F18F01", linewidth=1.5, alpha=0.8)
        axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1].fill_between(x, 0, error, alpha=0.3, color="#F18F01")
        axes[1].set_title("预测误差（预测 - 实际）", fontsize=12)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("误差")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / f"{model_name}_timeseries.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _plot_prediction_scatter(
        self, df: pd.DataFrame, pred_col: str, actual_col: str, model_name: str
    ) -> Path:
        """绘制预测vs实际散点图"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # 散点图
        ax.scatter(
            df[actual_col],
            df[pred_col],
            alpha=0.5,
            s=30,
            color="#2E86AB",
            edgecolors="white",
            linewidths=0.5,
        )

        # 理想预测线（y=x）
        min_val = min(df[actual_col].min(), df[pred_col].min())
        max_val = max(df[actual_col].max(), df[pred_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="理想预测 (y=x)")

        # 回归线
        z = np.polyfit(df[actual_col], df[pred_col], 1)
        p = np.poly1d(z)
        ax.plot(
            df[actual_col],
            p(df[actual_col]),
            "g-",
            linewidth=2,
            alpha=0.8,
            label=f"回归线 (y={z[0]:.2f}x+{z[1]:.2f})",
        )

        # 计算 R²
        from sklearn.metrics import r2_score

        r2 = r2_score(df[actual_col], df[pred_col])

        ax.set_xlabel("实际值", fontsize=12)
        ax.set_ylabel("预测值", fontsize=12)
        ax.set_title(f"{model_name} - 预测 vs 实际散点图\nR² = {r2:.4f}", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / f"{model_name}_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _plot_error_distribution(
        self, df: pd.DataFrame, pred_col: str, actual_col: str, model_name: str
    ) -> Path:
        """绘制误差分布图"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig)

        error = df[pred_col] - df[actual_col]

        # 1. 误差直方图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(error, bins=50, color="#2E86AB", alpha=0.7, edgecolor="black")
        ax1.axvline(0, color="red", linestyle="--", linewidth=2)
        ax1.axvline(error.mean(), color="green", linestyle="--", linewidth=2, label=f"均值: {error.mean():.4f}")
        ax1.set_xlabel("误差")
        ax1.set_ylabel("频数")
        ax1.set_title("误差分布直方图")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 误差 QQ 图
        ax2 = fig.add_subplot(gs[0, 1])
        from scipy import stats

        stats.probplot(error, dist="norm", plot=ax2)
        ax2.set_title("误差 Q-Q 图（正态性检验）")
        ax2.grid(True, alpha=0.3)

        # 3. 绝对误差分布
        ax3 = fig.add_subplot(gs[1, 0])
        abs_error = np.abs(error)
        ax3.hist(abs_error, bins=50, color="#A23B72", alpha=0.7, edgecolor="black")
        ax3.axvline(abs_error.mean(), color="green", linestyle="--", linewidth=2, label=f"MAE: {abs_error.mean():.4f}")
        ax3.set_xlabel("绝对误差")
        ax3.set_ylabel("频数")
        ax3.set_title("绝对误差分布")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 误差统计
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        stats_text = f"""
误差统计：

均值 (Mean):        {error.mean():.6f}
中位数 (Median):     {error.median():.6f}
标准差 (Std):        {error.std():.6f}
最小值 (Min):        {error.min():.6f}
最大值 (Max):        {error.max():.6f}

MAE:                {abs_error.mean():.6f}
RMSE:               {np.sqrt((error**2).mean()):.6f}
MAPE:               {(abs_error / (np.abs(df[actual_col]) + 1e-8)).mean() * 100:.2f}%

偏度 (Skewness):    {error.skew():.4f}
峰度 (Kurtosis):    {error.kurtosis():.4f}
        """

        ax4.text(
            0.1, 0.5, stats_text, fontsize=11, verticalalignment="center", family="monospace"
        )

        plt.suptitle(f"{model_name} - 误差分析", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = self.output_dir / f"{model_name}_error_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _plot_by_code(
        self, df: pd.DataFrame, pred_col: str, actual_col: str, model_name: str, code_col: str
    ) -> Path:
        """按股票代码绘制对比图"""
        unique_codes = df[code_col].unique()
        n_codes = min(len(unique_codes), 9)  # 最多显示9只股票

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()

        for i in range(n_codes):
            code = unique_codes[i]
            df_code = df[df[code_col] == code].copy().reset_index(drop=True)

            ax = axes[i]
            ax.plot(df_code.index, df_code[actual_col], label="实际", color="#2E86AB", linewidth=2)
            ax.plot(df_code.index, df_code[pred_col], label="预测", color="#A23B72", linewidth=2)
            ax.fill_between(
                df_code.index, df_code[actual_col], df_code[pred_col], alpha=0.2, color="gray"
            )

            # 计算该股票的 R²
            from sklearn.metrics import r2_score

            r2 = r2_score(df_code[actual_col], df_code[pred_col])

            ax.set_title(f"{code} (R²={r2:.3f})", fontsize=10)
            ax.set_xlabel("样本")
            ax.set_ylabel("收益率")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(n_codes, len(axes)):
            axes[i].axis("off")

        plt.suptitle(f"{model_name} - 分股票预测对比", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = self.output_dir / f"{model_name}_by_code.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _plot_dashboard(
        self, df: pd.DataFrame, pred_col: str, actual_col: str, model_name: str
    ) -> Path:
        """综合仪表板"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        error = df[pred_col] - df[actual_col]
        abs_error = np.abs(error)

        # 1. 时间序列（大图）
        ax1 = fig.add_subplot(gs[0, :])
        sample_size = min(len(df), 500)
        df_sample = df.head(sample_size).reset_index(drop=True)
        ax1.plot(df_sample.index, df_sample[actual_col], label="实际值", color="#2E86AB", linewidth=2)
        ax1.plot(df_sample.index, df_sample[pred_col], label="预测值", color="#A23B72", linewidth=2)
        ax1.fill_between(
            df_sample.index, df_sample[actual_col], df_sample[pred_col], alpha=0.2, color="gray"
        )
        ax1.set_title(f"{model_name} - 预测 vs 实际（前{sample_size}样本）", fontsize=12, fontweight="bold")
        ax1.set_xlabel("样本序号")
        ax1.set_ylabel("收益率")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 散点图
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(df[actual_col], df[pred_col], alpha=0.4, s=20, color="#2E86AB")
        min_val = min(df[actual_col].min(), df[pred_col].min())
        max_val = max(df[actual_col].max(), df[pred_col].max())
        ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        from sklearn.metrics import r2_score
        r2 = r2_score(df[actual_col], df[pred_col])
        ax2.set_title(f"散点图 (R²={r2:.4f})")
        ax2.set_xlabel("实际值")
        ax2.set_ylabel("预测值")
        ax2.grid(True, alpha=0.3)

        # 3. 误差直方图
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(error, bins=50, color="#F18F01", alpha=0.7, edgecolor="black")
        ax3.axvline(0, color="red", linestyle="--", linewidth=2)
        ax3.set_title("误差分布")
        ax3.set_xlabel("误差")
        ax3.set_ylabel("频数")
        ax3.grid(True, alpha=0.3)

        # 4. 误差箱线图
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.boxplot([error], vert=True, patch_artist=True, boxprops=dict(facecolor="#A23B72", alpha=0.7))
        ax4.axhline(0, color="red", linestyle="--", linewidth=2)
        ax4.set_title("误差箱线图")
        ax4.set_ylabel("误差")
        ax4.grid(True, alpha=0.3)

        # 5. 累积误差
        ax5 = fig.add_subplot(gs[2, 0])
        cumulative_error = error.cumsum()
        ax5.plot(cumulative_error, color="#2E86AB", linewidth=2)
        ax5.axhline(0, color="red", linestyle="--", linewidth=1)
        ax5.set_title("累积误差")
        ax5.set_xlabel("样本")
        ax5.set_ylabel("累积误差")
        ax5.grid(True, alpha=0.3)

        # 6. 滚动 MAE
        ax6 = fig.add_subplot(gs[2, 1])
        window = 50
        rolling_mae = abs_error.rolling(window=window).mean()
        ax6.plot(rolling_mae, color="#F18F01", linewidth=2)
        ax6.axhline(abs_error.mean(), color="green", linestyle="--", linewidth=2, label=f"平均 MAE: {abs_error.mean():.4f}")
        ax6.set_title(f"滚动 MAE（窗口={window}）")
        ax6.set_xlabel("样本")
        ax6.set_ylabel("MAE")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. 性能指标
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae = mean_absolute_error(df[actual_col], df[pred_col])
        rmse = np.sqrt(mean_squared_error(df[actual_col], df[pred_col]))
        mape = (abs_error / (np.abs(df[actual_col]) + 1e-8)).mean() * 100

        metrics_text = f"""
性能指标：

R² Score:      {r2:.6f}
MAE:           {mae:.6f}
RMSE:          {rmse:.6f}
MAPE:          {mape:.2f}%

误差统计：
均值:          {error.mean():.6f}
标准差:        {error.std():.6f}
最大正误差:    {error.max():.6f}
最大负误差:    {error.min():.6f}
        """

        ax7.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment="center", family="monospace")

        plt.suptitle(f"{model_name} - 预测性能综合仪表板", fontsize=16, fontweight="bold")

        output_path = self.output_dir / f"{model_name}_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def _generate_markdown_report(
        self,
        df: pd.DataFrame,
        pred_col: str,
        actual_col: str,
        model_name: str,
        output_files: dict[str, Path],
    ) -> Path:
        """生成 Markdown 报告"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        error = df[pred_col] - df[actual_col]
        abs_error = np.abs(error)

        # 计算指标
        r2 = r2_score(df[actual_col], df[pred_col])
        mae = mean_absolute_error(df[actual_col], df[pred_col])
        rmse = np.sqrt(mean_squared_error(df[actual_col], df[pred_col]))
        mape = (abs_error / (np.abs(df[actual_col]) + 1e-8)).mean() * 100

        # 生成 Markdown
        report_content = f"""# {model_name} - 预测报告

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| **R² Score** | {r2:.6f} |
| **MAE** | {mae:.6f} |
| **RMSE** | {rmse:.6f} |
| **MAPE** | {mape:.2f}% |

---

## 📈 误差统计

| 统计量 | 数值 |
|--------|------|
| 均值 | {error.mean():.6f} |
| 中位数 | {error.median():.6f} |
| 标准差 | {error.std():.6f} |
| 最小值 | {error.min():.6f} |
| 最大值 | {error.max():.6f} |
| 偏度 | {error.skew():.4f} |
| 峰度 | {error.kurtosis():.4f} |

---

## 📉 可视化图表

### 1. 预测 vs 实际时间序列

![时间序列]({output_files.get('timeseries', Path()).name})

### 2. 散点图

![散点图]({output_files.get('scatter', Path()).name})

### 3. 误差分析

![误差分析]({output_files.get('error', Path()).name})

"""

        if "by_code" in output_files:
            report_content += f"""### 4. 分股票对比

![分股票对比]({output_files['by_code'].name})

"""

        report_content += f"""### 5. 综合仪表板

![综合仪表板]({output_files.get('dashboard', Path()).name})

---

## 📝 数据概览

- **样本数量**: {len(df)}
- **预测列**: `{pred_col}`
- **实际列**: `{actual_col}`

---

## 🎯 结论

"""

        # 自动生成结论
        if r2 > 0.8:
            report_content += "- ✅ 模型拟合优秀 (R² > 0.8)\n"
        elif r2 > 0.6:
            report_content += "- ✅ 模型拟合良好 (R² > 0.6)\n"
        elif r2 > 0.4:
            report_content += "- ⚠️ 模型拟合一般 (R² > 0.4)\n"
        else:
            report_content += "- ❌ 模型拟合较差 (R² < 0.4)\n"

        if abs(error.mean()) < 0.01:
            report_content += "- ✅ 预测无明显偏差\n"
        else:
            bias_direction = "高估" if error.mean() > 0 else "低估"
            report_content += f"- ⚠️ 预测存在系统性{bias_direction}（均值误差: {error.mean():.4f}）\n"

        if mape < 10:
            report_content += "- ✅ 相对误差小 (MAPE < 10%)\n"
        elif mape < 20:
            report_content += "- ⚠️ 相对误差中等 (MAPE < 20%)\n"
        else:
            report_content += "- ❌ 相对误差较大 (MAPE > 20%)\n"

        report_content += "\n---\n\n*本报告由 hp-ml 自动生成*\n"

        # 保存报告
        report_path = self.output_dir / f"{model_name}_prediction_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_path


def generate_prediction_report(
    predictions_df: pd.DataFrame,
    prediction_col: str,
    actual_col: str,
    model_name: str = "model",
    output_dir: Path | str = "reports",
    date_col: str | None = None,
    code_col: str | None = None,
) -> dict[str, Path]:
    """
    生成预测报告（便捷函数）

    Args:
        predictions_df: 预测数据框
        prediction_col: 预测值列名
        actual_col: 实际值列名
        model_name: 模型名称
        output_dir: 输出目录
        date_col: 日期列名
        code_col: 股票代码列名

    Returns:
        生成的文件路径字典
    """
    generator = PredictionReportGenerator(output_dir=output_dir)

    return generator.generate_full_report(
        predictions_df=predictions_df,
        prediction_col=prediction_col,
        actual_col=actual_col,
        model_name=model_name,
        date_col=date_col,
        code_col=code_col,
    )
