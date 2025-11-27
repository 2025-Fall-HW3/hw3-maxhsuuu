"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import sys
import argparse


"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

start = "2019-01-01"
end = "2024-04-01"

# Initialize df and df_returns
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust = False)
    df[asset] = raw['Adj Close']

df_returns = df.pct_change().fillna(0)


"""
Problem 1: 

Implement an equal weighting strategy as dataframe "eqw". Please do "not" include SPY.
"""

class EqualWeightPortfolio:
    def __init__(self, exclude):
        # exclude：要排除的 ticker（例如 "SPY"）
        self.exclude = exclude

    def calculate_weights(self):
        # 要投資的資產（排除 exclude，那就是 11 個 sector）
        assets = df.columns[df.columns != self.exclude]

        # 注意：助教原本的 skeleton 就是用 df.columns
        # 所以我們照樣建立一個和 df 同 index / columns 的權重表
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 1 Below
        """

        import numpy as np

        n = len(assets)
        # 等權重 [1/N, 1/N, ..., 1/N]
        w_eq = np.ones(n) / n

        # 每一天的權重都一樣
        for date in df.index:
            self.portfolio_weights.loc[date, assets] = w_eq
            # 被排除的那個 column（self.exclude）保持 NaN，後面會被補成 0

        """
        TODO: Complete Task 1 Above
        """

        # 助教 template 本來就有這兩行，保持不動
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Problem 2:

Implement a risk parity strategy as dataframe "rp". Please do "not" include SPY.
"""

class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 2 Below
        """

        # 只用（不含 SPY）的報酬率
        returns = df_returns[assets]

        # 先把 SPY 權重設成 0
        self.portfolio_weights[self.exclude] = 0.0

        # 從第 (lookback + 1) 筆交易日開始算權重
        # 讓第一天有權重的日期 ≈ 第 52 個交易日（對應 rp.pkl 的 2019-03-18）
        for t in range(self.lookback + 1, len(returns)):
            # 用「前 50 天」的報酬算波動度（不含當天）
            window = returns.iloc[t - self.lookback : t]   # 長度 = lookback
            sigma = window.std()                           # 各資產波動度

            # 逆波動度
            inv_sigma = 1 / sigma

            # 避免除以 0 變成 inf
            inv_sigma.replace([np.inf, -np.inf], np.nan, inplace=True)
            inv_sigma = inv_sigma.fillna(0)

            # 如果這個視窗全部是 0，就跳過這一天（不更新權重）
            if inv_sigma.sum() == 0:
                continue

            # 正規化成權重，使 Σ w_i = 1
            w_t = inv_sigma / inv_sigma.sum()

            # 寫回這一天的權重
            date = returns.index[t]
            self.portfolio_weights.loc[date, assets] = w_t.values

        """
        TODO: Complete Task 2 Above
        """

        # 用前一日權重補缺值，剩下的再補 0
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)


    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns



"""
Problem 3:

Implement a Markowitz strategy as dataframe "mv". Please do "not" include SPY.
"""


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # 不包含 SPY
        assets = df.columns[df.columns != self.exclude]

        # 權重 DataFrame
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        # 從 lookback+1 開始（和 skeleton 一樣，第一個 rebalance 日在第 lookback+1 筆）
        for i in range(self.lookback + 1, len(df)):
            # 用「過去 lookback 天、不含今天」的報酬
            R_n = df_returns.copy()[assets].iloc[i - self.lookback : i]
            # 解 Markowitz 問題
            self.portfolio_weights.loc[df.index[i], assets] = self.mv_opt(
                R_n, self.gamma
            )

        # SPY 權重維持 0
        self.portfolio_weights[self.exclude] = 0.0

        # 時間上補齊
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        # 區間內的樣本 μ、Σ
        Sigma = R_n.cov().values      # (m x m)
        mu = R_n.mean().values        # (m,)
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                """
                TODO: Complete Task 3 Below
                """

                # 決策變數：各資產權重 w_i
                # long-only -> lb = 0；ub=1 雖然在 sum(w)=1 下是多餘的，但保留也沒關係
                w = model.addMVar(n, lb=0.0, ub=1.0, name="w")

                # 預期報酬項 μ^T w
                linear_term = mu @ w

                # 風險項 w^T Σ w
                quad_term = w @ Sigma @ w

                # 目標：max μ^T w − (γ/2) · w^T Σ w
                model.setObjective(
                    linear_term - (gamma / 2.0) * quad_term,
                    gp.GRB.MAXIMIZE,
                )

                # 預算約束：sum w_i = 1
                model.addConstr(w.sum() == 1, name="budget")

                """
                TODO: Complete Task 3 Above
                """
                model.optimize()

                # 狀態檢查（保留原本 skeleton 的寫法）
                if model.status == gp.GRB.INF_OR_UNBD:
                    print(
                        "Model status is INF_OR_UNBD. Reoptimizing with DualReductions set to 0."
                    )
                elif model.status == gp.GRB.INFEASIBLE:
                    print("Model is infeasible.")
                elif model.status == gp.GRB.INF_OR_UNBD:
                    print("Model is infeasible or unbounded.")

                solution = [0.0] * n
                if model.status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
                    for i in range(n):
                        var = model.getVarByName(f"w[{i}]")
                        solution[i] = var.X

        return solution

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns



if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 1"
    )
    """
    NOTE: For Assignment Judge
    """
    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader.py
    judge.run_grading(args)

