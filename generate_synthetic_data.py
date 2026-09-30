"""
generate_synthetic_data.py

Generates a synthetic dataset of financial transactions for prototype
development and testing of the fraud-detection module.

IMPORTANT: This is SYNTHETIC data generated for prototype/demonstration
purposes only. It does not contain, represent, or derive from any real
customer, account, or transaction data from any financial institution.

Author: Fatima Altagracia Genao Mendez
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ACCOUNTS = 500
N_TRANSACTIONS = 20000
FRAUD_RATE = 0.015  # ~1.5% of transactions flagged as fraudulent in the synthetic ground truth

rng = np.random.default_rng(RANDOM_SEED)


def generate_accounts(n_accounts: int) -> pd.DataFrame:
    account_ids = [f"ACC{100000 + i}" for i in range(n_accounts)]
    account_types = rng.choice(["checking", "savings", "business"], size=n_accounts, p=[0.55, 0.30, 0.15])
    avg_monthly_activity = rng.gamma(shape=2.0, scale=800, size=n_accounts)  # typical spend profile per account
    return pd.DataFrame({
        "account_id": account_ids,
        "account_type": account_types,
        "avg_monthly_activity": avg_monthly_activity.round(2),
    })


def generate_transactions(accounts: pd.DataFrame, n_transactions: int) -> pd.DataFrame:
    n_accounts = len(accounts)
    sender_idx = rng.integers(0, n_accounts, size=n_transactions)

    # Most transactions go to a small pool of frequent counterparties (merchants),
    # a minority go to other individual accounts (peer transfers) which is where
    # layering/money-laundering-style patterns are easier to simulate.
    is_peer_transfer = rng.random(n_transactions) < 0.25
    receiver_idx = np.where(
        is_peer_transfer,
        rng.integers(0, n_accounts, size=n_transactions),
        -1,  # merchant sentinel; replaced below
    )
    merchant_ids = rng.integers(0, 40, size=n_transactions)  # 40 synthetic merchant codes

    base_amounts = rng.lognormal(mean=4.0, sigma=1.0, size=n_transactions)  # typical purchase amounts
    hours = rng.integers(0, 24, size=n_transactions)
    channel = rng.choice(["ACH", "card_present", "card_not_present", "wire"], size=n_transactions,
                          p=[0.35, 0.30, 0.25, 0.10])

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:07d}" for i in range(n_transactions)],
        "sender_account": accounts["account_id"].values[sender_idx],
        "is_peer_transfer": is_peer_transfer,
        "receiver_account": np.where(
            is_peer_transfer,
            accounts["account_id"].values[receiver_idx],
            [f"MERCH{m:03d}" for m in merchant_ids],
        ),
        "amount": base_amounts.round(2),
        "hour_of_day": hours,
        "channel": channel,
    })

    # ---- Inject synthetic fraud patterns ----
    n_fraud = int(n_transactions * FRAUD_RATE)
    fraud_idx = rng.choice(n_transactions, size=n_fraud, replace=False)
    df["is_fraud"] = False
    df.loc[fraud_idx, "is_fraud"] = True

    # Pattern A: unusually large amount relative to typical activity, at an unusual hour
    half = n_fraud // 2
    df.loc[fraud_idx[:half], "amount"] = df.loc[fraud_idx[:half], "amount"] * rng.uniform(8, 25, size=half)
    df.loc[fraud_idx[:half], "hour_of_day"] = rng.integers(0, 5, size=half)  # late-night activity
    df.loc[fraud_idx[:half], "channel"] = "card_not_present"

    # Pattern B: rapid small peer-to-peer transfers between a tight, closed cluster
    # of accounts (a simplified "layering" / structuring pattern for the graph
    # analysis component). Extra synthetic edges are added within the same small
    # cluster so it forms a dense, self-contained sub-network distinguishable
    # from the sparser, randomly-connected legitimate peer-transfer traffic.
    layering_accounts = accounts["account_id"].values[rng.integers(0, n_accounts, size=6)]
    remaining = fraud_idx[half:]
    df.loc[remaining, "is_peer_transfer"] = True
    df.loc[remaining, "sender_account"] = rng.choice(layering_accounts, size=len(remaining))
    df.loc[remaining, "receiver_account"] = rng.choice(layering_accounts, size=len(remaining))
    df.loc[remaining, "amount"] = rng.uniform(150, 900, size=len(remaining)).round(2)  # kept below common reporting thresholds

    extra_n = 60
    extra_senders = rng.choice(layering_accounts, size=extra_n)
    extra_receivers = rng.choice(layering_accounts, size=extra_n)
    extra = pd.DataFrame({
        "transaction_id": [f"TXNL{i:05d}" for i in range(extra_n)],
        "sender_account": extra_senders,
        "is_peer_transfer": True,
        "receiver_account": extra_receivers,
        "amount": rng.uniform(150, 900, size=extra_n).round(2),
        "hour_of_day": rng.integers(0, 24, size=extra_n),
        "channel": "ACH",
        "is_fraud": True,
    })
    df = pd.concat([df, extra], ignore_index=True)

    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    accounts = generate_accounts(N_ACCOUNTS)
    transactions = generate_transactions(accounts, N_TRANSACTIONS)

    accounts.to_csv("data_accounts.csv", index=False)
    transactions.to_csv("data_transactions.csv", index=False)

    print(f"Generated {len(accounts)} synthetic accounts and {len(transactions)} synthetic transactions.")
    print(f"Synthetic fraud rate: {transactions['is_fraud'].mean():.3%}")
