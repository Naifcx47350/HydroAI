# data_generators.py
# -------------------------------
# Generates synthetic data or loads real data
# -------------------------------

import numpy as np
import pandas as pd
import random as rd

# data_generators.py (partial)


def generate_flow_data(num_points=(2880*2), random_state=None):
    """
    Generate synthetic water flow data (e.g., hourly). 
    We'll ensure it stays positive by shifting it up.
    """
    if random_state is not None:
        np.random.seed(random_state)

    t = np.arange(num_points)
    flow = 15 + np.sin(t * 0.1) * 5 + np.random.normal(0, 1, num_points)

    flow[200:220] = 50
    flow[500:520] = 0

    return pd.Series(flow, name='flow_rate')


def generate_quality_data(num_samples=1000, random_state=None):
    """
    Generate synthetic water quality data (pH, TDS, turbidity)
    and label as safe (1) or unsafe (0) based on simple thresholds.

    :param num_samples: Number of samples (int).
    :param random_state: (Optional) int seed for reproducible results.

    If None, no seed is set, so data is random each run.
    """
    if random_state is not None:
        np.random.seed(random_state)

    data = {
        'pH': np.random.normal(7.0, 0.5, num_samples),
        'TDS': np.random.normal(300, 100, num_samples),
        'turbidity': np.random.normal(1.0, 0.3, num_samples)
    }
    df = pd.DataFrame(data)

    # Label water as safe(1) or unsafe(0)
    df['safe'] = 1
    df.loc[
        (df.pH < 6.5) | (df.pH > 8.5) |
        (df.TDS > 500) | (df.turbidity > 5),
        'safe'
    ] = 0

    return df
