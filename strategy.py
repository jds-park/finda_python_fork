import numpy as np
def strategy(
    return_array: np.ndarray,
    universe_array: np.ndarray,
    common_feature_array: np.ndarray,
    specific_feature_array: np.ndarray,
):
    return_array = np.array(return_array)
    universe_array = np.array(universe_array)
    specific_feature_array = np.array(specific_feature_array)
    common_feature_array = np.array(common_feature_array)

    T, N = return_array.shape
    investable_mask = universe_array[-1, :]

    momentum = np.sum(return_array[-5:, :], axis=0)

    volatility = np.std(return_array[-20:, :], axis=0)
    volatility[volatility < 1e-6] = 1e-6
    vol_factor = 1.0 / volatility

    if len(specific_feature_array.shape) == 3:
        fundamental = specific_feature_array[-1, :, 0]
    else:
        fundamental = specific_feature_array[-1, :]
    fundamental = np.nan_to_num(fundamental, nan=0.0)

    try:
        market = np.mean(common_feature_array[-1, :])
        market_trend = 1 if market > np.mean(
            common_feature_array[-15:, :]) else -1
    except:
        market_trend = 1

    def standardize(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-6)

    if len(fundamental) == len(momentum):
        alpha = (0.45 * standardize(momentum) * market_trend +
                 0.15 * standardize(fundamental) +
                 0.40 * standardize(vol_factor))
    else:
        alpha = standardize(momentum) * market_trend

    alpha[~investable_mask] = -np.inf

    top_k = min(50, N)
    selected = np.argsort(-alpha)[:top_k]
    weights = np.zeros(N)
    weights[selected] = 1.0 / top_k

    return weights