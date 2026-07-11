# Model Comparison Report - RBacon ML Project

**Generated:** 2026-05-28 12:44:46

## Models Compared

1. **MLP** - Multi-Layer Perceptron (Original neural network)
2. **Random Forest** - Ensemble of decision trees
3. **XGBoost** - Gradient boosting optimized
4. **LSTM** - Long Short-Term Memory (Sequential - FAILED)

## Overall Performance

| Model | Overall R² | Performance | Rank |
|-------|------------|-------------|------|
| Random Forest | 0.9670 | EXCELLENT ⭐⭐⭐ | #1 |
| XGBoost | 0.9582 | EXCELLENT ⭐⭐⭐ | #2 |
| MLP | 0.9142 | EXCELLENT ⭐⭐⭐ | #3 |
| LSTM | -2562.0911 | POOR ❌ | #4 |

## Per-Target R² Comparison

| Target | Random Forest | XGBoost | MLP | LSTM | Best Model |
|--------|--------|--------|--------|--------|------------|
| mean_age | 0.9996 | 1.0000 | 0.9980 | -17.1875 | XGBoost |
| median_age | 0.9999 | 0.9997 | 0.9918 | -0.6677 | Random Forest |
| lo95 | 0.9999 | 0.9999 | 0.9907 | -0.9226 | Random Forest |
| hi95 | 0.9995 | 1.0000 | 0.9984 | -282.6386 | XGBoost |
| ci_width | 0.9996 | 1.0000 | 0.9985 | -17632.5367 | XGBoost |
| acc_rate | 0.7973 | 0.7517 | 0.5670 | -0.4731 | Random Forest |
| sed_rate | 0.9730 | 0.9562 | 0.8547 | -0.2118 | Random Forest |

## Recommendation

**🏆 BEST MODEL: RANDOM FOREST**

- Highest overall R² (0.967)
- Lowest error rates (MAE = 51 years)
- Excellent across all target variables
- Most reliable for production deployment