# Acknowledgments

This project contains the code for the paper: [Multi-step reward ensemble methods for adaptive stock trading](https://www.sciencedirect.com/science/article/abs/pii/S0957417423010497). Our codes are based on the repository from this paper: [Learning financial asset-specific trading rules via deep reinforcement learning](https://www.sciencedirect.com/science/article/abs/pii/S0957417422000239).

We would like to express our sincere gratitude to the authors of [Learning financial asset-specific trading rules via deep reinforcement learning](https://www.sciencedirect.com/science/article/abs/pii/S0957417422000239) for their open-source repository, which provided significant inspiration and foundation for our work. Their innovative approach to financial trading using deep reinforcement learning has been instrumental in shaping our research direction.


# Exp1

## File Description

- 1-clean_origin_data.ipynb: Used for cleaning raw data. All subsequent data originates from this process.
- 1-forward_adjusted_data_plot.ipynb
- 2-main-multi.py: Multi-process implementation for single reward (Note: Sometimes it freezes when starting two processes, possibly due to Pool issues)
- 2-main-single.ipynb: Single process implementation for single reward (not frequently used)
- 3-1-single-reward_performance.ipynb: Plotting buy/sell points for single reward function.
- 4-wilcoxon-test.ipynb: Wilcoxon test for result validation. Reproduction steps:

1. Run `2-main-multi.py`
2. Run `4-wilcoxon-test.ipynb`

Other files can be run as needed.

# Exp2

Basically the same as Exp1, except using regularized reward functions.

# Exp3

Selecting rewards based on FP5 and FPR-X using ts/greedy methods across four time periods.

1. Run 2-main-multi.py (This file differs from Exp1 and Exp2 only in reward functions and time periods, so it can be reused)
2. 4-1-ts.ipynb: Select rewards using ts method
3. 4-2-ts-greedy.ipynb: Select rewards using greedy method
4. 4-4-ts-concat-reward.ipynb: Combine selected rewards using ts/greedy methods
5. 4-3-single-reward.ipynb: Run single reward function
6. 4-6-X.ipynb: Run CCI/MA/MACD/MV
7. 4-7-wilcoxon-test.ipynb: Run Wilcoxon test for all rewards and methods