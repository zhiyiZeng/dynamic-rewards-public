# Exp1

## 文件说明

- 1-clean_origin_data.ipynb: 用于清洗原始数据，之后的数据都是从这里来的。
- 1-前复权数据绘图.ipynb
- 2-main-multi.py: single reward的多进程（但是经常会出现起两个进程就直接卡死的情况，不知道为什么，怀疑是Pool有问题）
- 2-main-single.ipynb: 不晓得，单进程跑single reward，一直没用。
- 3-1-single-reward表现.ipynb：单一reward函数的买卖点位绘图。
- 4-wilcoxon-test.ipynb：对结果做wilcoxon test检验。复现步骤

1. 跑 `2-main-multi.py`
2. 跑 `4-wilcoxon-test.ipynb`

其他文件按需求跑即可。

# Exp2

基本同上，除了reward函数是regularized。

# Exp3

在4个时间区间内，基于FP5, FPR-X进行ts/greedy的选择。

1. 跑2-main-multi.py（这个文件和Exp1，Exp2仅仅在于reward函数和时间区间的不同，所以可以复用）。
2. 4-1-ts.ipynb：用ts方法选择rewards。
3. 4-2-ts-greedy.ipynb：用greedy方法选择rewards。
4. 4-4-ts-concat-reward.ipynb: 用ts/greedy方法组装选择的rewards。
5. 4-3-single-reward.ipynb: 跑单一reward函数。
6. 4-6-X.ipynb：跑CCI/MA/MACD/MV。
7. 4-7-wilcoxon-test.ipynb：所有rewards和方法跑wilcoxon test。
