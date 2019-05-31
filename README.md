# RL CartPole-v1

## 依赖

- matplotlib  
- pytorch  
- gym  
- numpy

## 运行

对任务1直接运行代码即可：
```
python ppo.py
```

对任务2需要将代码中`is_reducce_ob`改为`True`，然后运行：
```
python ppo.py
```

对任务3：
```
pyhton r2d2.py
```

## 算法

### PPO

#### 任务1（217步）

![ppo_task1](img/ppo_task1_217.png)

#### 任务2（2896步）

![ppo_task2](img/ppo_task2_2896.png)

### DQN

#### 任务1（289步）

![dqn_task1](img/dqn_task1_289.png)

#### 任务2（2453步）

![dqn_task2](img/dqn_task2_2453.png)

### Hill Climbing

#### 任务1（58步）

![hc_task1](img/hc_task1_58.png)

#### 任务2（426步）

![hc_task2](img/hc_task2_426.png)

### R2D2

#### 任务3（409步）

![r2d2_task3](img/r2d2_task3_409.png)

## 结果

**参数调整：** 所有的算法的超参数都选择的是该算法最常用的值。

**输入特征：** 对于任务1，输入特征为当前状态（4个值）。 对于任务2，输入为前一个状态与当前状态（4个值）。
对任务3输入仅为当前状态（2个值）。

**结果评估：** 对于任务1，PPO与DQN的表现相仿，都能在尝试200-300次后解决问题；HC略快，只需要
30-60次左右。 对于任务2，PPO与DQN大致需要接近2500-3000次尝试；HC则仅仅需要几十到几百次。 所以
从实验结果来看对于CartPole这种相对简单的问题，HC的结果更好一些。 但是这三种算法都无法解决任务3，
所以最终选择使用基于LSTM的R2D2，在大概400轮左右成功解决。

HC任务1结果分布（100次）
![hc_dist](img/hc_dist.png)

*其余任务运行时间较久所以结果分布分析略过*