# Hopfield Network in C++

| ![](models/food.png) | ![](output/donut.gif)  | ![](output/burger.gif) |
|----------------------|------------------------|------------------------|

A simple C++ implementation of a Hopfield Network with Hebbian learning rules.

### Mathematical Details

We structure $n$ images with $p$ pixels each into an $n\times p$ matrix called $M$ (memory).

Then, we initialize the network weights given by $W\in\mathbb{R}^{p\times p}$ where,

$$W = \frac{1}{n}M^\top M$$

To run inference, we initialize a state vector $\vec{s}\in\mathbb{R}^p$ with random integer entries between -2 and 2. Finally, we iteratively update random entries of $\vec{s}$ according to the following rule,

$$s_i= \begin{cases}
1, &         \vec{s}\cdot W_i \ge0,\\
-1, &         \text{otherwise}
\end{cases}$$
