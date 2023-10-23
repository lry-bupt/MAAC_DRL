# MAAC_DRL
This repository contains the Python implementation of our submitted paper titled "Deep Reinforcement Learning for Joint Trajectory and Communication Design in Internet of Robotic Things" .
## Quick Links
[[Installation]](#installation)  [[Installation]](#installation) [[Usage]](#usage) 
## Introduction
We learn the multi-agent actor-critic deep reinforcement learning (MAAC-DRL) algorithms to reduce the decoding error rate and arriving time of robots in industrial Internet of Robotic Things () with the requirements of ultra-reliable and low-latency communications.

Here are the settings of the considered IoRT environment.
| Notation     | Simulation Value   | Physical Meaning                                             |
| ------------ | ------------------ | ------------------------------------------------------------ |
| $K$      | $\{2, 4, 6\}$                | the number of users    |
| $L$ | $\{2, 3\}$     | the number of antennas    |
| $K_{\rm MU}$ | $\{1, 2, 3\}$     | the number of robots     |
| $D$        | $100 \ {\rm bits}$      | packet size    |
| $M$        | $50 \ {\rm symbols}$     | the number of transmitted symbols    |
| $T_{\max}$   | $2000 \ {\rm s}$ | the moving deadline of robots   |
| $H_0$  | $1 \ {\rm m}$   | the height of antennas     |
| $P_{\max}$  | $[0.02, 0.1] \ {\rm W}$   | the maximal transmit power |
| $\sigma^2$     | $-100 \ {\rm dBm/Hz}$   | the variance of the additive white Gaussian noise                  |
| $v$          | $5 \ {\rm m/s}$    | the moving speed    |



## Results
<table style="padding: 0; border-spacing: 0;">
<tr style="padding: 0; border-spacing: 0;">
<td style="padding: 0; border-spacing: 0; width: 50%"><img src="./_doc/simulation_fig.png"></td>
<td style="padding: 0; border-spacing: 0; width: 50%"><img src="./_doc/simulation_fig2.png"></td>
</tr>
</table>

For more details and simulation results, please check our paper.

## Installation
Dependencies can be installed by Conda:

For example to install env used for IoRT environments with URLLC requirements:
```
conda env create -f environment/environment.yml URLLC
conda activate URLLC
```

Then activate it by
```
conda activate URLLC
```
To run on atari environment, please further install the considered environment by 
```
pip install -r requirements.txt
```

## Usage

Here are the parameters of our simulations.
| Notation     | Simulation Value   | Physical Meaning                                             |
| ------------ | ------------------ | ------------------------------------------------------------ |
| $lr$      | $\{10^{-4}, 2 \times 10^{-3}\}$                | the learning rate of the DRL algorithms    |
| $\kappa_1$ | $\{0, 0.01, 0.1\}$     | the parameters of the reward designs    |
| $\|\mathcal{D}_0\|$ | $128$     | the size of the mini-batch buffer   |
| $\|\mathcal{D}\|$        | $10^{6}$      | the maximal size of the experevce buffer    |


