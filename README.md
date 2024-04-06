# OpenSMDREnv: Reinforcement learning environment for spring-mass-damper system
Rohan Thorat, Juhi Singh, Prof. Rajdip Nayek 

This package includes 1-DOF, 3-DOF, 5-DOF, and 76-DOF dynamical system environment.

Users can change the default mass, spring, and damper properties, i.e. mass, stiffness, and damping coefficient.

Requires:
* `numpy`
* `gymnasium`
* `scipy`

## 3-DOF
Default system parameter values are taken from this article: https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9399(1989)115:8(1609)

$M = \begin{bmatrix}    1 & 1 & 2 \\ 2 & 4 $ 1\end{bmatrix}$

## 5-DOF
Default system parameter values are taken from this article: https://ascelibrary.org/doi/10.1061/%28ASCE%29EM.1943-7889.0001226

## 76-DOF
The 76-DOF dynamical system is of 76-story 306 m concrete office tower proposed for the city of Melbourne, Australia. It is a benchmark system which is usually used in the domain of mechanical and civil engineering.
Default system parameter values are taken from this article: https://ascelibrary.org/doi/10.1061/%28ASCE%29EM.1943-7889.0001226

*We hope you find this code helpful*


