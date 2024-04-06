# OpenSMDREnv: Reinforcement learning environment for spring-mass-damper system
Rohan Thorat, Juhi Singh, Prof. Rajdip Nayek 

This package includes 1-DOF, 3-DOF, 5-DOF, and 76-DOF dynamical system environment.
To install this package, run the below command:
```{bash}
pip install OpenSMRDEnv
```


Users can change the default mass, spring, and damper properties, i.e. mass, stiffness, and damping coefficient.

Requires:
* `numpy`
* `gymnasium`
* `scipy`

## Environments detail
### 1-DOF:
Default system parameter values are defined as below:

$\Delta t = 0.01$

<p align="center">
  <img src="imgs/1DOF.png">
</p>

### 3-DOF:
Default system parameter values are taken from this article: https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9399(1989)115:8(1609)

$\Delta t = 0.01$

<p align="center">
  <img src="imgs/3DOF.png">
</p>

### 5-DOF:
Default system parameter values are taken from this article: https://ascelibrary.org/doi/10.1061/%28ASCE%29EM.1943-7889.0001226

$\Delta t = 0.01$

<p align="center">
  <img src="imgs/5DOF.png">
</p>

The exact values of the M, K, and C matrix can be seen in the file named '5dof_MKC_matrix.mat' located in the 'src' folder.

### 76-DOF:
The 76-DOF dynamical system is of 76-story 306 m concrete office tower proposed for the city of Melbourne, Australia. It is a benchmark system which is usually used in the domain of mechanical and civil engineering.
Default system parameter values are taken from this article: https://ascelibrary.org/doi/10.1061/%28ASCE%29EM.1943-7889.0001226

$\Delta t = 0.01$

The value of the M, K, and C matrix can be seen in the file named '76dof_MKC_matrix.mat' located in the 'src' folder.


## Add one example of how to call the environment function

*We hope you find this code helpful*


