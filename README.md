# The Metropolis-Hastings algorithm

The Metropolis-Hastings algorithm is a Markov chain Monte Carlo method used to acquire a sequence of random samples from a probability distribution 
where direct sampling is difficult. This sequence can be used to compute an integral or to estimate the distribution. It randomly takes different samples from a space and determines if the current sample is more likely to come from the posterior than the previous sample.

[More details about the algorithm](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm)

[A video to further understand the algorithm](https://www.youtube.com/watch?v=0lpT-yveuIA)


## Flowchart of Metropolis-Hastings (M-H) algorithm for the parameter estimation using the Markov Chain Monte Carlo (MCMC) approach

![Flowchart of Metropolis-Hastings (M-H) algorithm for the parameter estimation using the Markov Chain Monte Carlo (MCMC) approach](images/Metr-Hast-algorithm.png)


## Code Overview

We explore three different probability density functions, the first is 2-dim and bimodal, the other two are 10-dim and unimodal MVNs.
We analyse the behavior of the algorithm and we explore its critical parameters.
