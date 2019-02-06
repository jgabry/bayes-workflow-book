
# Fake-data Simulation

## "Hello World" example

We demonstrate fake-data simulation with the linear regression example from the previous chapter.  Here is the Stan program:


```
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real a;
  real b;
  real<lower=0> sigma;
}
model {
  y ~ normal(a + b * x, sigma);
}
```

### Simulating fake data conditional on assumed parameter values

We need to construct the data:  N, x, and y.  But to do this we must also pick assumed values for the parameters:  a, b, sigma.

We do this as follows:
- Set N = 100
- Set a = 2
- Set b = 3
- Set sigma = 5
- Sample x as N independent values uniformly distributed between 0 and 10
- Sample y_n, n=1,...,N independently with mean a + b*x and standard deviation sigma.

These steps can all be done in R as in the linked code.



### Fitting the Stan model and reading the output

We can now fit the Stan model, passing in the data list N, x, y.




Here is the summary of the fitted model:


```
Inference for Stan model: simplest-regression.
4 chains, each with iter=2000; warmup=1000; thin=1; 
post-warmup draws per chain=1000, total post-warmup draws=4000.

        mean se_mean   sd   2.5%    25%    50%    75%    98% n_eff Rhat
a        3.4    0.03 1.01    1.4    2.7    3.4    4.1    5.4  1604    1
b        2.8    0.00 0.17    2.5    2.7    2.8    2.9    3.1  1589    1
sigma    4.9    0.01 0.36    4.3    4.7    4.9    5.2    5.7  2166    1
lp__  -207.0    0.03 1.25 -210.3 -207.6 -206.7 -206.1 -205.6  1494    1

Samples were drawn using NUTS(diag_e) at Mon Jan 14 21:19:43 2019.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```

(The below description will all change as we are changing default Stan output.)

Now we go through the output:

* The first few lines summarize the Stan run, with the name of the
  file, the number of chains and iterations.  In this case, Stan ran
  the default 4 chains with 1000 warmup iterations followed by 1000
  post-warmup iterations, yielding 4000 post-warmup simulation draws
  in total.

* The left column of the table has the names of parameters,
  transformed parameters, and generated quantities produced by
  `model.stan`.  In this case, the parameters are a, b, and sigma; the
  only transformed parameter is `lp__` (the log-posterior density or
  target function created by the Stan model); and there are no
  generated quantities.

* The next column of the table shows the mean (average) of the 4000
  draws for each quantity.

* The next column shows the Monte Carlo standard error, which is an
  estimate of the uncertainty in the mean.

* The next column shows the standard deviation of the draws for each
  quantity.  As the number of simulation draws increases, mean should
  approach the posterior mean, se_mean should go to zero, and sd
  should approach the posterior standard deviation.  For most purposes
  we can ignore se_mean.

* The next several columns give quantiles of the simulations.

* The next columns gives the effective sample size and
  $\widehat{R}$. Typically we want $\widehat{R}$ to be less then 1.1
  for each row of the table.

In the above output, $\widehat{R}$ is less then 1.1 for all
quantities, so the chains seem to have mixed well, and we use the
results to summarize the posterior distribution.

### Comparing fitted model to assumed parameter values

We can compare the
posterior inferences to the assumed parameter values (here, $a=2$, $b=3$,
and $\sigma=5$).  These assumed values are roughly within the range of
uncertainty of the inferences.

## Prior predictive simulation

Do an example here

## Topics in fake-data simulation

We will expand upon these topics:

- Choosing assumed parameter values (as in above example) or drawing parameters from the prior

- When choosing assumed parameter values can give a bad answer (if they're in the tail of the prior)

- Fake-data simulation for hierarchical models:  You can choose assumed values for the hyperparameters but you should draw the lower-level parameters from the model

- Choosing or drawing random values for unmodeled data such as N and x

- What if fake-data simulation gives a bad answer, if the model cannot recover the assumed parameter values?
