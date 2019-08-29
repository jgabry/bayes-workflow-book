data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] eta;
}
model {
  eta ~ normal(0, 1);
  y ~ normal(mu + tau*eta, sigma);
}
