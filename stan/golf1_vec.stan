data {
  int J;
  int n[J];
  vector[J] x;
  int y[J];
  real r;
  real R;
}
transformed data {
  vector[J] asin_R_minus_r_div_x = to_vector(rep_array(asin(R-r), J)) ./ x;
}
parameters {
  real<lower=0> sigma;
}
model {
  vector[J] p = 2*Phi(asin_R_minus_r_div_x / sigma) - 1;
  y ~ binomial(n, p);
}
generated quantities {
  real sigma_degrees;
  sigma_degrees = (180/pi())*sigma;
}

