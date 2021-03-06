(Note: moved from `stan-dev/bayes-workflow-book`)

This is the repository for the book *Bayesian Workflow Using Stan* (working title). The book will have many authors.

The book is in its early stages of development so the content on the master branch will change substantially.

### Rules for working on workflow

 * _Branch_ (on a well named branch) and then submit a pull request for merging into master.
 * At **all times** the master branch should compile. This is required for merging.
 * Keep a list of packages that are needed to compile the book and add to it if you add a package
 

### Directory Structure

* `*.Rmd` files: book sections to potentially include (not all are currently included)
* `_bookdown.yml`: book includes (only the Rmd files listed here are included in the book)
* `_output.yml`: output config
* `stan/*.stan` : directory of Stan programs
* `data/{*.R, *.rds}` : directory for data used by programs
* `bibtex/all.bib`: BibTeX file for references
* `programs/{*.R, *.stan}` : legacy programs from old manual (deprecated until
  they're moved into new style with R inline in .Rmd)


### Building the Book from Source

You will need to have RStan installed in the R environment from which
you build.

#### RStudio

In RStudio: to build the project, open `index.Rmd` in RStudio and click `knit`
    - change output on first line of `index.Rmd` for `gitbook` and `pdf_book` (not differeing `_`)

#### Outside of RStudio

First, you will need to install `pandoc` and `pandoc-citeproc` in
addition to the `bookdown` package in R.  After that, it can be built
from within R in this directory using `bookdown::render_book('index.Rmd')`
or from the shell using `./build.sh` to build both PDF and HTML
versions.


### Style Guide for Authors

* All lines should be 80 or fewer characters unless absolutely
mandated by content

* y ~ normal(mu, sigma) # Not: N(), not sigma^2, roman font for
  "normal", LaTeX math for $y$, $\mu$, $\sigma$

* normal(y | mu, sigma) # Vertical bar, not semicolon

* Poisson, Weibull, LKJ # Use capital letters for distributions that
  are named after people

* E(y)  # Roman font for E, LaTeX math for $y$, parentheses not brackets

* ()  # Always parentheses, never brackets

* No special fonts for distributions, just roman and math fonts

* p(y) # Probability density and probability mass function

* Pr(A)  # probability of an event

* Follow the Stan style guide for code
    - int<lower = 0> N;  # Put in the lower bound
    - for (n in 1:N); # Not:  for (i in 1:n);
    - foo_bar # Underscores rather than dots or CamelCase

* All Stan code should be best practice except when explaining
  something, in which case we should explicitly show the best-practice
  alternative


### Licensing

The code is licensed under BSD-3 and the text under CC-BY ND 4.0.

