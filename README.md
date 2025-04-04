# savvyGLM: Shrinkage Methods for Generalized Linear Models

The `savvyGLM` package offers a complete framework for fitting shrinkage estimators in *generalized linear models (GLMs)*. It integrates several shrinkage methods into the *Iteratively Reweighted Least Squares (IRLS)* algorithm to improve both convergence and estimation accuracy, particularly when standard maximum likelihood estimation is challenged by issues like multicollinearity or a high number of predictors. Shrinkage estimators introduce a small bias that produces a large reduction in variance, making the IRLS estimates more reliable than those based solely on the traditional OLS update. For further details on the shrinkage estimators employed in this package, please refer to [*Slab and Shrinkage Linear Regression Estimation*](http//...).

This package builds on theoretical work discussed in:

Asimit, V., Chen, Z., Dimitrova, D., Xie, Y., & Zhang, Y. (2025). [Shrinkage GLM Modeling](http//...).

The official documentation site is available at: <https://Ziwei-ChenChen.github.io/savvyGLM>

If you are interested in applying shrinkage methods within linear regression, please refer to the companion package [`savvySh`](https://github.com/Ziwei-ChenChen/savvySh).

## Installation Guide

You can install the development version of `savvyGLM` directly from GitHub:

``` r
remotes::install_github("Ziwei-ChenChen/savvyGLM")
```

Once installed, load the package:

``` r
library(savvyGLM)
```

## Features

The package supports several shrinkage approaches within IRLS algorithm, including:

-   **Stein Estimator (St):** Applies a single shrinkage factor to all coefficients.

-   **Diagonal Shrinkage (DSh):** Applies separate shrinkage factors to each coefficient.

-   **Slab Regression (SR):** Adds a penalty that shrinks the solution along a fixed direction.

-   **Generalized Slab Regression (GSR):** Extends SR by allowing shrinkage along multiple directions.

-   **Shrinkage Estimator (Sh):** Uses a full shrinkage matrix estimated by solving a *Sylvester equation*. This method is optional due to its higher computational cost.

These methods build on the robust features of the `glm2` package—such as step-halving—to further enhance the performance of GLMs.

## Usage

This is a basic example that shows you how to solve a common problem:

``` r
set.seed(123)
n <- 100
p <- 5

x <- matrix(rnorm(n * p), n, p)
beta_true <- c(1, 0.5, -0.5, 1, -1)
eta <- x %*% beta_true
prob <- 1 / (1 + exp(-eta))
y <- rbinom(n, size = 1, prob = prob)

fit <- savvy_glm.fit2(
  x      = cbind(1, x), 
  y      = y,
  model_class = "SR",
  family = binomial(link = "logit"),
  control = glm.control(trace = TRUE)
)

print(fit$coefficients)
print(fit$chosen_fit)
```

## Authors

-   Ziwei Chen – [ziwei.chen.3\@citystgeorges.ac.uk](ziwei.chen.3@citystgeorges.ac.uk)

-   Vali Asimit – [asimit\@citystgeorges.ac.uk](asimit@citystgeorges.ac.uk)

## License

This package is licensed under the MIT License.
