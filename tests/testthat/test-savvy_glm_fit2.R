library(testthat)
library(savvyGLM)

test_that("Logistic regression with intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, model_class = c("DSh","SR"), family = binomial(link = "logit")))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Logistic regression without intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  fit <- suppressWarnings(savvy_glm.fit2(x, y, family = binomial(link = "logit"), intercept = FALSE))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Gaussian family with intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y,  model_class = "LW", family = gaussian()))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Gaussian family without intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  fit <- suppressWarnings(savvy_glm.fit2(x, y, family = gaussian(), intercept = FALSE))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Poisson family with intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rpois(n, lambda = 2)
  fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y,  model_class = c("SR", "Sh"), family = poisson()))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Rank-deficient matrix handling", {
  set.seed(123)
  n <- 10
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  x[, 2] <- x[, 1]
  fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, model_class = c("SR", "Sh", "QIS"), family = binomial(link = "logit")))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
})

test_that("All zeros in response variable", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rep(0, n)
  fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, model_class = "St", family = binomial(link = "logit")))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
})

test_that("All ones in response variable", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rep(1, n)
  fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y,  model_class = "GSR", family = binomial(link = "logit")))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
})

test_that("Handling large datasets", {
  set.seed(123)
  n <- 10000
  p <- 50
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, model_class = "SR", family = binomial(link = "logit")))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Offset handling", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rpois(n, lambda = 2)
  offset <- rep(0.5, n)
  fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, family = poisson(), offset = offset))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
  expect_true(all(fit$offset == offset), info = "Offset should be correctly handled")
})

test_that("Weight handling", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  weights <- runif(n)
  fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, family = binomial(link = "logit"), weights = weights))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
  expect_equal(fit$prior.weights, weights, info = "Weights should be correctly handled")
})

test_that("Valideta and Validmu functions", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)

  family <- binomial(link = "logit")
  family$valideta <- function(eta) FALSE

  expect_error(
    savvy_glm.fit2(cbind(1, x), y, family = family),
    "cannot find valid starting values"
  )
})

test_that("Invalid family object", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)

  expect_error(
    savvy_glm.fit2(cbind(1, x), y, family = list()),
    "'family' argument seems not to be a valid family object"
  )
})

test_that("Initialization with mustart", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  mustart <- rep(mean(y), n)
  fit <- suppressWarnings(savvy_glm.fit2(x, y, mustart = mustart, family = gaussian()))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Handling of empty model", {
  n <- 100
  y <- rnorm(n)
  fit <- suppressWarnings(savvy_glm.fit2(matrix(nrow = n, ncol = 0), y, family = gaussian()))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_equal(length(fit$coefficients), 0, info = "Coefficients should be empty for an empty model")
  expect_true(fit$converged, info = "Empty model should converge")
})

test_that("Valideta and validmu functions in empty model", {
  n <- 100
  y <- rnorm(n)

  family <- gaussian()
  family$valideta <- function(eta) all(eta > -1)
  family$validmu <- function(mu) all(mu < 2)

  fit <- suppressWarnings(savvy_glm.fit2(matrix(, n, 0), y, family = family))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_equal(length(fit$coefficients), 0, info = "Coefficients should be empty for an empty model")
  expect_true(fit$converged, info = "Empty model should converge")
})

test_that("Invalid eta in empty model", {
  n <- 100
  y <- rnorm(n)

  family <- gaussian()
  family$valideta <- function(eta) FALSE # Impossible constraint

  expect_error(
    savvy_glm.fit2(matrix(, n, 0), y, family = family),
    "invalid linear predictor values in empty model"
  )
})

test_that("Invalid mu in empty model", {
  n <- 100
  y <- rnorm(n)

  family <- gaussian()
  family$validmu <- function(mu) FALSE

  expect_error(
    savvy_glm.fit2(matrix(, n, 0), y, family = family),
    "invalid fitted means in empty model"
  )
})

test_that("NA values in varmu", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  family <- gaussian()
  family$variance <- function(mu) {
    res <- rep(1, length(mu))
    res[1] <- NA
    res
  }

  expect_error(
    savvy_glm.fit2(x, y, family = family),
    "NAs in V\\(mu\\)"
  )
})

test_that("Zero values in varmu", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  family <- gaussian()
  family$variance <- function(mu) {
    res <- rep(1, length(mu))
    res[1] <- 0
    res
  }

  expect_error(
    savvy_glm.fit2(x, y, family = family),
    "0s in V\\(mu\\)"
  )
})

test_that("NA values in mu.eta.val", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  family <- gaussian()
  family$mu.eta <- function(eta) {
    res <- rep(1, length(eta))
    res[1] <- NA
    res
  }

  expect_error(
    savvy_glm.fit2(x, y, family = family),
    "NAs in d\\(mu\\)/d\\(eta\\)"
  )
})

capture_warnings <- function(expr) {
  warnings <- NULL
  withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  warnings
}

test_that("Trace output for deviance and iterations", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  family <- gaussian()
  family$valideta <- function(eta) TRUE
  family$validmu <- function(mu) TRUE
  control <- list(trace = TRUE, maxit = 1, epsilon = 1e-8)
  trace_output <- capture.output({
    suppressWarnings(savvy_glm.fit2(x, y, family = family, control = control))
  })

  expect_true(any(grepl("Deviance =", trace_output)), info = "Trace output should contain deviance information")
  expect_true(any(grepl("Iterations -", trace_output)), info = "Trace output should contain iteration information")
})

test_that("No valid set of coefficients", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  family <- gaussian()
  family$dev.resids <- function(y, mu, wt) {
    res <- (y - mu)^2
    res[1] <- Inf
    res
  }
  expect_error(
    savvy_glm.fit2(x, y, family = family, control = list(maxit = 5)),
    "No valid set of coefficients found for any fitting function")
})

test_that("Algorithm stopped at boundary value", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  family <- gaussian()
  family$linkinv <- function(eta) {
    eta[eta > 1] <- 1e10  # Force extreme eta values to trigger boundary
    eta
  }

  warnings <- capture_warnings({
    savvy_glm.fit2(x, y, family = family, control = list(maxit = 5))
  })
  found_truncation <- any(grepl("step size truncated", warnings))
  expect_true(found_truncation, info = "Warning about step size truncation expected")

  expect_true(any(grepl("inner loop", warnings)), info = "Warning about inner loop expected")
})

test_that("All fragile link functions for Binomial family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)

  links <- c("logit", "probit", "cauchit", "cloglog")

  for (l in links) {
    fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, model_class = "GSR", family = binomial(link = l), use_robust_start = TRUE))
    expect_true(is.numeric(fit$coefficients), info = paste("Coefficients should be numeric for", l, "link"))
    expect_true(fit$converged, info = paste("Model should converge for", l, "link"))
  }
})

test_that("All fragile link functions for Poisson family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rpois(n, lambda = 5)
  links <- c("log", "sqrt", "identity")

  for (l in links) {
    fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, model_class = "LW", family = poisson(link = l), use_robust_start = TRUE))
    expect_true(is.numeric(fit$coefficients), info = paste("Coefficients should be numeric for", l, "link"))
  }
})

test_that("All fragile link functions for Gaussian family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- exp(rnorm(n, mean = 1, sd = 0.5))
  links <- c("identity", "log", "inverse")

  for (l in links) {
    fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, family = gaussian(link = l), use_robust_start = TRUE))
    expect_true(is.numeric(fit$coefficients), info = paste("Coefficients should be numeric for", l, "link"))
  }
})

test_that("All fragile link functions for Gamma family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(abs(rnorm(n * p)), n, p)
  true_beta <- c(1, rep(0.1, p))
  eta <- drop(cbind(1, x) %*% true_beta)
  links <- c("inverse", "identity", "log")

  for (l in links) {
    if (l == "inverse") mu <- 1 / eta
    else if (l == "identity") mu <- eta
    else if (l == "log") mu <- exp(eta)
    y <- rgamma(n, shape = 2, rate = 2 / mu)

    fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, model_class = "St", family = Gamma(link = l), use_robust_start = TRUE))
    expect_true(is.numeric(fit$coefficients), info = paste("Coefficients should be numeric for", l, "link"))
  }
})

test_that("All fragile link functions for Inverse Gaussian family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(abs(rnorm(n * p)), n, p)
  true_beta <- c(1, rep(0.1, p))
  eta <- drop(cbind(1, x) %*% true_beta)

  links <- c("1/mu^2", "inverse", "identity", "log")

  for (l in links) {
    if (l == "1/mu^2") mu <- 1 / sqrt(eta)
    else if (l == "inverse") mu <- 1 / eta
    else if (l == "log") mu <- exp(eta)
    else if (l == "identity") mu <- eta

    y <- abs(mu + rnorm(n, 0, 0.05 * mean(mu))) + 0.01
    fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, model_class = "St", family = inverse.gaussian(link = l), use_robust_start = TRUE))
    expect_true(is.numeric(fit$coefficients), info = paste("Coefficients should be numeric for", l, "link"))
  }
})

test_that("Custom Power link function handling via Quasi family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(abs(rnorm(n * p)), n, p)
  y <- rpois(n, lambda = 5) + 1

  custom_family <- quasi(link = stats::power(-2), variance = "mu")
  fit <- suppressWarnings(savvy_glm.fit2(cbind(1, x), y, model_class = "DSh", family = custom_family, use_robust_start = TRUE))
  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric for custom power(-2) link")
})
