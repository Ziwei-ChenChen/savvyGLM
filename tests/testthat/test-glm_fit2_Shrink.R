library(testthat)
library(ShrinkageGLMs)

test_that("Logistic regression with intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  fit <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, model_class = c("DSh","SR"), family = binomial(link = "logit")))

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
  fit <- suppressWarnings(glm.fit2_Shrink(x, y, family = binomial(link = "logit"), intercept = FALSE))

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
  fit <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y,  model_class = "SR", family = gaussian()))

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
  fit <- suppressWarnings(glm.fit2_Shrink(x, y, family = gaussian(), intercept = FALSE))

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
  fit <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y,  model_class = c("SR", "Sh"), family = poisson()))

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
  x[, 2] <- x[, 1]  # Make the matrix rank-deficient
  fit <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, model_class = c("SR", "Sh", "GSR"), family = binomial(link = "logit")))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
})

test_that("All zeros in response variable", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rep(0, n)
  fit <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, family = binomial(link = "logit")))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
})

test_that("All ones in response variable", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rep(1, n)
  fit <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y,  model_class = "GSR", family = binomial(link = "logit")))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
})

test_that("Handling large datasets", {
  set.seed(123)
  n <- 10000
  p <- 50
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  fit <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, family = binomial(link = "logit")))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Different link functions for Gaussian family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  fit_identity <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, family = gaussian(link = "identity")))
  expect_true(is.numeric(fit_identity$coefficients), info = "Coefficients should be numeric for identity link")
  expect_true(fit_identity$converged, info = "Model should converge for identity link")

  fit_log <- suppressWarnings(try(glm.fit2_Shrink(cbind(1, x), y, family = gaussian(link = "log")), silent = TRUE))
  if (inherits(fit_log, "try-error")) {
    expect_error(glm.fit2_Shrink(cbind(1, x), y, family = gaussian(link = "log")),
                 "cannot find valid starting values: please specify some",
                 info = "Error expected for log link due to invalid starting values")
  } else {
    expect_true(is.numeric(fit_log$coefficients), info = "Coefficients should be numeric for log link")
    expect_true(fit_log$converged, info = "Model should converge for log link")
  }

  fit_inverse <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, family = gaussian(link = "inverse")))
  expect_true(is.numeric(fit_inverse$coefficients), info = "Coefficients should be numeric for inverse link")
  expect_true(fit_inverse$converged, info = "Model should converge for inverse link")
})

test_that("Different link functions for Poisson family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rpois(n, lambda = 2)

  fit_log <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, family = poisson(link = "log")))
  expect_true(is.numeric(fit_log$coefficients), info = "Coefficients should be numeric for log link")
  expect_true(fit_log$converged, info = "Model should converge for log link")

  fit_inverse <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, family = poisson(link = "inverse")))
  expect_true(is.numeric(fit_inverse$coefficients), info = "Coefficients should be numeric for inverse link")
  expect_true(fit_inverse$converged, info = "Model should converge for inverse link")

  fit_sqrt <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, family = poisson(link = "sqrt")))
  expect_true(is.numeric(fit_sqrt$coefficients), info = "Coefficients should be numeric for sqrt link")
  expect_true(fit_sqrt$converged, info = "Model should converge for sqrt link")
})


test_that("Offset handling", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rpois(n, lambda = 2)
  offset <- rep(0.5, n)
  fit <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, family = poisson(), offset = offset))

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
  fit <- suppressWarnings(glm.fit2_Shrink(cbind(1, x), y, family = binomial(link = "logit"), weights = weights))

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
  data <- data.frame(y, x)

  family <- binomial(link = "logit")
  family$valideta <- function(eta) all(eta > -1)
  family$validmu <- function(mu) all(mu < 1.5)

  expect_error(
    glm.fit2_Shrink(cbind(1, x), y, family = family),
    "cannot find valid starting values: please specify some"
  )
})

test_that("Invalid family object", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)

  expect_error(
    glm.fit2_Shrink(cbind(1, x), y, family = list()),
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
  fit <- suppressWarnings(glm.fit2_Shrink(x, y, mustart = mustart, family = gaussian()))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Handling of empty model", {
  n <- 100
  y <- rnorm(n)
  fit <- suppressWarnings(glm.fit2_Shrink(matrix(nrow = n, ncol = 0), y, family = gaussian()))

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

  fit <- suppressWarnings(glm.fit2_Shrink(matrix(, n, 0), y, family = family))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_equal(length(fit$coefficients), 0, info = "Coefficients should be empty for an empty model")
  expect_true(fit$converged, info = "Empty model should converge")
})

test_that("Invalid eta in empty model", {
  n <- 100
  y <- rnorm(n)

  family <- gaussian()
  family$valideta <- function(eta) all(eta > 0)

  expect_error(
    glm.fit2_Shrink(matrix(, n, 0), y, family = family),
    "invalid linear predictor values in empty model"
  )
})

test_that("Invalid mu in empty model", {
  n <- 100
  y <- rnorm(n)

  family <- gaussian()
  family$validmu <- function(mu) all(mu > 1)

  expect_error(
    glm.fit2_Shrink(matrix(, n, 0), y, family = family),
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
    glm.fit2_Shrink(x, y, family = family),
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
    glm.fit2_Shrink(x, y, family = family),
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
    glm.fit2_Shrink(x, y, family = family),
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
    suppressWarnings(glm.fit2_Shrink(x, y, family = family, control = control))
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

  # Custom family to force infinite deviance
  family <- gaussian()
  family$dev.resids <- function(y, mu, wt) {
    res <- (y - mu)^2
    res[1] <- Inf  # Introduce an infinite value
    res
  }

  # Capture errors
  expect_error(
    glm.fit2_Shrink(x, y, family = family, control = list(maxit = 5)),
    "no valid set of coefficients has been found: please supply starting values")
})

test_that("Algorithm stopped at boundary value", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  family <- gaussian()
  family$linkinv <- function(eta) {
    eta[eta > 1] <- 1e10  # Force extreme eta values
    eta
  }

  warnings <- capture_warnings({
    glm.fit2_Shrink(x, y, family = family, control = list(maxit = 5))
  })
  expect_true(any(grepl("step size truncated due to increasing deviance", warnings)), info = "Warning about step size truncated due to increasing deviance expected")
  expect_true(any(grepl("inner loop 3; cannot correct step size", warnings)), info = "Warning about inner loop 3 expected")
})
