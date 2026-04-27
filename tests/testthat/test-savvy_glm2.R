library(testthat)
library(savvyGLM)

test_that("Logistic regression with intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x1 <- matrix(rnorm(n * p), n, p)
  y1 <- rbinom(n, 1, prob = 0.5)
  data1 <- data.frame(y1, x1)
  fit1 <- suppressWarnings(savvy_glm2(y1 ~ ., family = binomial(link = "logit"), data = data1,  model_class = c("St", "LW")))

  expect_true(is.numeric(fit1$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit1$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit1$converged, info = "Model should converge")
})

test_that("Logistic regression without intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x2 <- matrix(rnorm(n * p), n, p)
  y2 <- rbinom(n, 1, prob = 0.5)
  data2 <- data.frame(y2, x2)
  fit2 <- suppressWarnings(savvy_glm2(y2 ~ . - 1, family = binomial(link = "logit"), data = data2, model_class = "QIS"))

  expect_true(is.numeric(fit2$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit2$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit2$converged, info = "Model should converge")
})

test_that("Gaussian family with intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x3 <- matrix(rnorm(n * p), n, p)
  y3 <- rnorm(n)
  data3 <- data.frame(y3, x3)
  fit3 <- suppressWarnings(savvy_glm2(y3 ~ ., family = gaussian(), data = data3, model_class = "GSR"))

  expect_true(is.numeric(fit3$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit3$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit3$converged, info = "Model should converge")
})

test_that("Poisson family with intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x4 <- matrix(rnorm(n * p), n, p)
  y4 <- rpois(n, lambda = 2)
  data4 <- data.frame(y4, x4)
  fit4 <- suppressWarnings(savvy_glm2(y4 ~ ., family = poisson(), data = data4, model_class = c("LW", "QIS")))

  expect_true(is.numeric(fit4$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit4$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit4$converged, info = "Model should converge")
})

test_that("Gaussian family without intercept", {
  set.seed(123)
  n <- 100
  p <- 5
  x5 <- matrix(rnorm(n * p), n, p)
  y5 <- rnorm(n)
  data5 <- data.frame(y5, x5)
  fit5 <- suppressWarnings(savvy_glm2(y5 ~ . - 1, family = gaussian(), data = data5))

  expect_true(is.numeric(fit5$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit5$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit5$converged, info = "Model should converge")
})

test_that("Handling missing data", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  x[sample(length(x), 10)] <- NA
  data <- data.frame(y, x)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, na.action = na.omit))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Different control parameters", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data,
                                     model_class = c("SR", "St"), control = glm.control(epsilon = 1e-8, maxit = 100)))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Large dataset", {
  set.seed(123)
  n <- 10000
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, model_class = c("LW", "SR")))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Small dataset", {
  set.seed(123)
  n <- 10
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, model_class = "QIS"))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Different link functions for Gaussian family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)

  y_id <- rnorm(n)
  fit_identity <- suppressWarnings(savvy_glm2(y_id ~ ., family = gaussian(link = "identity"), data = data.frame(y_id, x), model_class = "QIS"))
  expect_true(is.numeric(fit_identity$coefficients), info = "Coefficients should be numeric for identity link")
  expect_true(fit_identity$converged, info = "Model should converge for identity link")

  y_pos <- exp(rnorm(n, mean = 1, sd = 0.5))
  fit_log <- suppressWarnings(savvy_glm2(y_pos ~ ., family = gaussian(link = "log"), data = data.frame(y_pos, x), use_robust_start = TRUE))
  expect_true(is.numeric(fit_log$coefficients), info = "Coefficients should be numeric for log link")
  expect_true(fit_log$converged, info = "Model should converge for log link")

  fit_inverse <- suppressWarnings(savvy_glm2(y_pos ~ ., family = gaussian(link = "inverse"), data = data.frame(y_pos, x), use_robust_start = TRUE))
  expect_true(is.numeric(fit_inverse$coefficients), info = "Coefficients should be numeric for inverse link")
})

test_that("Different link functions for Binomial family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)

  links <- c("logit", "probit", "cauchit", "cloglog")
  for (l in links) {
    fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = l), data = data, model_class = "LW", use_robust_start = TRUE))
    expect_true(is.numeric(fit$coefficients), info = paste("Coefficients should be numeric for", l, "link"))
  }
})

test_that("Different link functions for Gamma family", {
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
    data <- data.frame(y, x)

    fit <- suppressWarnings(savvy_glm2(y ~ ., family = Gamma(link = l), data = data, model_class = "QIS", use_robust_start = TRUE))
    expect_true(is.numeric(fit$coefficients), info = paste("Coefficients should be numeric for", l, "link"))
  }
})

test_that("Different link functions for Inverse Gaussian family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(abs(rnorm(n * p)), n, p)
  true_beta <- c(1, rep(0.1, p))
  eta <- drop(cbind(1, x) %*% true_beta)

  links <- c("1/mu^2", "inverse", "log", "identity")
  for (l in links) {
    if (l == "1/mu^2") mu <- 1 / sqrt(eta)
    else if (l == "inverse") mu <- 1 / eta
    else if (l == "log") mu <- exp(eta)
    else if (l == "identity") mu <- eta

    y <- abs(mu + rnorm(n, 0, 0.05 * mean(mu))) + 0.01
    data <- data.frame(y, x)

    fit <- suppressWarnings(savvy_glm2(y ~ ., family = inverse.gaussian(link = l), data = data, model_class = "St", use_robust_start = TRUE))
    expect_true(is.numeric(fit$coefficients), info = paste("Coefficients should be numeric for", l, "link"))
  }
})

test_that("Custom Power link function handling via Quasi family", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(abs(rnorm(n * p)), n, p)
  y <- rpois(n, lambda = 5) + 1
  data <- data.frame(y, x)

  custom_family <- quasi(link = stats::power(-2), variance = "mu")
  fit <- suppressWarnings(savvy_glm2(y ~ ., data = data, model_class = "LW", family = custom_family, use_robust_start = TRUE))
  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric for custom power(-2) link")
})

test_that("Offset handling", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rpois(n, lambda = 2)
  offset <- rep(0.5, n)
  data <- data.frame(y, x)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = poisson(), data = data, offset = offset))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
  expect_true(all(fit$offset == offset), info = "Offset should be correctly handled")
})

test_that("Valideta and Validmu functions", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)

  family <- binomial(link = "logit")
  family$valideta <- function(eta) FALSE

  expect_error(
    savvy_glm2(y ~ ., family = family, data = data),
    "cannot find valid starting values"
  )
})

test_that("Initialization with mustart", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  data <- data.frame(y, x)

  mustart <- rep(mean(y), n)
  fit <- suppressWarnings(savvy_glm2(y ~ ., data = data, mustart = mustart, family = gaussian()))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Handling of empty model", {
  n <- 100
  y <- rnorm(n)
  data <- data.frame(y)
  fit <- suppressWarnings(savvy_glm2(y ~ 0, data = data, family = gaussian()))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_equal(length(fit$coefficients), 0, info = "Coefficients should be empty for an empty model")
  expect_true(fit$converged, info = "Empty model should converge")
})

test_that("Valideta and validmu functions in empty model", {
  n <- 100
  y <- rnorm(n)
  data <- data.frame(y)

  family <- gaussian()
  family$valideta <- function(eta) all(eta > -1)
  family$validmu <- function(mu) all(mu < 2)

  fit <- suppressWarnings(savvy_glm2(y ~ 0, data = data, family = family))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_equal(length(fit$coefficients), 0, info = "Coefficients should be empty for an empty model")
  expect_true(fit$converged, info = "Empty model should converge")
})

test_that("Invalid eta in empty model", {
  n <- 100
  y <- rnorm(n)
  data <- data.frame(y)

  family <- gaussian()
  family$valideta <- function(eta) FALSE

  expect_error(
    savvy_glm2(y ~ 0, data = data, family = family),
    "invalid linear predictor values in empty model"
  )
})

test_that("Invalid mu in empty model", {
  n <- 100
  y <- rnorm(n)
  data <- data.frame(y)

  family <- gaussian()
  family$validmu <- function(mu) FALSE

  expect_error(
    savvy_glm2(y ~ 0, data = data, family = family),
    "invalid fitted means in empty model"
  )
})

test_that("NA values in varmu", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  data <- data.frame(y, x)

  family <- gaussian()
  family$variance <- function(mu) {
    res <- rep(1, length(mu))
    res[1] <- NA
    res
  }

  expect_error(
    savvy_glm2(y ~ ., data = data, family = family),
    "NAs in V\\(mu\\)"
  )
})

test_that("Zero values in varmu", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  data <- data.frame(y, x)

  family <- gaussian()
  family$variance <- function(mu) {
    res <- rep(1, length(mu))
    res[1] <- 0
    res
  }

  expect_error(
    savvy_glm2(y ~ ., data = data, family = family),
    "0s in V\\(mu\\)"
  )
})

test_that("NA values in mu.eta.val", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  data <- data.frame(y, x)

  family <- gaussian()
  family$mu.eta <- function(eta) {
    res <- rep(1, length(eta))
    res[1] <- NA
    res
  }

  expect_error(
    savvy_glm2(y ~ ., data = data, family = family),
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
  data <- data.frame(y, x)

  family <- gaussian()
  family$valideta <- function(eta) TRUE
  family$validmu <- function(mu) TRUE
  control <- list(trace = TRUE, maxit = 1, epsilon = 1e-8)
  trace_output <- capture.output({
    suppressWarnings(savvy_glm2(y ~ ., data = data, family = family, control = control))
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
  data <- data.frame(y, x)

  family <- gaussian()
  family$dev.resids <- function(y, mu, wt) {
    res <- (y - mu)^2
    res[1] <- Inf
    res
  }
  expect_error(
    savvy_glm2(y ~ ., data = data, family = family, control = list(maxit = 5)),
    "No valid set of coefficients found for any fitting function")
})

test_that("Algorithm stopped at boundary value", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  data <- data.frame(y, x)
  family <- gaussian()
  family$linkinv <- function(eta) {
    eta[eta > 1] <- 1e10
    eta
  }

  warnings <- capture_warnings({
    savvy_glm2(y ~ ., data = data, family = family, control = list(maxit = 5))
  })
  found_truncation <- any(grepl("step size truncated", warnings))
  expect_true(found_truncation, info = "Warning about step size truncation expected")
})

test_that("Rank-deficient matrix handling", {
  set.seed(123)
  n <- 10
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  x[, 2] <- x[, 1]
  data <- data.frame(y, x)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
})

test_that("All zeros in response variable", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rep(0, n)
  data <- data.frame(y, x)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, model_class = "St"))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
})

test_that("All ones in response variable", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rep(1, n)
  data <- data.frame(y, x)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, model_class = "GSR"))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
})

test_that("Convergence with different starting values", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)
  start_values <- rep(0.5, p + 1)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, start = start_values))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Family argument as a string", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = "binomial", data = data))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(is.character(fit$chosen_fit), info = "Chosen fitting method should be character")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Invalid family argument string", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)

  expect_error(
    savvy_glm2(y ~ ., family = "invalid_family", data = data),
    "object 'invalid_family' of mode 'function' was not found"
  )
})

test_that("Using model.frame method", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)
  mf <- savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, method = "model.frame")
  expect_true(is.data.frame(mf), info = "model.frame should return a data frame")
  expect_equal(nrow(mf), n, info = "model.frame should return correct number of rows")
  expect_equal(ncol(mf), p + 1, info = "model.frame should return correct number of columns")
})

test_that("Handling response vector dimensions", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- matrix(rbinom(n, 1, prob = 0.5), n, 1)
  data <- data.frame(y, x)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data))

  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(fit$converged, info = "Model should converge")
})

test_that("Handling weights argument validation", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)

  weights <- runif(n, 0, 1)
  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, weights = weights))
  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(fit$converged, info = "Model should converge")

  weights <- runif(n, -1, 0)
  expect_error(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, weights = weights),
               "negative weights not allowed")

  weights <- rep("a", n)
  expect_error(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, weights = weights),
               "'weights' must be a numeric vector")
})

test_that("Offset length mismatch", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)

  offset <- runif(n + 1)
  expect_error(
    savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, offset = offset),
    "variable lengths differ \\(found for '\\(offset\\)'\\)"
  )
})

test_that("Invalid family object", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)

  invalid_family <- list(family = "invalid")
  expect_error(
    savvy_glm2(y ~ ., family = invalid_family, data = data),
    "'family' argument seems not to be a valid family object"
  )
})

test_that("Unrecognized family with NULL family argument and suppresses print output", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)

  expect_error(
    capture.output(suppressMessages(savvy_glm2(y ~ ., family = NULL, data = data))),
    "'family' not recognized"
  )
})

test_that("Exclude x and y from the fit object", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)

  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, x = FALSE, y = FALSE))
  expect_false(is.null(fit$x), info = "x matrix should not be included in the fit object")
  expect_true(is.null(fit$y), info = "y vector should not be included in the fit object")
})

test_that("Include x and y in the fit object", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)

  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, x = TRUE, y = TRUE))
  expect_true(!is.null(fit$x), info = "x matrix should be included in the fit object")
  expect_true(!is.null(fit$y), info = "y vector should be included in the fit object")
})

test_that("Non-convergence in null deviance fit", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)
  offset <- rep(10, n)

  captured_warnings <- list()
  fit <- withCallingHandlers(
    savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data, control = list(maxit = 1), offset = offset),
    warning = function(w) {
      captured_warnings <<- c(captured_warnings, list(w))
      invokeRestart("muffleWarning")
    }
  )

  warning_messages <- sapply(captured_warnings, function(w) w$message)
  null_deviance_warning <- any(grepl("fitting to calculate the null deviance did not converge -- increase maxit?", warning_messages))

  expect_true(null_deviance_warning, info = "Warning expected for non-convergence in null deviance fit")
  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
})

test_that("Handling response matrix Y", {
  set.seed(123)
  n <- 100
  p <- 5

  x <- matrix(rnorm(n * p), n, p)
  y <- matrix(rbinom(n, 1, prob = 0.5), n, 1)
  rownames(y) <- paste0("obs", 1:n)
  data <- data.frame(y = y[,1], x)

  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data))
  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(!is.null(fit$y), info = "Response variable should be included in the fit object")
  expect_true(all(names(fit$y) == paste0("obs", 1:n)), info = "Response variable names should be correctly set")

  y <- matrix(rbinom(n, 1, prob = 0.5), n, 1)
  rownames(y) <- NULL
  data <- data.frame(y = y[,1], x)

  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data))
  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(!is.null(fit$y), info = "Response variable should be included in the fit object")
  expect_true(is.null(rownames(fit$y)), info = "Response variable rownames should be NULL when not set")

  y <- rbinom(n, 1, prob = 0.5)
  data <- data.frame(y, x)

  fit <- suppressWarnings(savvy_glm2(y ~ ., family = binomial(link = "logit"), data = data))
  expect_true(is.numeric(fit$coefficients), info = "Coefficients should be numeric")
  expect_true(!is.null(fit$y), info = "Response variable should be included in the fit object")
})

test_that("savvy_glm2 handles invalid method argument", {
  set.seed(123)
  x <- matrix(rnorm(100), ncol = 2)
  y <- rpois(50, lambda = exp(x %*% c(0.5, -0.2)))
  data <- data.frame(y, x1 = x[,1], x2 = x[,2])

  expect_error(
    savvy_glm2(y ~ x1 + x2, family = poisson(), data = data, method = 123),
    "invalid 'method' argument"
  )
})

test_that("savvy_glm2 handles one-dimensional array response Y correctly", {
  set.seed(123)
  x <- matrix(rnorm(100), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- array(rpois(50, lambda = exp(x %*% c(0.5, -0.2))), dim = c(50))
  rownames_y <- as.character(1:length(y))

  y_matrix <- as.matrix(y)
  rownames(y_matrix) <- rownames_y
  dim(y_matrix) <- c(50, 1)

  data <- data.frame(y = y_matrix, x)
  fit <- suppressWarnings(savvy_glm2(y ~ x1 + x2, family = poisson(), data = data))

  expect_true(is.null(dim(fit$y)))
  expect_true(!is.null(names(fit$y)))
  expect_equal(names(fit$y), rownames_y)
  expect_true(!is.null(fit$coefficients))
})

test_that("savvy_glm2 processes valid method argument and control options", {
  set.seed(123)
  x <- matrix(rnorm(100), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- rpois(50, lambda = exp(x %*% c(0.5, -0.2)))
  data <- data.frame(y, x)

  fit <- suppressWarnings(savvy_glm2(y ~ x1 + x2, family = poisson(),
                                     data = data, model_class = "DSh", method = "savvy_glm.fit2"))

  expect_s3_class(fit, "glm")
  expect_equal(fit$method, "savvy_glm.fit2")
})

test_that("savvy_glm2 handles missing data argument correctly", {
  set.seed(123)
  x <- matrix(rnorm(100), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- rpois(50, lambda = exp(x %*% c(0.5, -0.2)))

  formula <- y ~ x1 + x2
  x1 <- x[, 1]
  x2 <- x[, 2]

  fit <- suppressWarnings(savvy_glm2(formula, family = poisson()))

  expect_s3_class(fit, "glm")
  expect_true(!is.null(fit$coefficients))
  expect_true("y" %in% names(fit$model))
  expect_true("x1" %in% names(fit$model))
  expect_true("x2" %in% names(fit$model))
})

