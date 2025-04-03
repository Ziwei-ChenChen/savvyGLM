
# Ridge Regression Estimation Function
RR_est <- function(x, y) {
  lambda_grid <- 10^seq(-6, 2, length = 50)
  cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = lambda_grid, intercept = FALSE)
  lambda_min <- cv_fit$lambda.min
  theta_rr <- as.vector(coef(cv_fit, s = "lambda.min")[-1, 1])
  fitted <- predict(cv_fit, newx = x, s = lambda_min)
  resid <- y - fitted
  sigma_rr <- sqrt(sum(resid^2) / (length(y) - ncol(x)))
  list(est = theta_rr, sigma = sigma_rr, optimal_lambda = lambda_min)
}

# OLS
OLS_est <- function(x, y) {
  if (qr(x)$rank < ncol(x)) return(RR_est(x, y))
  fit <- lm(y ~ x - 1)
  list(est = as.vector(coef(fit)), sigma = summary(fit)$sigma, optimal_lambda = 0)
}

# Sigma_lambda
Sigma_Lambda <- function(x, lambda) {
  crossprod(x) + lambda * diag(ncol(x))
}

# St Estimator
St_ost <- function(x, y) {
  ols <- OLS_est(x, y)
  est <- ols$est
  sigma2 <- ols$sigma^2
  lambda <- ols$optimal_lambda

  Sigma_inv <- MASS::ginv(Sigma_Lambda(x, lambda))
  M0 <- sigma2 * sum(diag(Sigma_inv))
  a_star <- sum(est^2) / (sum(est^2) + M0)

  as.vector(a_star * est)
}

# DSh Estimator
DSh_ost <- function(x, y) {
  ols <- OLS_est(x, y)
  est <- ols$est
  sigma2 <- ols$sigma^2
  lambda <- ols$optimal_lambda

  Sigma_inv <- MASS::ginv(Sigma_Lambda(x, lambda))
  b_k <- function(k) {
    beta_k2 <- est[k]^2
    beta_k2 / (beta_k2 + sigma2 * Sigma_inv[k, k])
  }

  b_vec <- sapply(seq_along(est), b_k)
  as.vector(b_vec * est)
}

# SR Estimator
SR_ost <- function(x, y) {
  ols <- OLS_est(x, y)
  est <- ols$est
  sigma2 <- ols$sigma^2
  lambda <- ols$optimal_lambda
  p <- length(est)

  Sigma_inv <- MASS::ginv(Sigma_Lambda(x, lambda))
  J <- matrix(1, p, p)
  u <- rep(1, p)

  al_u <- function(l) t(u) %*% (Sigma_inv %^% l) %*% u
  a0 <- sum(u^2)
  a1 <- al_u(1)
  a2 <- al_u(2)
  a3 <- al_u(3)
  delta <- sigma2 * (a0 * a3 - a1 * a2) + a3 * (t(est) %*% u)^2

  if (delta > 0) {
    mu <- (sigma2 * a2) / delta
    scalar <- as.numeric(mu / (1 + mu * a1))
    adj <- scalar * Sigma_inv %*% J
  } else {
    adj <- Sigma_inv %*% J
  }

  as.vector((diag(p) - adj) %*% est)
}

# GSR Estimator
GSR_ost <- function(x, y) {
  ols <- OLS_est(x, y)
  est <- ols$est
  sigma2 <- ols$sigma^2
  lambda <- ols$optimal_lambda
  p <- length(est)
  Sigma <- Sigma_Lambda(x, lambda)
  eig <- eigen(Sigma, symmetric = TRUE)
  U <- eig$vectors
  eigenvalues <- eig$values
  beta_proj <- as.vector(crossprod(U, est))
  mu_star <- sigma2 / (beta_proj^2)
  scalar_coeffs <- mu_star / eigenvalues / (1 + mu_star / eigenvalues)
  adjustment_matrix <- diag(1, p) - U %*% diag(scalar_coeffs) %*% t(U)
  est_GSR <- adjustment_matrix %*% est
  as.vector(est_GSR)
}

# Sylvester equation solver.
# Solves the Sylvester equation A * X + X * B = C using a Schur decomposition method.
# Adapted from: https://github.com/sakuramomo1005/biADMM/blob/master/R/sylvester.R
sylvester <- function(A, B, C, tol = 1e-4) {
  A1 <- Schur(A)
  Q1 <- A1$Q
  R1 <- A1$T

  A2 <- Schur(B)
  Q2 <- A2$Q
  R2 <- A2$T

  C_trans <- t(Q1) %*% C %*% Q2
  n <- nrow(R2)
  p <- ncol(R1)
  X <- matrix(0, p, n)
  I <- diag(p)
  k <- 1

  while (k <= n) {
    if (k < n && abs(R2[k+1, k]) >= tol) {
      # Process 2x2 block.
      r11 <- R2[k, k]
      r12 <- R2[k, k+1]
      r21 <- R2[k+1, k]
      r22 <- R2[k+1, k+1]
      if (k == 1) {
        temp <- matrix(0, p, 2)
      } else {
        temp <- X[, 1:(k-1), drop = FALSE] %*% R2[1:(k-1), k:(k+1), drop = FALSE]
      }
      b_block <- C_trans[, k:(k+1), drop = FALSE] - temp
      A_mat <- R1 %*% R1 + (r11 + r22) * R1 + (r11 * r22 - r12 * r21) * I
      X[, k:(k+1)] <- ginv(A_mat) %*% b_block
      k <- k + 2
    } else {
      # Process single column.
      if (k == 1) {
        temp <- matrix(0, p, 1)
      } else {
        temp <- X[, 1:(k-1), drop = FALSE] %*% R2[1:(k-1), k, drop = FALSE]
      }
      b_col <- C_trans[, k, drop = FALSE] - temp
      X[, k] <- ginv(R1 + R2[k, k] * I) %*% b_col
      k <- k + 1
    }
  }
  Q1 %*% X %*% t(Q2)
}

# Main Sh function.
# Computes the Shrinkage (Sh) estimator by solving a Sylvester equation to obtain
# a targeted shrinkage solution.
# Parameters:
#   est              - Initial coefficient estimate vector.
#   sigma_square     - Variance of noise.
#   Sigma_lambda_inv - Inverse of the regularized covariance matrix.
# Returns:
#   A vector representing the Sh estimator.
Sh_ost <- function(x, y) {
  ols <- OLS_est(x, y)
  est <- ols$est
  sigma2 <- ols$sigma^2
  lambda <- ols$optimal_lambda
  Sigma_inv <- MASS::ginv(Sigma_Lambda(x, lambda))

  B <- est %*% t(est)
  C_star <- sylvester(Sigma_inv, B, B)
  as.vector(C_star %*% est)
}




