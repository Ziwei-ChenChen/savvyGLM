utils::globalVariables("n", add = TRUE)

#' @title Generalized Linear Models Fitting with Slab and Shrinkage Estimators
#'
#' @description \code{savvy_glm.fit2} is a modified version of \code{glm.fit} in the stats package,
#' incorporating custom optimization functions (\code{St_ost}, \code{DSh_ost}, \code{SR_ost}, \code{GSR_ost}, and \code{Sh_ost}).
#' The user can specify a preferred shrinkage model via the \code{model_class} argument.
#' When the user does not explicitly supply a value for \code{model_class}, the function defaults to running only \code{"St"}, \code{"DSh"}, \code{"SR"}, and \code{"GSR"}.
#' If the user explicitly includes \code{"Sh"} (e.g. \code{model_class = c("St", "DSh", "SR", "GSR", "Sh")}), then \code{Sh_ost} is also evaluated.
#'
#' @usage savvy_glm.fit2(x, y, weights = rep(1, nobs),
#'                         model_class = c("St", "DSh", "SR", "GSR", "Sh"),
#'                         start = NULL, etastart = NULL, mustart = NULL,
#'                         offset = rep(0, nobs), family = gaussian(),
#'                         control = list(), intercept = TRUE,
#'                         use_parallel = TRUE)
#'
#' @param x A numeric matrix of predictors. As for \code{\link{glm.fit}}.
#' @param y A numeric vector of responses. As for \code{\link{glm.fit}}.
#' @param weights An optional vector of weights to be used in the fitting process. As for \code{\link{glm.fit}}.
#' @param model_class A character vector specifying the shrinkage model(s) to be used. Allowed values are \code{"St"}, \code{"DSh"}, \code{"SR"}, \code{"GSR"}, and \code{"Sh"}.
#' If a single value is provided, only that method is run. If multiple values are provided, the function runs the specified methods in parallel and returns the best one based on AIC.
#' When the user does not explicitly supply a value for \code{model_class}, the default is \code{c("St", "DSh", "SR", "GSR")} (i.e. \code{"Sh"} is not considered).
#' @param start Starting values for the parameters. As for \code{\link{glm.fit}}.
#' @param etastart Starting values for the linear predictor. As for \code{\link{glm.fit}}.
#' @param mustart Starting values for the mean. As for \code{\link{glm.fit}}.
#' @param offset An optional offset to be included in the model. As for \code{\link{glm.fit}}.
#' @param family A description of the error distribution and link function to be used in the model. As for \code{\link{glm.fit}}.
#' @param control A list of parameters for controlling the fitting process. As for \code{\link{glm.fit}}.
#' @param intercept A logical value indicating whether an intercept should be included in the model. As for \code{\link{glm.fit}}.
#' @param use_parallel Logical. If TRUE, enables parallel execution of the fitting process.
#'   Defaults to TRUE. Set to FALSE for serial execution.
#'
#' @details
#' \code{savvy_glm.fit2} extends \code{glm.fit} by using custom optimization functions to improve the convergence properties of the iteratively reweighted least squares (IRLS) algorithm.
#' The user may choose to run a specific shrinkage method by supplying a single value to the \code{model_class} argument.
#' If multiple values are provided (or if the default is used), the function evaluates a set of custom optimization methods and selects the final model based on the Akaike Information Criterion (AIC).
#' By default, the function considers only \code{St_ost}, \code{DSh_ost}, \code{SR_ost}, and \code{GSR_ost} (i.e. \code{model_class = c("St", "DSh", "SR", "GSR")}).
#' If the user explicitly includes \code{"Sh"} in \code{model_class} (for example, \code{model_class = c("St", "DSh", "SR", "GSR", "Sh")}), then the method \code{Sh_ost} is also evaluated.
#' The fitting process starts with initial parameter values and iterates through the IRLS algorithm. In each iteration, the coefficients are computed using the specified custom methods.
#' The method with the lowest AIC is chosen as the final model, ensuring that the model converges to the best solution given the data and the specified family.
#'
#' \strong{Custom Optimization Methods:}
#' \itemize{
#'   \item{\strong{Stein Estimator (St):}}{
#'     \itemize{
#'       \item{Applies a global multiplicative shrinkage factor to all coefficients.}
#'       \item{The factor is chosen to reduce MSE based on the overall signal-to-noise ratio.}
#'       \item{Simple and fast; works well when all coefficients can be shrunk similarly.}
#'     }
#'   }
#'   \item{\strong{Diagonal Shrinkage (DSh):}}{
#'     \itemize{
#'       \item{Extends the St approach by assigning a separate shrinkage factor to each coefficient.}
#'       \item{Each factor is computed from the coefficient size and its associated variance.}
#'       \item{More flexible than St, especially when coefficients vary in scale or importance.}
#'     }
#'   }
#'   \item{\strong{Generalized Slab Regression (GSR):}}{
#'     \itemize{
#'       \item{Extends SR by shrinking along multiple directions based on the data.}
#'       \item{These directions are usually chosen as leading principal components of the design matrix.}
#'       \item{Provides adaptive regularization in settings with collinearity or factor structure.}
#'     }
#'   }
#'   \item{\strong{Shrinkage Estimator (Sh):}}{
#'     \itemize{
#'       \item{Uses a non-diagonal shrinkage matrix derived from solving a Sylvester equation.}
#'       \item{Applies shrinkage by transforming the OLS estimator through a matrix that minimizes the MSE.}
#'       \item{Included only when \code{"Sh"} is specified in the \code{model_class} argument.}
#'     }
#'   }
#' }
#'
#' @return
#' The value returned by \code{savvy_glm.fit2} has the same structure as the value returned by \code{glm.fit}. It includes the following components:
#' \item{coefficients}{the estimated coefficients.}
#' \item{residuals}{the working residuals, that is the residuals in the final iteration of the IWLS fit.}
#' \item{fitted.values}{the fitted mean values, obtained by transforming the linear predictors by the inverse of the link function.}
#' \item{R}{the upper-triangular factor of the QR decomposition of the weighted model matrix.}
#' \item{rank}{the numeric rank of the fitted linear model.}
#' \item{qr}{the QR decomposition of the weighted model matrix.}
#' \item{family}{the family object used.}
#' \item{linear.predictors}{the final linear predictors.}
#' \item{deviance}{the deviance of the final model.}
#' \item{aic}{the AIC of the final model.}
#' \item{null.deviance}{the deviance for the null model.}
#' \item{iter}{the number of iterations used.}
#' \item{weights}{the final weights used in the fitting process.}
#' \item{prior.weights}{the weights initially supplied.}
#' \item{df.residual}{the residual degrees of freedom.}
#' \item{df.null}{the residual degrees of freedom for the null model.}
#' \item{y}{the response vector used.}
#' \item{converged}{a logical value indicating whether the IRLS iterations converged.}
#' \item{boundary}{a logical value indicating whether the algorithm stopped at a boundary value.}
#' \item{time}{the time taken for the fitting process.}
#' \item{chosen_fit}{the name of the chosen fitting method based on AIC.}
#'
#' @author Ziwei Chen and Vali Asimit\cr
#' Maintainer: Ziwei Chen <ziwei.chen.3@bayes.city.ac.uk>
#'
#' @references
#' The custom optimization methods used in this function are designed to improve the convergence properties of the GLM fitting process.
#' Marschner, I.C. (2011) glm2: Fitting generalized linear models with convergence problems. The R Journal, Vol. 3/2, pp.12-15.
#'
#' @importFrom stats model.matrix model.response gaussian binomial poisson
#' @importFrom MASS ginv
#' @importFrom expm "%^%"
#' @importFrom glmnet cv.glmnet
#' @importFrom glm2 glm2 glm.fit2
#' @importFrom parallel mclapply makeCluster parLapply stopCluster detectCores
#' @importFrom Matrix rankMatrix Schur
#' @importFrom stats lm coef predict
#'
#' @seealso
#' \code{\link{glm.fit}}, \code{\link{glm}}, \code{\link{glm.fit2}}
#'
#' @keywords models regression
#'
#' @export
savvy_glm.fit2 <- function (x, y, weights = rep(1, nobs),
                             model_class = c("St", "DSh", "SR", "GSR", "Sh"),
                             start = NULL, etastart = NULL, mustart = NULL,
                             offset = rep(0, nobs), family = gaussian(), control = list(),
                             intercept = TRUE, use_parallel = TRUE) {

  start_time <- Sys.time()
  control <- do.call("glm.control", control)

  x <- as.matrix(x)
  xnames <- dimnames(x)[[2L]]
  ynames <- if (is.matrix(y)) rownames(y) else names(y)
  nobs <- NROW(y)
  nvars <- ncol(x)
  EMPTY <- nvars == 0
  if (is.null(weights)) weights <- rep.int(1, nobs)
  if (is.null(offset)) offset <- rep.int(0, nobs)
  variance <- family$variance
  linkinv <- family$linkinv
  if (!is.function(variance) || !is.function(linkinv))
    stop("'family' argument seems not to be a valid family object",
         call. = FALSE)
  dev.resids <- family$dev.resids
  aic <- family$aic
  mu.eta <- family$mu.eta
  unless.null <- function(x, if.null) if (is.null(x)) if.null else x
  valideta <- unless.null(family$valideta, function(eta) TRUE)
  validmu <- unless.null(family$validmu, function(mu) TRUE)
  if (is.null(mustart)) {
    eval(family$initialize)
  } else {
    mukeep <- mustart
    eval(family$initialize)
    mustart <- mukeep
  }
  chosen_fit <- NULL
  best_fit <- list()
  if (EMPTY) {
    eta <- rep.int(0, nobs) + offset
    if (!valideta(eta))
      stop("invalid linear predictor values in empty model",
           call. = FALSE)
    mu <- linkinv(eta)
    if (!validmu(mu))
      stop("invalid fitted means in empty model",
           call. = FALSE)
    dev <- sum(dev.resids(y, mu, weights))
    w <- ((weights * mu.eta(eta)^2) / variance(mu))^0.5
    residuals <- (y - mu) / mu.eta(eta)
    good <- rep(TRUE, length(residuals))
    boundary <- conv <- TRUE
    coef <- numeric()
    iter <- 0L
  } else {
    coefold <- NULL
    eta <- if (!is.null(etastart))
      etastart
    else if (!is.null(start))
      if (length(start) != nvars)
        stop(gettextf("length of 'start' should equal %d and correspond to initial coefs for %s",
                      nvars, paste(deparse(xnames), collapse = ", ")),
             domain = NA)
    else {
      coefold <- start
      offset + as.vector(if (NCOL(x) == 1L) x * start else x %*% start)
    }
    else family$linkfun(mustart)
    mu <- linkinv(eta)
    if (!(validmu(mu) && valideta(eta)))
      stop("cannot find valid starting values: please specify some",
           call. = FALSE)
    devold <- sum(dev.resids(y, mu, weights))
    boundary <- conv <- FALSE

    ##------------- THE Iteratively Reweighted L.S. iteration -----------
    if (missing(model_class)) {
      model_class <- c("St", "DSh", "SR", "GSR")
    }
    all_fit_functions <- list("St" = St_ost, "DSh" = DSh_ost, "SR" = SR_ost, "GSR" = GSR_ost, "Sh" = Sh_ost)
    model_class <- match.arg(model_class, choices = names(all_fit_functions), several.ok = TRUE)
    if (!("Sh" %in% model_class)) {
      all_fit_functions <- all_fit_functions[c("St", "DSh", "SR", "GSR")]
    }
    fit_functions <- all_fit_functions[model_class]
    fit_names <- names(fit_functions)

    initial_eta <- eta
    initial_mu <- mu
    initial_devold <- devold
    initial_coefold <- coefold
    initial_boundary <- boundary
    initial_conv <- conv

    fit_model_parallel <- function(fit_func, fit_name) {
      eta <- initial_eta
      mu <- initial_mu
      devold <- initial_devold
      coefold <- initial_coefold
      boundary <- initial_boundary
      conv <- initial_conv
      current_iter <- 0L
      trace_messages <- NULL
      warnings <- NULL
      errors <- NULL

      result <- tryCatch({
        for (iter in 1L:control$maxit) {
          current_iter <- iter
          good <- weights > 0
          varmu <- variance(mu)[good]
          if (any(is.na(varmu)))
            stop("NAs in V(mu)")
          if (any(varmu == 0))
            stop("0s in V(mu)")
          mu.eta.val <- mu.eta(eta)
          if (any(is.na(mu.eta.val[good])))
            stop("NAs in d(mu)/d(eta)")
          good <- (weights > 0) & (mu.eta.val != 0)
          if (all(!good)) {
            conv <- FALSE
            warnings <- c(warnings, sprintf("no observations informative at iteration %d", iter))
            break
          }
          z <- (eta - offset)[good] + (y - mu)[good] / mu.eta.val[good]
          w <- sqrt((weights[good] * mu.eta.val[good]^2) / variance(mu)[good])
          fit_coefficients <- fit_func(x[good, , drop = FALSE] * w, z * w)
          fit_coefficients[is.na(fit_coefficients)] <- 0

          X_good <- x[good, , drop = FALSE] * w
          QR_decomp <- qr(X_good)
          fit_rank <- QR_decomp$rank
          R_matrix <- qr.R(QR_decomp)
          pivot_info <- QR_decomp$pivot

          if (any(!is.finite(fit_coefficients))) {
            conv <- FALSE
            warnings <- c(warnings, sprintf("non-finite coefficients at iteration %d", iter))
            break
          }
          if (nobs < fit_rank) {
            stop(gettextf("X matrix has rank %d, but only %d observations", fit_rank, nobs), domain = NA)
          }
          start <- fit_coefficients
          eta <- drop(x %*% start)
          mu <- linkinv(eta <- eta + offset)
          dev <- sum(dev.resids(y, mu, weights))

          if (control$trace) {
            trace_message <- sprintf("Deviance = %f Iterations - %d\n", dev, iter)
            trace_messages <- c(trace_messages, trace_message)
          }
          boundary <- FALSE
          if (!is.finite(dev)) {
            if (is.null(coefold))
              stop("no valid set of coefficients has been found: please supply starting values",
                   call. = FALSE)
            warnings <- c(warnings, "step size truncated due to divergence")
            ii <- 1
            while (!is.finite(dev)) {
              if (ii > control$maxit)
                stop("inner loop 1; cannot correct step size",
                     call. = FALSE)
              ii <- ii + 1
              start <- (start + coefold) / 2
              eta <- drop(x %*% start)
              mu <- linkinv(eta <- eta + offset)
              dev <- sum(dev.resids(y, mu, weights))
            }
            boundary <- TRUE
            if (control$trace) {
              trace_message <- sprintf("Step halved: new deviance = %f\n", dev)
              trace_messages <- c(trace_messages, trace_message)
            }
          }
          if (!(valideta(eta) && validmu(mu))) {
            if (is.null(coefold))
              stop("no valid set of coefficients has been found: please supply starting values",
                   call. = FALSE)
            warnings <- c(warnings, "step size truncated: out of bounds")
            ii <- 1
            while (!(valideta(eta) && validmu(mu))) {
              if (ii > control$maxit)
                stop("inner loop 2; cannot correct step size",
                     call. = FALSE)
              ii <- ii + 1
              start <- (start + coefold) / 2
              eta <- drop(x %*% start)
              mu <- linkinv(eta <- eta + offset)
            }
            boundary <- TRUE
            dev <- sum(dev.resids(y, mu, weights))
            if (control$trace) {
              trace_message <- sprintf("Step halved: new deviance = %f\n", dev)
              trace_messages <- c(trace_messages, trace_message)
            }
          }
          if (((dev - devold) / (0.1 + abs(dev)) >= control$epsilon) & (iter > 1)) {
            if (is.null(coefold))
              stop("no valid set of coefficients has been found: please supply starting values",
                   call. = FALSE)
            warnings <- c(warnings, "step size truncated due to increasing deviance")
            ii <- 1
            while ((dev - devold) / (0.1 + abs(dev)) > -control$epsilon) {
              if (ii > control$maxit) break
              ii <- ii + 1
              start <- (start + coefold) / 2
              eta <- drop(x %*% start)
              mu <- linkinv(eta <- eta + offset)
              dev <- sum(dev.resids(y, mu, weights))
            }
            if (ii > control$maxit) warnings <- c(warnings, "inner loop 3; cannot correct step size")
            else if (control$trace) {
              trace_message <- sprintf("Step halved: new deviance = %f\n", dev)
              trace_messages <- c(trace_messages, trace_message)
            }
          }
          if (abs(dev - devold) / (0.1 + abs(dev)) < control$epsilon) {
            conv <- TRUE
            coef <- start
            break
          } else {
            devold <- dev
            coef <- coefold <- start
          }
        }
        aic.model <- aic(y, nobs, mu, weights, dev) + 2 * fit_rank

        list(aic = aic.model,
             coef = coef,
             mu = mu,
             dev = dev,
             eta = eta,
             conv = conv,
             boundary = boundary,
             rank = fit_rank,
             family = family,
             QR_decomp = QR_decomp,
             R_matrix = R_matrix,
             pivot_info = pivot_info,
             residuals = (y - mu) / mu.eta(eta),
             good = good,
             weights = weights,
             w = w,
             y = y,
             iter = current_iter,
             trace = trace_messages,
             warnings = warnings,
             errors = errors,
             fit_name = fit_name)
      }, error = function(e) {
        errors <- c(errors, e$message)
        list(aic = Inf, errors = errors)
      })
      result
    }
    max_cores <- as.integer(Sys.getenv("_R_CHECK_LIMIT_CORES_", parallel::detectCores()))
    num_cores <- max(1, max_cores - 4)
    if (identical(Sys.getenv("_R_CHECK_PACKAGE_NAME_"), "savvyGLM")) {
      use_parallel <- FALSE
    }
    if (use_parallel) {
      if (.Platform$OS.type == "windows") {
        cl <- parallel::makeCluster(num_cores)
        parallel::clusterExport(cl,
                                varlist = c("fit_model_parallel", "initial_eta", "initial_mu", "initial_devold",
                                            "initial_coefold", "initial_boundary", "initial_conv",
                                            "x", "y", "weights", "offset", "family", "variance", "linkinv",
                                            "mu.eta", "dev.resids", "aic", "control", "fit_functions", "fit_names"),
                                envir = environment())
        results <- parallel::parLapply(cl, seq_along(fit_functions), function(i) {
          fit_model_parallel(fit_functions[[i]], fit_names[i])
        })
        parallel::stopCluster(cl)
      } else {
        results <- parallel::mclapply(seq_along(fit_functions), function(i) {
          fit_model_parallel(fit_functions[[i]], fit_names[i])
        }, mc.cores = num_cores)
      }
    } else {
      results <- lapply(seq_along(fit_functions), function(i) {
        fit_model_parallel(fit_functions[[i]], fit_names[i])
      })
    }
    valid_results <- Filter(function(res) !is.na(res$aic) && is.finite(res$aic), results)

    if (length(valid_results) == 0) {
      combined_errors <- unlist(lapply(results, function(res) res$errors))
      stop("No valid set of coefficients found for any fitting function. Errors encountered:\n",
           paste(combined_errors, collapse = "\n"))
    }
    best_result <- valid_results[[which.min(sapply(valid_results, function(res) res$aic))]]

    best_fit <- list(coefficients = best_result$coef,
                     fitted.values = best_result$mu,
                     R = if (!EMPTY) best_result$R_matrix,
                     rank = best_result$rank,
                     qr = if (!EMPTY) structure(best_result$QR_decomp[c("qr", "rank", "qraux", "pivot")], class = "qr"),
                     family = best_result$family,
                     linear.predictors = best_result$eta,
                     deviance = best_result$dev,
                     iter = best_result$iter,
                     prior.weights = best_result$weights,
                     y = best_result$y,
                     converged = best_result$conv,
                     boundary = best_result$boundary,
                     pivot_info = best_result$pivot_info,
                     good = best_result$good,
                     w = best_result$w,
                     residuals = best_result$residuals,
                     chosen_fit = best_result$fit_name,
                     trace = best_result$trace,
                     warnings = best_result$warnings)

    if (!is.null(best_fit$trace)) {
      cat(best_fit$trace, sep = "")
    }
    if (!is.null(best_fit$warnings)) {
      warning_messages <- paste0(seq_along(best_fit$warnings), ": ", best_fit$warnings, collapse = "\n")
      warning(sprintf("In glm.fit2_%s(x, y, control = control_list, family = family_type) :\n%s",
                      best_fit$chosen_fit, warning_messages))
    }
    coef <- best_fit$coefficients
    mu <- best_fit$fitted.values
    R_matrix <- best_fit$R
    fit_rank <- best_fit$rank
    QR_decomp <- best_fit$qr
    dev <- best_fit$deviance
    eta <- best_fit$linear.predictors
    conv <- best_fit$converged
    boundary <- best_fit$boundary
    pivot_info <- best_fit$pivot_info
    good <- best_fit$good
    weights <- best_fit$prior.weights
    w <- best_fit$w
    residuals <- best_fit$residuals
    chosen_fit <- best_fit$chosen_fit

    eps <- 10 * .Machine$double.eps
    if (family$family == "binomial") {
      if (any(mu > 1 - eps) || any(mu < eps))
        warning("glm.fit2: fitted probabilities numerically 0 or 1 occurred")
    }
    if (family$family == "poisson") {
      if (any(mu < eps))
        warning("glm.fit2: fitted rates numerically 0 occurred")
    }
    if (fit_rank < nvars) coef[pivot_info][seq.int(fit_rank + 1, nvars)] <- NA
    xxnames <- xnames[pivot_info]
    R_matrix <- as.matrix(R_matrix)
    nr <- min(sum(good), nvars)
    if (nr < nvars) {
      Rmat <- diag(nvars)
      Rmat[1L:nr, 1L:nvars] <- R_matrix[1L:nr, 1L:nvars]
    } else {
      Rmat <- R_matrix[1L:nvars, 1L:nvars]
    }
    Rmat <- as.matrix(Rmat)
    Rmat[row(Rmat) > col(Rmat)] <- 0
    names(coef) <- xnames
    colnames(R_matrix) <- xxnames
    dimnames(Rmat) <- list(xxnames, xxnames)
  }
  ##-------------- end IRLS iteration -------------------------------

  end_time <- Sys.time()
  elapsed_time <- end_time - start_time
  best_fit$time <- elapsed_time

  if (is.null(best_fit)) {
    stop("No valid set of coefficients found for any fitting function")
  }
  names(residuals) <- ynames
  names(mu) <- ynames
  names(eta) <- ynames
  wt <- rep.int(0, nobs)
  wt[good] <- w^2
  names(wt) <- ynames
  names(weights) <- ynames
  names(y) <- ynames

  wtdmu <- if (intercept) sum(weights * y) / sum(weights) else linkinv(offset)
  nulldev <- sum(dev.resids(y, wtdmu, weights))

  n.ok <- nobs - sum(weights == 0)
  nulldf <- n.ok - as.integer(intercept)
  rank <- if (EMPTY) 0 else fit_rank
  resdf <- n.ok - rank
  aic.model <- aic(y, nobs, mu, weights, dev) + 2 * rank

  list(coefficients = coef,
       residuals = residuals,
       fitted.values = mu,
       R = if (!EMPTY) Rmat,
       rank = rank,
       qr = if (!EMPTY) structure(QR_decomp[c("qr", "rank", "qraux", "pivot")], class = "qr"),
       family = family,
       linear.predictors = eta,
       deviance = dev,
       aic = aic.model,
       null.deviance = nulldev,
       iter = best_fit$iter,
       weights = wt,
       prior.weights = weights,
       df.residual = resdf,
       df.null = nulldf,
       y = y,
       converged = conv,
       boundary = boundary,
       time = elapsed_time,
       chosen_fit = chosen_fit)
}


