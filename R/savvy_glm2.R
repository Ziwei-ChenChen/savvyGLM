utils::globalVariables("n", add = TRUE)

#' @title Generalized Linear Models with Slab and Shrinkage Estimators
#'
#' @description
#' \code{savvy_glm2} extends the classical \code{glm2} function from the \pkg{glm2} package by embedding a set of shrinkage-based methods within the iteratively reweighted least squares (IRLS) algorithm.
#' These shrinkage methods (implemented via \code{savvy_glm.fit2}) are designed to improve convergence and estimation accuracy.
#' The user can specify one or more methods through the \code{model_class} argument. When multiple methods are provided (default is \code{c("St", "DSh", "SR", "GSR", "LW", "QIS")}),
#' the function can evaluate them in parallel (controlled by the \code{use_parallel} argument) and selects the final model based on the lowest AIC.
#'
#' @usage savvy_glm2(formula, family = gaussian, data, weights,
#'                  model_class = c("St", "DSh", "SR", "GSR", "LW", "QIS", "Sh"), subset,
#'                  na.action, start = NULL, etastart, mustart, offset,
#'                  control = list(...), model = TRUE,
#'                  method = "savvy_glm.fit2", x = FALSE, y = TRUE,
#'                  contrasts = NULL, use_parallel = FALSE, use_robust_start = FALSE, ...)
#'
#' @param formula An object of class "formula": a symbolic description of the model to be fitted. As for \code{\link[glm2]{glm2}}.
#' @param family A description of the error distribution and link function to be used in the model. As for \code{\link[glm2]{glm2}}.
#' @param data An optional data frame, list or environment containing the variables in the model. As for \code{\link[glm2]{glm2}}.
#' @param weights An optional vector of weights to be used in the fitting process. As for \code{\link[glm2]{glm2}}.
#' @param model_class A character vector specifying the shrinkage method(s) to be used in the underlying fitter \code{savvy_glm.fit2}.
#' Allowed values are \code{"St"}, \code{"DSh"}, \code{"SR"}, \code{"GSR"}, \code{"LW"}, \code{"QIS"}, and \code{"Sh"}. If a single value is provided, only that method is executed.
#' If multiple values are provided, the specified methods are evaluated (in parallel if \code{use_parallel = TRUE}) and the one with the lowest AIC is returned.
#' By default, the value is \code{c("St", "DSh", "SR", "GSR", "LW", "QIS")}; note that the \code{"Sh"} method is considered only if explicitly included.
#' @param subset An optional vector specifying a subset of observations to be used in the fitting process. As for \code{\link[glm2]{glm2}}.
#' @param na.action A function which indicates what should happen when the data contain NAs. As for \code{\link[glm2]{glm2}}.
#' @param start Starting values for the parameters in the linear predictor. As for \code{\link[glm2]{glm2}}. If \code{NULL} and \code{use_robust_start = TRUE}, robust optimization-based starting values may be calculated automatically for fragile link functions (see Details).
#' @param etastart Starting values for the linear predictor. As for \code{\link[glm2]{glm2}}.
#' @param mustart Starting values for the vector of means. As for \code{\link[glm2]{glm2}}.
#' @param offset An optional vector specifying a priori known component to be included in the linear predictor during fitting. As for \code{\link[glm2]{glm2}}.
#' @param control A list of control parameters to pass to the iterative fitting process. As for \code{\link[glm2]{glm2}}.
#' @param model A logical value indicating whether the model frame should be included as a component of the returned value. As for \code{\link[glm2]{glm2}}.
#' @param method The method to be used in fitting the model. The default is \code{savvy_glm.fit2}. This method uses IRLS
#' with custom optimization methods to ensure better convergence by evaluating different fitting methods and selecting the best one based on AIC.
#' As in \code{\link[glm2]{glm2}}, the alternative method "model.frame" returns the model frame and does no fitting.
#' @param x A logical value indicating whether the model matrix used in the fitting process should be returned as a component of the returned value. As for \code{\link[glm2]{glm2}}.
#' @param y A logical value indicating whether the response vector used in the fitting process should be returned as a component of the returned value. As for \code{\link[glm2]{glm2}}.
#' @param contrasts An optional list. See the contrasts.arg of \code{model.matrix.default}. As for \code{\link[glm2]{glm2}}.
#' @param use_parallel A logical value specifying whether to evaluate multiple shrinkage methods in parallel.
#' Defaults to \code{FALSE}. Setting this to \code{FALSE} forces sequential evaluation.
#' @param use_robust_start Logical. If \code{TRUE}, uses an optimization-based approach (via the \pkg{CVXR} package) to calculate robust starting values for fragile link functions (e.g., "log", "sqrt"). Defaults to \code{FALSE} to save computational time, as standard initialization works well for most typical datasets.
#' @param ... Additional arguments to be passed to the low level regression fitting functions. As for \code{\link[glm2]{glm2}}.
#'
#' @details
#' \code{savvy_glm2} improves upon the standard Generalized Linear Model (GLM) fitting process by incorporating
#' shrinkage estimator functions (\code{St_ost}, \code{DSh_ost}, \code{SR_ost}, \code{GSR_ost}, \code{LW_est}, \code{QIS_est}, and optionally \code{Sh_ost})
#' within the IRLS algorithm. The function begins with initial parameter estimates and iteratively updates the coefficients using the
#' specified shrinkage methods. When multiple methods are specified in \code{model_class}, they may be evaluated in parallel (if
#' \code{use_parallel = TRUE}), and the final model is selected based on the lowest AIC. In cases where any candidate model returns an \code{NA}
#' or non-finite AIC (such as when using quasi-likelihood families), the deviance is used uniformly as the selection criterion.
#'
#' \strong{Robust Starting Values:}
#' In situations where the user does not provide \code{start}, \code{etastart}, or \code{mustart}, and the chosen \code{family} utilizes specific link functions
#' (such as \code{"log"}, \code{"sqrt"}, \code{"inverse"}, \code{"1/mu^2"}, or \code{"logit"}), standard GLM initialization can sometimes lead to infinite deviance or divergence.
#' If \code{use_robust_start = TRUE}, the underlying \code{savvy_glm.fit2} automatically employs a data-driven optimization approach using the \pkg{CVXR} package to calculate stable starting coefficients.
#'
#' @return The value returned by \code{savvy_glm2} has exactly the same structure as that returned by \code{glm2}, except for:
#' \item{method}{the name of the fitter function used, which by default is \code{savvy_glm.fit2}.}
#' \item{chosen_fit}{the name of the chosen fitting method based on AIC.}
#'
#' @author Ziwei Chen, Vali Asimit and Claudio Senatore\cr
#' Maintainer: Ziwei Chen <Ziwei.Chen.3@citystgeorges.ac.uk>
#'
#' @references
#' Marschner, I. C. (2011). \emph{glm2: Fitting Generalized Linear Models with Convergence Problems}.
#' The R Journal, 3(2), 12–15.
#'
#' Asimit, V., Avramescu, O., Chen, Z., Rivas, D., & Senatore, C. (2026). \emph{GLM Solutions via Shrinkage}.
#'
#' Ledoit, O. and Wolf, M. (2004b). \emph{A well-conditioned estimator for large-dimensional covariance matrices}.
#' Journal of Multivariate Analysis, 88(2):365–411.
#'
#' Ledoit, O. and Wolf, M. (2022). \emph{Quadratic shrinkage for large covariance matrices}.
#' Bernoulli, 28(3): 1519-1547.
#'
#' @importFrom stats model.frame model.matrix model.response model.weights model.offset
#' @importFrom stats .getXlevels coef is.empty.model model.extract
#' @importFrom MASS ginv
#' @importFrom expm "%^%"
#' @importFrom utils globalVariables
#' @importFrom glmnet cv.glmnet
#' @importFrom glm2 glm2 glm.fit2
#' @importFrom parallel mclapply
#'
#' @seealso
#' \code{\link[stats]{glm}}, \code{\link[glm2]{glm2}}, \code{\link{savvy_glm.fit2}}
#'
#' @examples
#' set.seed(123)
#' n <- 200
#' p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' colnames(X) <- paste0("X", 1:p)
#'
#' linear_predictor <- X %*% c(0.5, -0.5, 0.2, 0, 0)
#' prob <- 1 / (1 + exp(-linear_predictor))
#' y <- rbinom(n, 1, prob)
#'
#' df <- data.frame(y = y, X)
#' fit1 <- savvy_glm2(y ~ ., data = df,
#'                    family = binomial(link = "logit"),
#'                    model_class = c("St", "DSh", "LW"),
#'                    use_parallel = FALSE)
#' print(fit1$chosen_fit)
#' print(coef(fit1))
#'
#' eta <- abs(X %*% c(0.5, -0.2, 0.1, 0, 0)) + 1
#' mu_quad <- eta^2
#' y_quad <- rpois(n, lambda = mu_quad)
#' df_quad <- data.frame(y = y_quad, X)
#'
#' fit2 <- savvy_glm2(y ~ ., data = df_quad,
#'                    family = poisson(link = "sqrt"),
#'                    model_class = c("SR", "St", "QIS"),
#'                    use_robust_start = TRUE,
#'                    use_parallel = FALSE)
#' print(fit2$chosen_fit)
#' print(coef(fit2))
#'
#' @export
savvy_glm2 <- function(formula, family = gaussian, data, weights,
                       model_class = c("St", "DSh", "SR", "GSR", "LW", "QIS", "Sh"), subset,
                       na.action, start = NULL, etastart, mustart, offset,
                       control = list(...), model = TRUE,
                       method = "savvy_glm.fit2", x = FALSE, y = TRUE,
                       contrasts = NULL, use_parallel = FALSE, use_robust_start = FALSE, ...) {
  call <- match.call()
  if (is.character(family))
    family <- get(family, mode = "function", envir = parent.frame())
  if (is.function(family))
    family <- family()
  if (is.null(family$family)) {
    print(family)
    stop("'family' not recognized")
  }
  if (missing(data))
    data <- environment(formula)
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "weights", "na.action",
               "etastart", "mustart", "offset"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  if (identical(method, "model.frame"))
    return(mf)
  if (!is.character(method) && !is.function(method))
    stop("invalid 'method' argument")
  if (identical(method, "savvy_glm.fit2"))
    control <- do.call("glm.control", control)
  mt <- attr(mf, "terms")
  Y <- model.response(mf, "any")
  if (length(dim(Y)) == 1L) {
    nm <- rownames(Y)
    dim(Y) <- NULL
    if (!is.null(nm))
      names(Y) <- nm
  }
  X <- if (!is.empty.model(mt))
    model.matrix(mt, mf, contrasts)
  else matrix(, NROW(Y), 0L)
  weights <- as.vector(model.weights(mf))
  if (!is.null(weights) && !is.numeric(weights))
    stop("'weights' must be a numeric vector")
  if (!is.null(weights) && any(weights < 0))
    stop("negative weights not allowed")
  offset <- as.vector(model.offset(mf))
  if (!is.null(offset)) {
    if (length(offset) != NROW(Y))
      stop(gettextf("number of offsets is %d should equal %d (number of observations)",
                    length(offset), NROW(Y)), domain = NA)
  }
  mustart <- model.extract(mf, "mustart")
  etastart <- model.extract(mf, "etastart")
  R.maj <- as.numeric(R.version$major)
  R.min <- as.numeric(unlist(strsplit(R.version$minor, ".", TRUE))[1])
  if (R.maj > 3 | (R.maj == 3 & R.min >= 5)) {
    fit <- eval(call(if (is.function(method)) "method" else method,
                     x = X, y = Y, weights = weights,
                     model_class = model_class, start = start,
                     etastart = etastart, mustart = mustart, offset = offset,
                     family = family, control = control,
                     intercept = attr(mt, "intercept") > 0L,
                     use_parallel = use_parallel,
                     use_robust_start = use_robust_start))
  }
  if (length(offset) && attr(mt, "intercept") > 0L) {
    fit2 <- eval(call(if (is.function(method)) "method" else method,
                      x = X[, "(Intercept)", drop = FALSE], y = Y, weights = weights,
                      model_class = model_class, offset = offset,
                      family = family, control = control,
                      intercept = TRUE,
                      use_parallel = use_parallel,
                      use_robust_start = use_robust_start))
    if (!fit2$converged)
      warning("fitting to calculate the null deviance did not converge -- increase maxit?")
    fit$null.deviance <- fit2$deviance
  }
  if (model)
    fit$model <- mf
  fit$na.action <- attr(mf, "na.action")
  if (x)
    fit$x <- X
  if (!y)
    fit$y <- NULL
  fit <- c(fit, list(call = call, formula = formula, terms = mt,
                     data = data, offset = offset, control = control, method = method,
                     contrasts = attr(X, "contrasts"), xlevels = .getXlevels(mt, mf)))
  class(fit) <- c(fit$class, c("glm", "lm"))
  fit
}

