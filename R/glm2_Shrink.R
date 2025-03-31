#' @title Generalized Linear Models with Slab and Shrinkage Estimators
#'
#' @description \code{glm2_Shrink} is a modified version of \code{glm2} in the stats package, incorporating custom optimization methods.
#' This function aims to improve convergence properties by selecting the best fitting method based on the Akaike Information Criterion (AIC).
#' In addition, the user can choose to run a specific shrinkage method by providing a value to the \code{model_class} argument.
#' When a single value is provided, only that method is executed.
#' When multiple values are given (or when the default is used), the function evaluates the specified methods in parallel and returns the one with the lowest AIC.
#'
#' @usage glm2_Shrink(formula, family = gaussian, data, weights,
#'                model_class = c("St", "DSh", "SR", "GSR"), subset,
#'                na.action, start = NULL, etastart, mustart, offset,
#'                control = list(...), model = TRUE,
#'                method = "glm.fit2_Shrink", x = FALSE, y = TRUE,
#'                contrasts = NULL, ...)
#'
#' @param formula An object of class "formula": a symbolic description of the model to be fitted. As for \code{\link{glm}}.
#' @param family A description of the error distribution and link function to be used in the model. As for \code{\link{glm}}.
#' @param data An optional data frame, list or environment containing the variables in the model. As for \code{\link{glm}}.
#' @param weights An optional vector of weights to be used in the fitting process. As for \code{\link{glm}}.
#' @param model_class A character vector specifying the shrinkage model(s) to be used in the underlying fitter \code{glm.fit2_Shrink}.
#' Allowed values are \code{"St"}, \code{"DSh"}, \code{"SR"}, \code{"GSR"}, and \code{"Sh"}.
#' If a single value is provided, only that method is run.
#' If multiple values are provided (or if the default is used), the function evaluates the specified methods in parallel and returns the one with the lowest AIC.
#' When the user does not explicitly supply a value, the default is \code{c("St", "DSh", "SR", "GSR")}, and the additional method \code{"Sh"} is not considered.
#' @param subset An optional vector specifying a subset of observations to be used in the fitting process. As for \code{\link{glm}}.
#' @param na.action A function which indicates what should happen when the data contain NAs. As for \code{\link{glm}}.
#' @param start Starting values for the parameters in the linear predictor. As for \code{\link{glm}}.
#' @param etastart Starting values for the linear predictor. As for \code{\link{glm}}.
#' @param mustart Starting values for the vector of means. As for \code{\link{glm}}.
#' @param offset An optional vector specifying a priori known component to be included in the linear predictor during fitting. As for \code{\link{glm}}.
#' @param control A list of control parameters to pass to the iterative fitting process. As for \code{\link{glm}}.
#' @param model A logical value indicating whether the model frame should be included as a component of the returned value. As for \code{\link{glm}}.
#' @param method The method to be used in fitting the model. The default is \code{glm.fit2_Shrink}. This method uses iteratively reweighted least squares (IRLS)
#' with custom optimization methods to ensure better convergence by evaluating different fitting methods and selecting the best one based on AIC.
#' As in \code{\link{glm}}, the alternative method "model.frame" returns the model frame and does no fitting.
#' @param x A logical value indicating whether the model matrix used in the fitting process should be returned as a component of the returned value. As for \code{\link{glm}}.
#' @param y A logical value indicating whether the response vector used in the fitting process should be returned as a component of the returned value. As for \code{\link{glm}}.
#' @param contrasts An optional list. See the contrasts.arg of \code{model.matrix.default}. As for \code{\link{glm}}.
#' @param ... Additional arguments to be passed to the low level regression fitting functions. As for \code{\link{glm}}.
#'
#' \code{glm2_Shrink} extends the \code{glm2} function by using custom optimization methods to improve the convergence properties of the iteratively
#' reweighted least squares (IRLS) algorithm. The function evaluates four custom optimization methods: \code{St_ost}, \code{DSh_ost}, \code{SR_ost}, and \code{GSR_ost}.
#' The user can also include the additional method \code{Sh_ost} by explicitly specifying \code{"Sh"} in the \code{model_class} argument.
#' The fitting process starts with initial parameter values and iterates through the IRLS algorithm.
#' In each iteration, the coefficients are computed using the specified custom methods.
#' The final model is chosen as the one with the lowest AIC, ensuring that the model converges to the best possible solution given the data and specified family.
#'
#' @return The value returned by \code{glm2_Shrink} has exactly the same structure as that returned by \code{glm}, except for:
#' \item{method}{the name of the fitter function used, which by default is \code{glm.fit2_Shrink}.}
#' \item{chosen_fit}{the name of the chosen fitting method based on AIC.}
#'
#' @references
#' The custom optimization methods used in this function are designed to improve the convergence properties of the GLM fitting process.
#' Marschner, I.C. (2011) glm2: Fitting generalized linear models with convergence problems. The R Journal, Vol. 3/2, pp.12-15.
#' \code{glm2_Shrink} uses code from \code{glm2}, whose authors are listed in the help documentation for the \strong{stats} package.
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
#' @seealso \code{\link{glm}}, \code{\link{glm2}}, \code{\link{glm.fit2_Shrink}}
#'
#' @export
glm2_Shrink <- function(formula, family = gaussian, data, weights,
                        model_class = c("St", "DSh", "SR", "GSR"), subset,
                        na.action, start = NULL, etastart, mustart, offset,
                        control = list(...), model = TRUE,
                        method = "glm.fit2_Shrink", x = FALSE, y = TRUE,
                        contrasts = NULL, ...) {
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
  if (identical(method, "glm.fit2_Shrink"))
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
                     intercept = attr(mt, "intercept") > 0L))
  }
  if (length(offset) && attr(mt, "intercept") > 0L) {
    fit2 <- eval(call(if (is.function(method)) "method" else method,
                      x = X[, "(Intercept)", drop = FALSE], y = Y, weights = weights,
                      model_class = model_class, offset = offset,
                      family = family, control = control,
                      intercept = TRUE))
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
