library(data.table)
library(hal9001)
library(sl3)
library(causalHAL)
library(doFuture)
library(future)

#out <- do_sims(10, 3000, 2, TRUE, do_local_alt = FALSE)



do_sims <- function(niter, n, pos_const, muIsHard, do_local_alt = FALSE) {
  seed_init <- 12345
  sim_results <- rbindlist(lapply(1:niter, function(iter) {
    set.seed(seed_init*iter)
    print(paste0("Iteration number: ", iter))
    try({
      if(!do_local_alt) {
        data_list <- get_data(n, pos_const, muIsHard)
      } else if(do_local_alt) {
        data_list <- get_data_local_alt(n, pos_const, muIsHard)
      }
      return(as.data.table(get_estimates(data_list$W, data_list$A, data_list$Y,iter, NULL)))
    })
    return(data.table())
  }))
  key <- paste0("iter=", niter, "_n=", n, "_pos=", pos_const, "_hard=", muIsHard, "_local_",do_local_alt )
  try({fwrite(sim_results, paste0("~/causalHAL/simResults/sim_results_", key, ".csv"))})
  return(sim_results)
}


#' generates dataset of size n.
#' constant in propensity score can be used to vary overlap.
#' two settings for outcome regression: easy form and hard form
get_data <- function(n, pos_const, muIsHard = TRUE) {
  d <- 4
  W <- replicate(d, runif(n, -1, 1))
  colnames(W) <- paste0("W", 1:d)
  pi0 <- plogis(pos_const * ( W[,1] + sin(4*W[,1]) +   W[,2] + cos(4*W[,2]) + W[,3] + sin(4*W[,3]) + W[,4] + cos(4*W[,4]) ))
  print("pos")
  print(range(pi0))
  A <- rbinom(n, 1, pi0)
  if(muIsHard) {
    mu0 <-  sin(4*W[,1]) + sin(4*W[,2]) + sin(4*W[,3])+  sin(4*W[,4]) + cos(4*W[,2])
  } else {
    mu0 <-  W[,1] + abs(W[,2])  + W[,3] + abs(W[,4])
  }
  tau <- 1 + W[,1] + abs(W[,2]) + cos(4*W[,3]) + W[,4]
  Y <- rnorm(n,  mu0 + A * tau, 0.5)
  return(list(W=W, A = A, Y = Y, ATE = 1.31, pi = pi0))
}

get_data_local_alt <- function(n, pos_const, muIsHard = TRUE) {
  ates <- list("0" = 4, "0.5" = 5.219036, "1"= 14.44244, "2"= 1048.994)

  d <- 4
  W <- replicate(d, runif(n, -1, 1))
  colnames(W) <- paste0("W", 1:d)
  pi0 <- plogis(pos_const * ( W[,1] + sin(4*W[,1]) +   W[,2] + cos(4*W[,2]) + W[,3] + sin(4*W[,3]) + W[,4] + cos(4*W[,4]) ))

  print("pos")
  print(range(pi0))
  A <- rbinom(n, 1, pi0)
  if(muIsHard) {
    mu0 <-  sin(4*W[,1]) + sin(4*W[,2]) + sin(4*W[,3])+  sin(4*W[,4]) + cos(4*W[,2])
  } else {
    mu0 <-  W[,1] + abs(W[,2])  + W[,3] + abs(W[,4])
  }

  mu0 <- mu0 - (pi0/(pi0*(1-pi0)))/sqrt(n)
  tau <- 1 +   ( 1 / (pi0*(1-pi0))  )/sqrt(n)
  Y <- rnorm(n,  mu0 + A * tau, 0.5)
  return(list(W=W, A = A, Y = Y, ATE = 1 +  ates[[as.character(pos_const)]]/sqrt(n), pi = pi))
}

#' Given simulated data (W,A,Y) and simulation iteration number `iter`,
#' computes ATE estimates, se, and CI for plug-in T-learner HAL, plug-in R-learner HAL, partially linear intercept model, AIPW.
get_estimates <- function(W, A, Y,iter, pi_true) {
  n <- length(Y)
  if(n <= 500) {
    num_knots <- c(10, 10, 1, 0)
  } else if(n <= 1000) {
    num_knots <- c(50, 15, 15, 15)
  } else if(n <= 3000) {
    num_knots <- c(75, 25,30,30)
  } else{
    num_knots <- c(100, 50, 50,50)
  }
  fit_T <- fit_hal_cate_plugin (W, A, Y,   max_degree_cate = 1, num_knots_cate = num_knots , smoothness_orders_cate = 1, screen_variable_cate = FALSE,   params_EY0W =  list(max_degree = 1, num_knots =  num_knots , smoothness_orders = 1, screen_variables = FALSE, fit_control = list(parallel = TRUE)), fit_control = list(parallel = TRUE), include_propensity_score = FALSE,   verbose = TRUE )
  ate_T <- unlist(inference_ate(fit_T))
  ate_T[1] <- "Tlearner"

  mu1 <- fit_T$internal$data$mu1
  mu0 <- fit_T$internal$data$mu0
  mu <- ifelse(A==1, mu1, mu0)

  lrnr_stack <- Stack$new(list(  Lrnr_earth$new(degree = 2,    family = "gaussian"),Lrnr_gam$new(family = "gaussian"), Lrnr_ranger$new(), Lrnr_xgboost$new(max_depth = 4, nrounds = 20),  Lrnr_xgboost$new(max_depth = 5, nrounds = 20)  ))
  lrnr_A<- make_learner(Pipeline, Lrnr_cv$new(lrnr_stack), Lrnr_cv_selector$new(loss_squared_error) )
  task_A <- sl3_Task$new(data.table(W, A = A), covariates = colnames(W), outcome = "A", outcome_type = "continuous")

  fit_pi <- lrnr_A$train(task_A)

  pi <- fit_pi$predict(task_A)
  pi <- truncate_pscore_adaptive(A, pi)



  m <- mu0 * (1-pi) + mu1 * (pi)
  fit_R <- fit_hal_cate_partially_linear(W, A, Y,  fit_control = list(parallel = TRUE), pi.hat = pi, m.hat = m, formula_cate = NULL, max_degree_cate = 1, num_knots_cate = num_knots, smoothness_orders_cate = 1,      verbose = TRUE)
  ate_R<-  unlist(inference_ate(fit_R))
  ate_R[1] <- "Rlearner"

  #
  cate.hat <- fit_R$data$tau_relaxed
  calibrator <- isoreg_with_xgboost(cate.hat, fit_R$data$pseudo_outcome, weights = fit_R$data$pseudo_weights)
  cate_cal <- calibrator(cate.hat)
  # Create a data.table
  dt <- data.table(tau_cal, cond_var, weight = (A - pi)^2)
  gamma_dt <- dt[, .(gamma = weighted.mean(cond_var, w = weight)), by = tau_cal]
  dt <- merge(dt, gamma_dt, by = "tau_cal", all.x = TRUE, sort = FALSE)
  gamma_n <- dt$gamma
  IF <-  (A - pi) * gamma_n * (Y - m - (A-pi)*cate_cal)
  CI <- mean(cate_cal) + 1.96*c(-1,1)*sd(IF)/sqrt(n)
  ate_cal <- c("cal", mean(cate_cal), sd(IF)/sqrt(n), CI)

  tau_int <- mean((A-pi) * (Y - m)) / mean((A-pi)^2)
  IF <- (A - pi) / mean((A-pi)^2) * (Y - m - (A-pi)*tau_int)
  CI <- tau_int + 1.96*c(-1,1)*sd(IF)/sqrt(n)
  ate_intercept <- c("intercept", tau_int, sd(IF)/sqrt(n), CI)
  names(ate_intercept) <- c("method", "coef","se", "CI_left", "CI_right")


  #

  IF <- mu1 - mu0 +  (A/pi - (1-A)/(1-pi)) * (Y - mu)
  est_AIPW <-  mean(IF)
  CI <- est_AIPW + 1.96*c(-1,1)*sd(IF)/sqrt(n)
  ate_aipw <- c("AIPW", est_AIPW, sd(IF)/sqrt(n), CI)
  names(ate_aipw) <- c("method", "coef","se", "CI_left", "CI_right")

  mat <- cbind(iter,rbind(ate_T, ate_R, ate_intercept, ate_aipw, ate_cal))
  colnames(mat)  <- c("iter", "method", "coef","se", "CI_left", "CI_right")


  return(mat)
}
