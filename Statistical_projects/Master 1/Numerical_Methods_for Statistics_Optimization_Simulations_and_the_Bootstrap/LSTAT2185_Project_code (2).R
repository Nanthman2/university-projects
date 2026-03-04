

################################################################################
###                                                                          ###
###          EXERCISE N°1                                                    ###
###                                                                          ###
################################################################################



################################################################################
# QUESTION 1.f ----------------------------------------------------------------#
################################################################################

#############################                           
###   Conjugate gradient  ### 
#############################

set.seed(2003)
#Objective function
f<-function(x){
  if (any(x < 0)) {
    stop("the function is not defined for x < 0")
  }
  ifelse(any(x==0),return(0),return(x[1]*log(x[1])+x[2]*log(x[2])))
}
#Gradient
gradient<-function(x){
  if (any(x <= 0)){
    stop("x must be > 0")
  }
  return(c(1 + log(x[1]), 1 + log(x[2])))
}

#Update of the step size alpha_t
compute_alpha <- function(x_t, p_t) {
  f_alpha <- function(alpha) {
    x_new <- x_t + alpha * p_t
    if (any(x_new <= 0)) {
      break  
    }
    f(x_new)
  }
  optimize(f_alpha, interval = c(0, 1))$minimum
}

#Initialization 
tolerance<-1e-3
x_t<-c(1,1)
r_t<- -gradient(x_t)
p_t<-r_t

#Conjugate gradient algorithm 

for (t in 1:100){
  cat("Itération",t,":\n")

  alpha_t <-compute_alpha(x_t,p_t)
  x_t <-x_t+alpha_t*p_t
  r_new<- -gradient(x_t)

  
  #Does it converge? 
  if (sqrt(sum(r_new^2))<tolerance){
    cat("Convergence finished\n")
    break
  }
  
  #Update of the direction p_t
  p_t<-r_new+(sum(r_new*r_new)/sum(r_t*r_t))*p_t
  
  #Update of r_t
  r_t <-r_new
  
  #Result of iteration t 
  cat("x_t :",x_t,"\n")
  cat("alpha_t :",alpha_t,"\n")
  cat("r_t :",r_t,"\n")
  cat("p_t :",p_t,"\n\n")
}

#Final iteration results 
cat("After",t,"itérations : x_t =",x_t,"\n")



#############################  
###         BFGS          ###
#############################


# BFGS algorithm
bfgs_algorithm <- function(x_init, max_iter, tolerance) {
  x_t <- x_init
  B_t <- diag(2)
  
  # Store the values of x1 and x2 for each iteration
  x1_values_bfgs <- numeric(max_iter)
  x2_values_bfgs <- numeric(max_iter)
  
  # Include the initial point
  x1_values_bfgs[1] <- x_t[1]
  x2_values_bfgs[1] <- x_t[2]
  
  for (t in 1:max_iter) {
    
    direction_t <- -solve(B_t) %*% gradient(x_t)
    x_next <- x_t + direction_t
    
    alpha_t <- 1
    
    while (TRUE) {
      x_new <- x_t + alpha_t * direction_t
      if (all(x_new > 0)) {
        
        if (f(x_new) < f(x_t) + 1e-4 * alpha_t * sum(gradient(x_t) * direction_t)) {
          break
        }
      }
      alpha_t <- alpha_t * 0.5
      if (alpha_t < 1e-10) {
        stop("Step size is too small, unable to converge.")
      }
    }
    
    x_next <- x_t + alpha_t * direction_t
    s_t <- as.vector(x_next - x_t)
    y_t <- as.vector(gradient(x_next) - gradient(x_t))
    
    B_t <- B_t + (y_t %*% t(y_t)) / sum(y_t * s_t) - (B_t %*% s_t %*% t(s_t) %*% B_t) / sum(s_t * (B_t %*% s_t))
    
    x_t <- x_next
    
    x1_values_bfgs[t] <- x_t[1]
    x2_values_bfgs[t] <- x_t[2]
    
    if (sqrt(sum(gradient(x_t)^2)) < tolerance) {
      break
    }
  }
  
  list(solution = x_t, iterations = t, x1_values = x1_values_bfgs[1:t], x2_values = x2_values_bfgs[1:t])
}

# Run BFGS algorithm
resultats <- bfgs_algorithm(x_init, max_iter, tolerance)
cat("After t=", resultats$iterations, "iterations", "x_t = ", resultats$solution, "\n")

# Plotting x1 and x2 over iterations for BFGS

plot(1:resultats$iterations, resultats$x1_values, type = "o", col = "blue", xlab = "Iteration", ylab = "x1", main = "Convergence of x1 (BFGS)")
plot(1:resultats$iterations, resultats$x2_values, type = "o", col = "red", xlab = "Iteration", ylab = "x2", main = "Convergence of x2 (BFGS)")



################################################################################
# QUESTION 1.G ----------------------------------------------------------------#
################################################################################



#############################
###         Newton        ###
#############################


#Hessienne
hessian<-function(x) {
  if (any(x<=0)) stop("Hessienne non définie pour x <= 0.")
  matrix(c(1/x[1], 0, 0, 1/x[2]), nrow=2, byrow=TRUE)
}

#Parameters 
x_init<-c(1, 1)
max_iter<-100       
tolerance<-1e-6    
alpha<-0.25         


#Newton algorithm
newton_minimization<-function(x_init, max_iter, tolerance, alpha) {
  x_t<-x_init  
  for (t in 1:max_iter){
    grad<-gradient(x_t)
    hess<-hessian(x_t)
    
    #Hessienne invertible? 
    if (det(hess)<= 1e-10) stop("Hessian singular")
    
    #find direction
    direction<- -solve(hess)%*%grad
    
    #Update of x_next
    x_next<-x_t+alpha*direction
    
    
    #Does it converge ? 
    if (sqrt(sum(grad^2)) < tolerance) {
      cat("Convergence reached after", t, "itérations.\n")
      return(list(solution=x_next, iterations=t, gradient=grad))
    }
    
    #Update for next iteration
    x_t<-x_next
  }
  
  cat("Did not converged after", max_iter, "itérations.\n")
  return(list(solution=x_t, iterations=max_iter, gradient=gradient(x_t)))
}

result<-newton_minimization(x_init, max_iter, tolerance, alpha)

#Result
cat("x_t = ", result$solution, "\n")


################################################################################
###                                                                          ###
###          EXERCISE N°2                                                    ###
###                                                                          ###
################################################################################



################################################################################
# QUESTION 2.A ----------------------------------------------------------------#
################################################################################


# Data :
x <- c(82.8, 111.6, 79.2, 97.2, 79.2, 75.6, 115.2, 86.4, 86.4, 93.6, 
       79.2, 86.4, 100.8, 97.2, 90.0, 86.4, 90.0, 122.4, 93.6, 97.2, 
       93.6, 118.8)

# max(log-likelihood) = min(neg_log_likelihood) :

neg_log_likelihood <- function(theta, x) {
  mu <- theta[1]
  sigma <- theta[2]
  xi <- theta[3]
  n <- length(x)
  
  log_likelihood <- -n * log(sigma) + 
    (-(1/xi) - 1) * sum(log(1 + xi * (x - mu) / sigma)) - 
    sum((1 + xi * (x - mu) / sigma)^(-1/xi))
  
  return(-log_likelihood)
}


grad <- function(theta, x) {
  mu <- theta[1]
  sigma <- theta[2]
  xi <- theta[3]
  n <- length(x)
  
  dmu <- (1 + xi) / sigma * sum(1 / (1 + xi * (x - mu) / sigma)) - 
    sum(1 / sigma * (1 + xi * (x - mu) / sigma)^(-(1/xi + 1)))
  dsigma <- -n / sigma + 
    (1 + xi) / sigma^2 * sum((x - mu) / (1 + xi * (x - mu) / sigma)) - 
    1 / sigma^2 * sum((x - mu) * (1 + xi * (x - mu) / sigma)^(-(1/xi + 1)))
  dxi <- 1 / xi^2 * sum(log(1 + xi * (x - mu) / sigma)) + 
    (-1 / xi - 1) * sum((x - mu) / (sigma + xi * (x - mu))) - 
    sum(exp(-1/xi * log(1 + xi * (x - mu) / sigma)) * 
          (1 / xi^2 * log(1 + xi * (x - mu) / sigma) - 
             1 / xi * (x - mu) / (sigma + xi * (x - mu))))
  
  return(c(-dmu, -dsigma, -dxi))  
}


# Initial guess for mu and sigma
mu_0 <- mean(x)
sigma_0 <- sd(x)
xi_0 <- 0.1

# Conjugate Gradient
result <- optim(par = c(mu_0,sigma_0,xi_0),fn = neg_log_likelihood,gr = grad,x = x,method = "CG")

cat("Conjugate Gradient model 1:\n")
print(result)                # 87.51981848  9.57258864  0.06992916

# BFGS
result <- optim(par = c(mu_0,sigma_0,xi_0),fn = neg_log_likelihood,gr = grad,x = x,method = "BFGS")

# Résultats
cat("BFGS model 1:\n")
print(result)   # 87.51880159  9.57241095  0.06994663


library(ismev) # Similar results using the package ismev !
res <- gev.fit(x)
res                # 87.51954725  9.57247758  0.06991094
gev.diag(res)  

################################################################################
# QUESTION 2.B ----------------------------------------------------------------#
################################################################################


neg_log_likelihood_2 <- function(theta, x) {
  mu <- theta[1]
  sigma <- theta[2]
  
  n <- length(x)
  log_likelihood <- -n * log(sigma) - sum(exp(-(x - mu) / sigma) + (x - mu) / sigma)
  
  return(-log_likelihood)  
}

# Gradient de la log-vraisemblance
grad_2 <- function(theta, x) {
  mu <- theta[1]
  sigma <- theta[2]
  n <- length(x)
  
  dmu <- -1/sigma * sum(exp(-((x-mu)/sigma)) + 1)
  dsigma <- -n/sigma * sum( ((x-mu)/sigma^2) * (1-exp(-((x-mu)/sigma))) )
  
  return(c(-dmu, -dsigma))
}



mu_0 <- mean(x)  
sigma_0 <- sd(x)  

result2 <- optim(par = c(mu_0, sigma_0), fn = neg_log_likelihood_2, gr = grad_2, x = x, method = "CG")

cat("Conjugate Gradient model 2 :\n")
print(result)


result2 <- optim(par = c(mu_0, sigma_0), fn = neg_log_likelihood_2,gr =  grad_2, x = x, method = "BFGS")

cat("BFGS model 2:\n")
print(result)



################################################################################
# QUESTION 2.C ----------------------------------------------------------------#
################################################################################

ordered <- sort(x)
n <- length(ordered)


empirical <- vector('numeric',length(ordered))

for(i in 1:length(empirical)){empirical[i]=i/(length(x)+1)}

GEV_cdf <- function(data,mu,sigma,xi){
  if(xi==0){
    GEV <- exp(-exp(-((data-mu)/sigma)))}
  else{
    GEV <- exp(-(1+xi*((data-mu)/sigma))^(-1/xi))}
  return(GEV)
}

mu1 <- 87.51880159 ;  sigma1 <- 9.57241095 ;  xi1 <- 0.06994663


mu2 <- 88.483434 ;sigma2 <- 9.685979

model1=vector('numeric',length(x))

for(i in 1:length(model1)){
  model1[i]=GEV_cdf(ordered[i],mu1,sigma1,xi1)}

model2=vector('numeric',length(x))

for(i in 1:length(model2)){
  model2[i]=GEV_cdf(ordered[i],mu2,sigma2,0)}


plot(empirical, model1, type = "p", col = "blue", xlab = "Empirical c.d.f", ylab = "Theoretical c.d.f")
points(empirical, model2, col = "red")
abline(0, 1, col = "black") 

grid(col = "lightgray", lty = "dotted")

legend("bottomright", legend = c("Model 1", "Model 2"), col = c("blue", "red"), pch = 1,bty = "n")

################################################################################
# QUESTION 2.D ----------------------------------------------------------------#
################################################################################


logL1 <- -result$value
logL1

AIC1 <- 2*3 - 2 * logL1 
cat("AIC du modèle 1 :", AIC1, "\n")

logL2 <- -result2$value  
logL2

AIC2 <- 2*2 - 2 * logL2
cat("AIC du modèle 2 :", AIC2, "\n")

################################################################################
###                                                                          ###
###          EXERCISE N°3                                                    ###
###                                                                          ###
################################################################################

################################################################################
# QUESTION 3.B ----------------------------------------------------------------#
################################################################################


# -------------------------------------
# Point 1: Bootstrap bias and variance
# -------------------------------------

set.seed(2003)

#Initialization
lambda<-5
c<-2   
n<-30 
B<-10000
                                                B_prime<-500
data <-rweibull(n, shape = c, scale = lambda)
                                                exact_theta <-lambda^c

t_hat <- mean(data^c)

bootstrap_ts <-numeric(B)

for (b in 1:B){
  bootstrap_sample<-sample(data, size=n, replace=TRUE)
  bootstrap_ts[b]<-mean(bootstrap_sample^c)
}

#bias bootstrap
bias_bootstrap<-mean(bootstrap_ts) - t_hat


#variance bootstrap
variance_bootstrap<- mean(bootstrap_ts^2) - (mean(bootstrap_ts))^2

#Results
cat("Point 1: Bootstrap Results\n")
cat("Bias:", bias_bootstrap, "\n")
cat("Variance:", variance_bootstrap, "\n")
cat("Error bound:", 1.96*sqrt(variance_bootstrap)/(sqrt(B)), "\n")

# Correction needed ?
cat("Efron and Tibshirani rule of thumb :",abs(bias_bootstrap)/sqrt(variance_bootstrap))

y <- seq(min(bootstrap_ts),max(bootstrap_ts),length = 300)
plot(density(bootstrap_ts),main = "Smoothed distribution of T(X)", sub = paste("initial sample of size n =",n,"number of resamples B =",B),lwd=2)
abline(v=exact_theta,col="red")


# -------------------------------------
# Point 2: Confidence Intervals
# -------------------------------------

#Initialisation 
set.seed(2003)
lambda<-5 
c<-2 
n<-30
alpha<-0.05
M<-1000

#Weibull observations 
x<-rweibull(n, shape = c, scale = lambda)
mu_hat<-mean(x^c) 
sigma_hat<-sqrt(var(x^c)) 

#Bootstrap vector 
TstarM<-numeric(M)       
TstartM<-numeric(M)      
mustarM<-numeric(M)      

#First level of bootstrap
for (m in 1:M){
  x_star<-sample(x, replace = TRUE) 
  mu_star<-mean(x_star^c)        
  sigma_star<-sqrt(var(x_star^c))  
  
  mustarM[m]<-mu_star
  TstarM[m]<- (mu_star - mu_hat)
  TstartM[m]<-sqrt(n)*TstarM[m]/sigma_star
}

#1 Asymptotic CI
CI_asymptotic<-c(
  mu_hat -qnorm(1 -alpha/2)*sigma_hat/sqrt(n),
  mu_hat+qnorm(1 -alpha/2)*sigma_hat/sqrt(n)
)

#2 base bootstrap CI
CI_bootstrap<-c(
  mu_hat -quantile(TstarM, 1 -alpha/2),
  mu_hat -quantile(TstarM, alpha/2)
)

#3 percentile bootstrap CI
CI_percentile<-c( mu_hat +
  quantile(TstarM, alpha /2), mu_hat +
  quantile(TstarM, 1 -alpha/2)
)

#4 t-bootstrap CI
CI_t_bootstrap<-c(
  mu_hat -quantile(TstartM, 1-alpha/2)*sigma_hat/sqrt(n),
  mu_hat -quantile(TstartM,alpha /2)*sigma_hat/sqrt(n)
)

#Results
cat("Intervalle de confiance asymptotique :", CI_asymptotic, "\n")
cat("Intervalle bootstrap de base :", CI_bootstrap, "\n")
cat("Intervalle bootstrap percentile :", CI_percentile, "\n")
cat("Intervalle t-bootstrap :", CI_t_bootstrap, "\n")




#Iterated t-Bootstrap Confidence Interval

#Initialisation
B1<-1000
B2<-500 
theta_hat<-mean(x^c)

# First-level bootstrap
T_star<-numeric(B1)      
U_star<-numeric(B1)       
for (b1 in 1:B1) {
  #First level bootstrap
  x_star<-sample(x, replace=TRUE)
  theta_star<-mean(x_star^c)
  
  # Second level bootstrap
  second_bootstrap_T<-numeric(B2)
  for (b2 in 1:B2) {
    x_star_star<-sample(x_star, replace=TRUE)
    theta_star_star<-mean(x_star_star^c)
    second_bootstrap_T[b2]<-sqrt(n)*(theta_star_star- theta_star)
  }
  
  #variance for second-level bootstrap
  variance_star<- var(second_bootstrap_T)
  U_star[b1]<-sqrt(n)*(theta_star- theta_hat)/sqrt(variance_star)
}

#quantils of U_star for CI
q_U_star <- quantile(U_star, probs = c(alpha/2, 1 -alpha/ 2))

#Final iterated t-bootstrap confidence interval
CI_iterated_t <- c(
  theta_hat- q_U_star[2]* sqrt(var(x^c))/sqrt(n),
  theta_hat- q_U_star[1]* sqrt(var(x^c))/sqrt(n)
)

# Results
cat("Iterated t-bootstrap confidence interval (95%):", CI_iterated_t, "\n")




################################################################################
# QUESTION 3.C ----------------------------------------------------------------#
################################################################################


# -------------------------------------
# Point 1: Hypothesis Testing
# -------------------------------------

set.seed(2003) 

#Initialisation

lambda_true<-5
c<-2         
n<-30         
B<-1000        
threshold<-10 
x <-rweibull(n, shape=c, scale=lambda_true)

T_stat<-function(data, c){
  mean(data^c)
}
T_obs<-T_stat(x, c)

bootstrap_H0 <- function(n, num_iterations,c,lam) {
  set.seed(2003)
  bootstrap_estimates <- numeric(num_iterations)
  initial_sample <- rweibull(n, shape = c, scale = lam)
  for (i in 1:num_iterations) {
    sample <- sample(initial_sample,size = n,replace = TRUE)
    bootstrap_estimates[i] <- T_stat(sample, c)
  }
  return(bootstrap_estimates)
}

# true H0 E[X^c] = 25 because (Weibull(5,2))

bootstrap_H0_estimates_T <- bootstrap_H0(30,1000,2,5)

# Bootstrap estimates under a "fake H0" 

bootstrap_H0_estimates_F <- bootstrap_H0(30,1000,2,4)

# Comparaison et test de l'hypothèse
# Calcul de la p-value sous H0 et H1
p_value_H0_T <- (sum(bootstrap_H0_estimates_T>=T_obs)+ 1)/(B + 1)
p_value_H0_F <- (sum(bootstrap_H0_estimates_F>=T_obs)+ 1)/(B + 1)

# Affichage des résultats
cat("Observed T(X) :", T_obs, "\n")
cat("P-value under H0 true :", p_value_H0_T, "\n")
cat("P-value under H0 false :", p_value_H0_F, "\n")

y <- seq(min(bootstrap_H0_estimates_T),max(bootstrap_H0_estimates_T),length = 300)
plot(density(bootstrap_H0_estimates_T),main = "Smoothed distribution of T(X) under H0", sub = paste("initial sample of size n =",n,"number of resamples B =",B),lwd=2)
abline(v=T_obs,col="red")

y <- seq(min(bootstrap_H0_estimates_F),max(bootstrap_H0_estimates_F),length = 300)
plot(density(bootstrap_H0_estimates_F),main = "", sub = paste("initial sample of size n =",n,"number of resamples B =",B),lwd=2)
abline(v=T_obs,col="red")


# -------------------------------------
# Point 2: Power
# -------------------------------------

set.seed(2003)

# Non-Parametric Bootstrap
bootstrap_non_parametric <- function(data, num_iterations, c) {
  bootstrap_estimates <- numeric(num_iterations)
  for (i in 1:num_iterations) {
    sample <- sample(data, size = length(data), replace = TRUE)
    bootstrap_estimates[i] <- T_stat(sample, c)
  }
  return(bootstrap_estimates)
}

# Parametric Bootstrap
bootstrap_parametric <- function(n, num_iterations, c, lambda) {
  bootstrap_estimates <- numeric(num_iterations)
  for (i in 1:num_iterations) {
    sample <- rweibull(n, shape = c, scale = lambda)
    bootstrap_estimates[i] <- T_stat(sample, c)
  }
  return(bootstrap_estimates)
}


# Power Simulation
simulate_power <- function(num_simulations, n, c, B, lambda_H0, lambda_H1, alpha) {
  rejections_non_param <- 0
  rejections_param <- 0
  
  for (sim in 1:num_simulations) {
    # Generate sample under H1 (lambda_H1)
    data <- rweibull(n, shape = c, scale = lambda_H1)
    T_obs <- T_stat(data, c)
    
    # Non-Parametric Bootstrap
    boot_non_param <- bootstrap_non_parametric(data, B, c)
    p_value_non_param <- (sum(boot_non_param>=T_obs)+ 1)/(B + 1)
    rejections_non_param <- rejections_non_param + (p_value_non_param < alpha)
    
    # Parametric Bootstrap
    boot_param <- bootstrap_parametric(n, B, c, lambda_H0)
    p_value_param <- (sum(boot_param>=T_obs)+ 1)/(B + 1)
    rejections_param <- rejections_param + (p_value_param < alpha)
  }
  
  # Calculate power
  power_non_param <- rejections_non_param / num_simulations
  power_param <- rejections_param / num_simulations
  
  return(list(
    power_non_param = power_non_param,
    power_param = power_param
  ))
}

# Parameters
num_simulations <- 500
lambda_H0 <- 5      # Under H0
lambda_H1 <- 4      # Under H1
n <- 30
c <- 2
B <- 1000
alpha <- 0.05

# Run Power Simulation
power_results <- simulate_power(num_simulations, n, c, B, lambda_H0, lambda_H1, alpha)

# Results
cat("Power of Non-Parametric Bootstrap:", power_results$power_non_param, "\n")
cat("Power of Parametric Bootstrap:", power_results$power_param, "\n")





################################################################################
# QUESTION 3.D ----------------------------------------------------------------#
################################################################################

# Monte-Carlo simulation function
monte_carlo_simulation <- function(n, B, B_bootstrap, true_theta, alpha) {
  coverage <- matrix(0, nrow = B, ncol = 4)
  widths <- matrix(0, nrow = B, ncol = 4)
  
  for (b in 1:B) {
    # Weibull observations 
    x <- rweibull(n, shape = c, scale = lambda)
    theta_hat <- mean(x^c)
    sigma_hat <- sqrt(var(x^c)) / sqrt(n)
    
    # 1 Asymptotic interval
    CI_asymptotic <- c(
      theta_hat - qnorm(1 - alpha / 2) * sigma_hat,
      theta_hat + qnorm(1 - alpha / 2) * sigma_hat
    )
    
    # 2 Base bootstrap
    bootstrap_samples <- replicate(B_bootstrap, mean(sample(x^c, replace = TRUE)))
    T_star <- (bootstrap_samples - theta_hat)
    CI_bootstrap <- c(
      theta_hat - quantile(T_star, 1 - alpha / 2),
      theta_hat - quantile(T_star, alpha / 2)
    )
    
    # 3 Percentile bootstrap
    CI_percentile <- c(
      theta_hat + quantile(T_star, alpha / 2),
      theta_hat + quantile(T_star, 1 - alpha / 2)
    )
    
    # 4 t-Bootstrap
    U <- numeric(B_bootstrap)
    for (b2 in 1:B_bootstrap) {
      x_star <- sample(x, replace = TRUE)
      theta_star <- mean(x_star^c)
      sigma_star <- sqrt(var(x_star^c)) / sqrt(n)
      U[b2] <- (theta_star - theta_hat) / sigma_star
    }
    q_alpha <- quantile(U, probs = c(alpha / 2, 1 - alpha / 2))
    CI_t_bootstrap <- c(
      theta_hat - q_alpha[2] * sigma_hat,
      theta_hat - q_alpha[1] * sigma_hat
    )
    
    # Store the results
    intervals <- list(CI_asymptotic, CI_bootstrap, CI_percentile, CI_t_bootstrap)
    for (i in 1:4) {
      coverage[b, i] <- (intervals[[i]][1] <= true_theta && intervals[[i]][2] >= true_theta)
      widths[b, i] <- diff(intervals[[i]])
    }
  }
  
  # Coverage probability and average widths
  coverage_probs <- colMeans(coverage)
  avg_widths <- colMeans(widths)
  
  return(c(coverage_probs, avg_widths))
}

# Parallel execution
library(foreach)
library(doParallel)

set.seed(2003)
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Parameters
n_values <- c(10, 50, 100, 500)
B <- 1000
B_bootstrap <- 500
true_theta <- 25
lambda <- 5
c <- 2
alpha <- 0.05

results <- foreach(n = n_values, .combine = rbind, .packages = c("stats")) %dopar% {
  monte_carlo_simulation(n, B, B_bootstrap, true_theta, alpha)
}

stopCluster(cl)

# Results
for (i in seq_along(n_values)) {
  cat("\nSample size n =", n_values[i], "\n")
  cat("Coverage probabilities (asymptotic, base bootstrap, percentile, t-bootstrap):\n")
  print(results[i, 1:4])
  cat("Average widths of CIs (asymptotic, base bootstrap, percentile, t-bootstrap):\n")
  print(results[i, 5:8])
}
