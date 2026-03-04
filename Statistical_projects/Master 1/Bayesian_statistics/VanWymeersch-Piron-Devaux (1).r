############################################################################
## Project: Introduction to Bayesian statistics
############################################################################

# --- Data import ---
HospitalVisits <- read.delim("HospitalVisits.txt")
y <- HospitalVisits$y
x1 <- HospitalVisits$age
x2 <- HospitalVisits$chronic
hospital <- HospitalVisits$hospital
G <- 30

# --- Libraries needed ---
library(mvtnorm)
library(coda)
library(dplyr)
library(patchwork)
library(ggplot2)
library(rjags)
library(R2WinBUGS)
library(R2jags)

#############################################################################
# QUESTION 3 ---------------------------------------------------------------#
#############################################################################

# --- Log-likelihood and Log-posterior ---

# Log-likelihood function for Poisson distribution
loglik <- function(beta, y, x1, x2, hospital, v) {
  mu <- v[hospital] * exp(beta[1] + beta[2]*x1 + beta[3]*x2)
  ll <- sum(dpois(y, mu, log = TRUE))  
  return(ll)
}

# Log-posterior function for beta
logpost_beta <- function(beta, y, x1, x2, hospital, v, sigma2 = 100) {
  ll <- loglik(beta, y, x1, x2, hospital, v)  # Likelihood
  lp <- sum(dnorm(beta, 0, sqrt(sigma2), log = TRUE))  # Log-prior for beta
  return(ll + lp)
}

# Log-posterior function for alpha
logpost_alpha <- function(u, v, a0 = 0.01, b0 = 0.01) {
  # Reparametrize alpha -> u = log(alpha)
  alpha <- exp(u) 
  G <- 30  
  # Log-posterior for alpha with the Jacobian adjustment
  lp <- G * alpha * u - G * lgamma(alpha) +
    (alpha - 1) * sum(log(v)) - alpha * sum(v) +
    (a0 - 1) * u - b0 * alpha + u
  return(lp)
}

# --- Posterior Estimation (Laplace Approximation) ---

# Used to find the posterior mode using optim
find_posterior_mode <- function(init, y, x1, x2, v, hospital) {
  # Negative log-posterior function for optimization
  neg_log_post <- function(beta) -logpost_beta(beta, y, x1, x2, hospital, v)
  opt <- optim(init, neg_log_post, hessian = TRUE)
  return(list(mode = opt$par, Sigma = solve(opt$hessian)))
}

init <- c(0, 0, 0)  
opt <- find_posterior_mode(init, y, x1, x2, v = rep(1, 30), hospital) 
beta_mode <- opt$mode
Sigma <- opt$Sigma

# Function to initialize the chains from a multivariate normal distribution

starting_values <- function(starting_index,scale = .01){
  set.seed(123)
  list(
    beta = rmvnorm(5,beta_mode,scale*diag(nrow = 3))[starting_index,],
    u = rnorm(1, mean = log(1), sd = 0.1),
    v = rep(1, 30)
  )
}


# ---MCMC SAMPLER ---

run_mcmc <- function(starting_values, M=20000, y, x1, x2, hospital, sd_beta = c(1, 1, 1), sd_u = 1) {
  # Starting values for the parameters
  beta <- starting_values$beta  
  u <- starting_values$u        # Initial alpha (log-scale)
  alpha <- exp(u)               # Convert to alpha
  v <- starting_values$v        
  
  G <- 30  
  n <- length(y)  
  
  # Output storage for posterior samples
  beta_post <- matrix(NA, M, 3)   
  alpha_post <- numeric(M)         
  v_post <- matrix(NA, M, G)     
  
  # Acceptance rates
  accept_beta <- rep(0, 3)
  accept_alpha <- 0
  
  # MCMC sampling loop
  for (m in 1:M) {
    # ---- 1. Sample v_g | y, beta, alpha [Gibbs] ----
    for (g in 1:G) {
      idx <- which(hospital == g)
      y_g <- sum(y[idx])
      phi_g <- sum(exp(beta[1] + beta[2] * x1[idx] + beta[3] * x2[idx]))
      v[g] <- rgamma(1, shape = alpha + y_g, rate = alpha + phi_g)
    }
    
    # ---- 2. Sample beta | y, v, alpha [Metropolis component-wise] ----
    for (j in 1:3) {
      beta_prop <- beta
      beta_prop[j] <- rnorm(1, beta[j], sd_beta[j])                 # Propose new value for beta_j
      lp_prop <- logpost_beta(beta_prop, y, x1, x2, hospital, v)    # Log-posterior for proposed beta
      lp_curr <- logpost_beta(beta, y, x1, x2, hospital, v)         # Log-posterior for current beta
      prob <- min(1, exp(lp_prop - lp_curr))
      if (runif(1) < prob) {                                        # Accept the proposal
        beta[j] <- beta_prop[j]
        accept_beta[j] <- accept_beta[j] + 1
      }
    }
    
    # ---- 3. Sample alpha | v [MH on log scale] ----
    u_prop <- rnorm(1, u, sd_u)                                     # Propose new alpha (log scale)
    lp_prop <- logpost_alpha(u_prop, v)                             # Log-posterior for proposed alpha
    lp_curr <- logpost_alpha(u, v)                                  # Log-posterior for current alpha
    prob <- min(1, exp(lp_prop - lp_curr))                          # Acceptance probability
    if (runif(1) < prob) {                                          # Accept the proposal
      u <- u_prop
      alpha <- exp(u)                                               # Convert back to alpha
      accept_alpha <- accept_alpha + 1
    }
    
    beta_post[m, ] <- beta
    alpha_post[m] <- alpha
    v_post[m, ] <- v
  }
  
  cat("Acceptance rates:\n")
  cat("  beta:", round(accept_beta / M, 3), "\n")
  cat("  alpha:", round(accept_alpha / M, 3), "\n")
  
  list(beta_post = beta_post, alpha_post = alpha_post, v_post = v_post)
}



# Preliminary investigation of Chain 1 ----------------------------------------#

M <- 50000
sd_beta <- c(0.062, 0.00115, 0.08)
sd_u <- 0.56
start <- starting_values(starting_index = 1)

res <- run_mcmc(start,M, y, x1, x2, hospital,sd_beta,sd_u)

# ACF of Chain 1 (without burnin)
burnin <- 0
beta_post <- res$beta_post[-(1:burnin), ]
v_post <- res$v_post[-(1:burnin), ]
alpha_post <- res$alpha_post[-(1:burnin)]

mcmc_beta <- mcmc(beta_post)
mcmc_alpha <- mcmc(alpha_post)

par(mfrow = c(2, 2))
acf(mcmc_beta[, 1], main = "ACF Beta 0",lag.max = 50000)
acf(mcmc_beta[, 2], main = "ACF Beta 1",lag.max = 50000)
acf(mcmc_beta[, 3], main = "ACF Beta 2",lag.max = 50000)
acf(mcmc_alpha, main = "ACF Alpha",lag.max = 50000)


# Lauching 5 Chains  ------------------------------------------------#

burnin <- 5000
n_chains <- 5
chains <- vector("list", n_chains)

for (i in 1:n_chains) {
  cat("Running chain", i, "\n")
  start_i <- starting_values(i)
  chains[[i]] <- run_mcmc(
    starting_values = start_i,
    M = M,
    y = y,
    x1 = x1,
    x2 = x2,
    hospital = hospital,
    sd_beta = c(0.062, 0.00115, 0.08),
    sd_u = 0.56
  )
}

# Traceplots


mcmc_chains_full <- lapply(chains, function(res) {
  mcmc(cbind(
    beta0 = res$beta_post[, 1],
    beta1 = res$beta_post[, 2],
    beta2 = res$beta_post[, 3],
    alpha = res$alpha_post
  ))
})
multi_chain_full <- mcmc.list(mcmc_chains_full)

par(mfrow = c(2, 2))
traceplot(multi_chain_full, col = 1:n_chains)


# Gelman-Rubin on first 10000


mcmc_chains_early <- lapply(chains, function(res) {
  mcmc(cbind(
    beta0 = res$beta_post[1:10000, 1],
    beta1 = res$beta_post[1:10000, 2],
    beta2 = res$beta_post[1:10000, 3],
    alpha = res$alpha_post[1:10000]
  ))
})

multi_chain_early <- mcmc.list(mcmc_chains_early)

gelman_early <- gelman.diag(multi_chain_early, autoburnin = FALSE)
print(gelman_early)

gelman.plot(multi_chain_early, autoburnin = FALSE)


# Cut Burn-in

mcmc_chains <- lapply(chains, function(res) {
  mcmc(cbind(
    beta0 = res$beta_post[-(1:burnin), 1],
    beta1 = res$beta_post[-(1:burnin), 2],
    beta2 = res$beta_post[-(1:burnin), 3],
    alpha = res$alpha_post[-(1:burnin)]
  ))
})

multi_chain <- mcmc.list(mcmc_chains)
mcmc_chains_all <- lapply(chains, function(res) {
  v_names <- paste0("v_", 1:ncol(res$v_post))
  mcmc(cbind(
    beta0 = res$beta_post[-(1:burnin), 1],
    beta1 = res$beta_post[-(1:burnin), 2],
    beta2 = res$beta_post[-(1:burnin), 3],
    alpha = res$alpha_post[-(1:burnin)],
    setNames(as.data.frame(res$v_post[-(1:burnin), ]), v_names)
  ))
})
multi_chain_all <- mcmc.list(mcmc_chains_all)

# Traceplot after burn-in

par(mfrow = c(2, 2))
traceplot(multi_chain, col = 1:5)


# Final diagnostics

gelman_res <- gelman.diag(multi_chain, autoburnin = FALSE)
print(gelman_res)

gelman.plot(multi_chain, autoburnin = FALSE)

geweke_res <- lapply(multi_chain, geweke.diag)

for (i in seq_along(multi_chain)) {
  geweke.plot(multi_chain[[i]])
}

ess <- effectiveSize(multi_chain)
print(ess)

################################################################################
# Multichain MCMC with extreme settings ---------------------------------------#
################################################################################
# # We commented this section to avoid running it by default
 # as it is computationally intensive and not necessary for the main analysis.
# n_chains <- 4
# chains_extreme <- vector("list", n_chains)
 
# # -----------------------------
# # Starting values for chains
# # -----------------------------
 
#  starting_values <- function(starting_index) {
#  beta_list <- list(
#      c(0, 0, 0),              # Chain 1: neutral
#      c(5, 0.5, 1),            # Chain 2: strongly positive
#      c(-5, -0.5, -1),         # Chain 3: strongly negative
#      c(2, -0.8, 0.3)          # Chain 4: mixed
#    )
   
#   eta_list <- list(
#     log(1),     # alpha = 1
#     log(10),    # alpha = 10
#     log(0.1),   # alpha = 0.1
#     log(5)      # alpha = 5
#   )
   
#   list(
#     beta = beta_list[[starting_index]],
#     u  = eta_list[[starting_index]],  
#     v  = rep(1, 30)
#   )
# }
# 
# # -----------------------------
# # Run MCMC chains
# # -----------------------------
# 
# for (i in 1:n_chains) {
#   cat("Running chain", i, "\n")
#   
#   start_i <- starting_values(i)
#   
#   chains_extreme[[i]] <- run_mcmc(
#     starting_values = start_i,
#     M = M,
#     y = y,
#     x1 = x1,
#     x2 = x2,
#     hospital = hospital,
#     sd_beta = c(0.062, 0.00115, 0.08),sd_u = 0.56
#   )
# }
# burnin <- 0
# # Create mcmc objects from samples
# mcmc_chains <- lapply(chains_extreme, function(res) {
#   mcmc(cbind(
#     beta0 = res$beta_post[-(1:burnin), 1],
#     beta1 = res$beta_post[-(1:burnin), 2],
#     beta2 = res$beta_post[-(1:burnin), 3],
#     alpha = res$alpha_post[-(1:burnin)]
#   ))
# })
# 
# multi_chain <- mcmc.list(mcmc_chains)
# # Traceplots before burn-in
# par(mfrow = c(2, 2))
# traceplot(multi_chain, col = 2:5)



################################################################################
# QUESTION 4 ------------------------------------------------------------------#
################################################################################

################################################################################
#(a) Summary of our model parameter -------------------------------------------#
################################################################################

# --- B coefficients ---

beta_mcmc <- mcmc(beta_post)
print(summary(beta_mcmc))
print(HPDinterval(beta_mcmc, prob = 0.95))

# --- Alpha ---

alpha_mcmc <- mcmc(alpha_post)
print(summary(alpha_mcmc))
print(HPDinterval(alpha_mcmc, prob = 0.95))

# --- Random effects (Vg) ---

#CI
post_interv_v <- apply(v_post, 2, quantile, probs = c(0.025, 0.5, 0.975))
median_global_v <- median(as.vector(v_post))


# Densities of parameters 

par(mfrow = c(1, 1))

# Beta0
ggplot(data.frame(beta0 = beta_post[, 1]), aes(x = beta0)) +
  geom_histogram(aes(y=after_stat(density)), bins = 30, fill = 'lightgray', color = 'black', alpha = 0.7) +
  geom_density(color = 'blue') +
  labs(title = 'Density of Beta0', x = 'Beta0', y = 'Density') +
  theme_minimal(base_size = 12)

# Beta1
ggplot(data.frame(beta1 = beta_post[, 2]), aes(x = beta1)) +
  geom_histogram(aes(y=after_stat(density)), bins = 30, fill = 'lightgray', color = 'black', alpha = 0.7) +
  geom_density(color = 'blue') +
  labs(title = 'Density of Beta1', x = 'Beta1', y = 'Density') +
  theme_minimal(base_size = 12)

# Beta2
ggplot(data.frame(beta2 = beta_post[, 3]), aes(x = beta2)) +
  geom_histogram(aes(y=after_stat(density)), bins = 30, fill = 'lightgray', color = 'black', alpha = 0.7) +
  geom_density(color = 'blue') +
  labs(title = 'Density of Beta2', x = 'Beta2', y = 'Density') +
  theme_minimal(base_size = 12)

# Alpha
ggplot(data.frame(alpha = alpha_post), aes(x = alpha)) +
  geom_histogram(aes(y=after_stat(density)), bins = 30, fill = 'lightgray', color = 'black', alpha = 0.7) +
  geom_density(color = 'blue') +
  labs(title = 'Density of Alpha', x = 'Alpha', y = 'Density') +
  theme_minimal(base_size = 12)

# v_g

plot(1:G, xlim = range(post_interv_v), ylim = c(0.5, G + 0.5),
     type = "n", yaxt = "n", ylab = "Hospital index", xlab = "Random effect (v_g)",
     main = "95% Credible Intervals for v_g")
axis(2, at = 1:G, labels = 1:G)

for (g in 1:G) {
  lines(post_interv_v[c(1, 3), g], rep(g, 2), lwd = 2)
  points(post_interv_v[2, g], g, pch = 20)
}
abline(v = median_global_v, col = "red", lwd = 2, lty = 2)



################################################################################
#(b) Posterior predictive performance -----------------------------------------#
################################################################################

# --- Extraction ---
all_samples_mat <- do.call(rbind, lapply(multi_chain_all, as.matrix))
beta_post  <- all_samples_mat[, c("beta0", "beta1", "beta2")]
alpha_post <- all_samples_mat[, "alpha"]
v_post     <- all_samples_mat[, grep("^v_", colnames(all_samples_mat))]

M <- nrow(beta_post)
n <- length(y)

# --- Predictions ---
y_pred <- matrix(NA, nrow = M, ncol = n)
for (m in 1:M) {
  mu <- v_post[m, hospital] * exp(beta_post[m, 1] + beta_post[m, 2] * x1 + beta_post[m, 3] * x2)
  y_pred[m, ] <- rpois(n, mu)
}
yrep_vec <- as.vector(y_pred)

#Numerical comparaison y vs y_pred
print(summary(y))
summary(as.vector(y_pred))

# --- Visual comparaison ---
df_obs <- data.frame(y = y, type = "Observed")
df_pred <- data.frame(y = yrep_vec, type = "Predicted")
df_densite <- rbind(df_obs, df_pred)

df_tab <- table(df_densite$type, df_densite$y)
df_freq <- as.data.frame(df_tab)
colnames(df_freq) <- c("type", "y", "count")

df_freq$y <- as.numeric(as.character(df_freq$y))

freq_by_type <- tapply(df_freq$count, df_freq$type, sum)
df_freq$freq <- mapply(function(t, c) c / freq_by_type[[t]], df_freq$type, df_freq$count)


df_obs <- df_freq[df_freq$type == "Observed", ]

p1 <- ggplot(df_obs, aes(x = factor(y), y = freq)) + geom_col(fill = "steelblue", color = "black") +
  labs(title = "Observed", x = "Number of visits", y = "Frequency") + theme_minimal(base_size = 13) +theme(plot.title = element_text(hjust = 0.5))+
  scale_x_discrete(limits = as.character(0:16))

p1

df_pred <- df_freq[df_freq$type == "Predicted", ]

p2 <- ggplot(df_pred, aes(x = factor(y), y = freq)) + geom_col(fill = "tomato", color = "black") +
  labs(title = "Predicted", x = "Number of visits", y = "Frequency") +
  theme_minimal(base_size = 13) +theme(plot.title = element_text(hjust = 0.5)) + scale_x_discrete(limits = as.character(0:16)) 
p2

p1 / p2

ggplot(df_freq[df_freq$y <= 16, ], aes(factor(y),freq, fill = type)) +geom_col(position = "identity", alpha = 0.5, color = "black") +
  scale_fill_manual(values = c("Observed" = "steelblue", "Predicted" = "tomato")) +
  labs(x = "Number of visits", y = "Frequency", fill = "Data type") + theme_minimal(base_size = 13)+scale_x_discrete(limits = as.character(0:32))+theme(
    legend.position = "top"
  )

################################################################################
# QUESTION 5 ------------------------------------------------------------------#
################################################################################


################################################################################
#(a) JAGS Implementation ------------------------------------------------------#
################################################################################

ModelQ5 <- function() {
  
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dpois(mu[i])
    mu[i] <- v[hospital[i]] * exp(beta0 + beta1 * x1[i] + beta2 * x2[i])
  }
  
  # Priors for regression coefficients
  beta0 ~ dnorm(0, 0.01)
  beta1 ~ dnorm(0, 0.01)
  beta2 ~ dnorm(0, 0.01)
  
  # Random effects per hospital
  for (g in 1:G) {
    v[g] ~ dgamma(alpha, alpha)
  }
  
  # Hyperprior on alpha
  alpha ~ dgamma(0.01, 0.01)
}

model.file <- "question5_model.bug"
write.model(ModelQ5, model.file)

mydata <- list(y = y,x1 = x1,x2 = x2,hospital = hospital,N = length(y),G = 30)

myinits <- vector("list", 5)
for (i in 1:5) {
  start_i <- starting_values(i)  
  myinits[[i]] <- list(
    beta0 = start_i$beta[1],
    beta1 = start_i$beta[2],
    beta2 = start_i$beta[3],
    alpha = exp(start_i$u),
    v = start_i$v
  )
}

jags_model <- jags.model(
  file = model.file,
  data = mydata,
  inits = myinits,
  n.chains = 5
)

samples <- coda.samples(
  model = jags_model,
  variable.names = c("beta0", "beta1", "beta2", "alpha", "v"),
  n.iter = 45000
)

samples_matrix <- as.matrix(samples) # For latter plots

# Numerical Summary
print(summary(samples))

# Graphical Summary

# v_g
v_post_jags <- as.matrix(samples)[, grep("^v", colnames(as.matrix(samples)))]

post_interv_v <- apply(v_post_jags, 2, quantile, probs = c(0.025, 0.5, 0.975))

median_global_v <- median(as.vector(v_post_jags))

par(mfrow = c(1, 1))
plot(1:G, xlim = range(post_interv_v), ylim = c(0.5, G + 0.5),
     type = "n", yaxt = "n", ylab = "Hospital index", xlab = "Random effect (v_g)",
     main = "95% CI for hospital random effects (JACGS)")

axis(2, at = 1:G, labels = 1:G)

for (g in 1:G) {
  lines(post_interv_v[c(1, 3), g], rep(g, 2), lwd = 2)
  points(post_interv_v[2, g], g, pch = 20)
}

abline(v = median_global_v, col = "red", lwd = 2, lty = 2)


# Beta0
ggplot(data.frame(beta0 = samples_matrix[, "beta0"]), aes(x = beta0)) +
  geom_histogram(aes(y=after_stat(density)), bins = 30, fill = 'lightgray', color = 'black', alpha = 0.7) +
  geom_density(color = 'blue') +
  labs(title = 'Density of Beta0', x = 'Beta0', y = 'Density') +
  theme_minimal(base_size = 12)

# Beta1
ggplot(data.frame(beta1 = samples_matrix[, "beta1"]), aes(x = beta1)) +
  geom_histogram(aes(y=after_stat(density)), bins = 30, fill = 'lightgray', color = 'black', alpha = 0.7) +
  geom_density(color = 'blue') +
  labs(title = 'Density of Beta1', x = 'Beta1', y = 'Density') +
  theme_minimal(base_size = 12)

# Beta2
ggplot(data.frame(beta2 = samples_matrix[, "beta2"]), aes(x = beta2)) +
  geom_histogram(aes(y=after_stat(density)), bins = 30, fill = 'lightgray', color = 'black', alpha = 0.7) +
  geom_density(color = 'blue') +
  labs(title = 'Density of Beta2', x = 'Beta2', y = 'Density') +
  theme_minimal(base_size = 12)

# Alpha
ggplot(data.frame(alpha = samples_matrix[, "alpha"]), aes(x = alpha)) +
  geom_histogram(aes(y=after_stat(density)), bins = 30, fill = 'lightgray', color = 'black', alpha = 0.7) +
  geom_density(color = 'blue') +
  labs(title = 'Density of Alpha', x = 'Alpha', y = 'Density') +
  theme_minimal(base_size = 12)


################ - Q6 - JAGS with new prior #########################

# DATA
#HospitalVisits <- read.delim("HospitalVisits.txt")
#y <- HospitalVisits$y          # Target variable (number of visits)
#x1 <- HospitalVisits$age       # Age of individuals
#x2 <- HospitalVisits$chronic   # Binary indicator of chronic disease
#hospital <- HospitalVisits$hospital  # Hospital ID for each individual
#G <- 30                        # Number of hospitals

#Model function for JAGS
set.seed(123) #For reproducibility
ModelQ6 <- function() {
  for (i in 1:N) {
    y[i] ~ dpois(mu[i])
    mu[i] <- exp(z[hospital[i]] + beta0 + beta1 * x1[i] + beta2 * x2[i])
  }
  
  # Priors for regression coefficients
  beta0 ~ dnorm(0, 0.01)
  beta1 ~ dnorm(0, 0.01)
  beta2 ~ dnorm(0, 0.01)
  
  # Random intercepts per hospital
  for (g in 1:G) {
    z[g] ~ dnorm(0, tau)
  }
  
  # Hyperprior on tau
  tau ~ dgamma(0.01, 0.01)
}

# Save in a bug file
model.file <- "question6_model.bug"
write.model(ModelQ6, model.file)

#Data for JAGS
mydata <- list(
  y = y,
  x1 = x1,
  x2 = x2,
  hospital = hospital,
  N = length(y),
  G = 30
)

# Initializing values

myinits <- vector("list", 5) #with posterior mode for beta's
for (i in 1:5) {
  start_i <- starting_values(i)  
  myinits[[i]] <- list(
    beta0 = start_i$beta[1],
    beta1 = start_i$beta[2],
    beta2 = start_i$beta[3],
    tau = 1,
    z = rep(0, mydata$G)
  )
}


set.seed(123)
#JAGS model construction
jags_model <- jags.model(
  file = model.file,
  data = mydata,
  inits = myinits,
  n.chains = 5,  n.adapt = 5000 
)
set.seed(123)
update(jags_model, n.iter = 5000)


set.seed(123)
#Sampling posterior
samples <- coda.samples(
  model = jags_model,
  variable.names = c("beta0", "beta1", "beta2", "tau", "z"),
  n.iter = 45000
)

#Summary posterior distribution
print(summary(samples))

#Traceplots for Beta's and Tau

coda :: traceplot ( samples[, c("beta0", "beta1", "beta2", "tau")] )

#Gelman-Rubin R
gelman.diag (samples )


obj1 = samples [[ 1 ]]
obj2 = samples [[ 2 ]]
obj3 = samples [[ 3 ]]
obj4 = samples [[ 4 ]]
obj5 = samples [[ 5 ]]

#Geweke
geweke.diag ( samples )


# ESS measures
combined_chain = rbind ( obj1 , obj2 , obj3, obj4, obj5 )
effectiveSize(combined_chain)

#HPD Intervals
combined_mcmc <- coda::mcmc(combined_chain)
HPD_ints <- HPDinterval(combined_mcmc, prob = 0.95)
print(HPD_ints)

#Credible intervals for v_g
z_cols <- colnames(combined_mcmc[, 5:34])
z_samples <- combined_mcmc[, z_cols]
v_samples <- exp(as.matrix(z_samples))

post_credint_v <- apply(v_samples, 2, quantile, probs = c(0.025, 0.5, 0.975))
median_global_v <- median(as.vector(v_samples))

par(mfrow = c(1,1))
plot(1:G, xlim = range(post_credint_v), ylim = c(0.5, G+0.5), type = "n",
     yaxt = "n", ylab = "Hospital index", main = "CI 95% for hospital random effects (JAGS)")
axis(2, at = 1:G, labels = 1:G)
for (g in 1:G) {
  lines(post_credint_v[c(1,3), g], rep(g, 2), lwd = 2)
  points(post_credint_v[2, g], g, pch = 20)
}
abline(v = median_global_v, col = "red", lwd = 2, lty = 2)

# Density + Histogram of tau
ggplot(data.frame(tau = combined_mcmc[, 4]), aes(x = combined_mcmc[, 4])) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = 'lightgray', color = 'black', alpha = 0.7) +
  geom_density(color = 'blue') +
  labs(title = 'Density of Tau', x = 'Tau', y = 'Density') +
  theme_minimal(base_size = 12)

#Predictions for new model

b0_samples <- combined_mcmc[, "beta0"]
b1_samples <- combined_mcmc[, "beta1"]
b2_samples <- combined_mcmc[, "beta2"]

n <- length(y)
n_iter <- 5000
y_rep_matrix <- matrix(NA, nrow = n_iter, ncol = n)
set.seed(123)
for (m in 1:n_iter) {
  beta0 <- b0_samples[m]
  beta1 <- b1_samples[m]
  beta2 <- b2_samples[m]
  z_m <- as.numeric(z_samples[m, ])
  
  mu <- exp(z_m[hospital] + beta0 + beta1 * x1 + beta2 * x2)
  y_rep_matrix[m, ] <- rpois(n, lambda = mu)
}



y_rep_vec <- as.vector(y_rep_matrix) #Obtain the values in a vector



#Combined histograms
set.seed(123) #Set seed to reproduce results
y_rep_sample <- sample(y_rep_vec, size = 10000, replace = FALSE) #Sufficent number of values to be precise, considering all values is computationally expensive

df_obs <- data.frame(valeur = y, type_element = "Observed") #df with y
df_pred <- data.frame(valeur = y_rep_sample, type_element = "Predicted") #df with predicted y's
df_combined <- rbind(df_obs, df_pred)
ggplot() +
  geom_histogram(data = df_obs, aes(x = valeur, y=after_stat(density), fill = type_element),
                 binwidth = 1, boundary = -0.5, alpha = 0.6, color = "black") +
  geom_histogram(data = df_pred, aes(x = valeur, y=after_stat(density), fill = type_element),
                 binwidth = 1, boundary = -0.5, alpha = 0.6, color = "black") +
  scale_fill_manual(name = "Data type", values = c(
    "Observed" = "skyblue",
    "Predicted" = "salmon"
  )) +
  labs(
    x = "Number of visits",
    y = "Density"
  ) +
  theme_minimal()

#Individual plots
df_obs_ind <- data.frame(valeur = y, type = "Observed")
df_pred_ind <- data.frame(valeur = y_rep_sample, type = "Predicted")
df_combined_ind <- rbind(df_obs_ind, df_pred_ind)

ggplot(df_combined_ind, aes(x = valeur, fill = type)) +
  geom_histogram(aes(y=after_stat(density)), binwidth = 1, boundary = -0.5, color = "black", alpha = 0.8) +
  facet_wrap(~ type, ncol = 1) +
  scale_fill_manual(values = c("Observed" = "steelblue", "Predicted" = "tomato")) +
  theme_minimal() +
  labs(,
       x = "Number of visits",
       y = "Density"
  )
#Summary statistics for observed vs. predicted 
y_bar <- mean(y)
y_pred_bar <- mean(y_rep_sample)
sd_y <- sd(y)
sd_y_pred <- sd(y_rep_sample)
Q1_y <- quantile(y, 0.25, na.rm = TRUE)
Q3_y<- quantile(y, 0.75, na.rm = TRUE)
Q1_y_pred <- quantile(y_rep_sample, 0.25, na.rm = TRUE)
Q3_y_pred<- quantile(y_rep_sample, 0.75, na.rm = TRUE)
median_y <- median(y, na.rm = TRUE)
median_y_pred <- median(y_rep_sample, na.rm = TRUE)
IQR_y <- Q3_y - Q1_y
IQR_y_pred <- Q3_y_pred - Q1_y_pred

mean(v_samples) #posterior mean for v_g
median(v_samples) #median
min(v_samples)
max(v_samples)
