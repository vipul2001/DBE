# Load necessary libraries
library(CHMM)

# Load data (adjust paths and number of files as necessary)
output_dir <- "data_csv"
num_files <- length(list.files(output_dir)) / 2  # Adjust if the number of files differs

# Initialize lists to store the data
observations_o <- list()
observations_o1 <- list()


# Load CSV files
for (i in 1:num_files) {
  obs_o <- read.csv(file.path(output_dir, paste0("observations_o_", i - 1, ".csv"))) 
  obs_o1 <- read.csv(file.path(output_dir, paste0("observations_o1_", i - 1, ".csv")))
 
  observations_o[[i]] <- obs_o$observations_o
  observations_o1[[i]] <- obs_o1$observations_o1

}

# Combine data into vectors for each observation and true state
obs_o_combined <- unlist(observations_o)
obs_o1_combined <- unlist(observations_o1)


# Create data for CHMM
data_combined <- data.frame(obs_o_combined, obs_o1_combined)

# Set the number of hidden states (e.g., 4 states)
n_states <- 6

# Initialize vectors to store results
log_likelihoods <- numeric(10)
accuracies_o <- numeric(10)
accuracies_o1 <- numeric(10)

# Run the CHMM model 10 times and store the results
for (iteration in 1:10) {
  # Randomly initialize omega (sum of 4 elements = 1)
  omega <- runif(10)
  
  #omega <- omega / sum(omega)  # Normalize so that the sum is 1
  
  # Create and fit the Coupled HMM model
  model <- coupledHMM(
    X = data_combined,       # combined observations
    nb.states = n_states,    # number of hidden states
    S = cor(data_combined)
    ,  # correlation matrix
    exact=TRUE,
    omega = omega, 
    viterbi=TRUE,# Omega values for the states
    meth.init = "kmeans",    # initialization method
    var.equal = FALSE,        # Equal variance assumption
    itmax = 100,             # Max iterations for EM algorithm
    threshold = 1e-03        # Stopping threshold for EM
  )
  print(model$omega)
  model<-model$model
  
  
  # Access the results from the CHMM_EM output
  posterior_probs <- model$postPr  # Posterior probabilities for each state
  
  # Calculate log-likelihood
  log_likelihoods[iteration] <- model$loglik / 100
  cat("Iteration ", iteration, " Log-Likelihood: ", log_likelihoods[iteration], "\n")
  
}

# Calculate average and standard deviation for log-likelihood and accuracies
avg_log_likelihood <- mean(log_likelihoods)
std_log_likelihood <- sd(log_likelihoods)




# Output the results
cat("\nAverage Log-Likelihood: ", avg_log_likelihood, "\n")
cat("Standard Deviation of Log-Likelihood: ", std_log_likelihood, "\n")