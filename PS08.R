library(tidyverse)
library(caret)

# Package for easy timing in R
library(tictoc)


# Get data ----------------------------------------------------------------
# Accelerometer Biometric Competition Kaggle competition data
# https://www.kaggle.com/c/accelerometer-biometric-competition/data
train <- read_csv("~/Downloads/train.csv")

# YOOGE!
dim(train)



# knn modeling ------------------------------------------------------------
model_formula <- as.formula(Device ~ X + Y + Z)



# Time knn here -----------------------------------------------------------
# I sample uniformly from all possible values of k and n. 

n <- c(3e+06, 6e+06, 9e+06, 12e+06, 15e+06, 18e+06, 21e+06, 24e+06, 27e+06, nrow(train))
k <- n

runtime <- expand.grid(n, k) %>%
  as_tibble() %>%
  rename(n = Var1, k = Var2)

time <- c()
m <- 1
i <- 1

while(i <= length(n)){
  data <- train[1:n[i],]
  j <- 1
  while(j <= length(k)){
    tic()
    caret::knn3(model_formula, data = data, k = k[j])
    timer_info <- toc()
    time[m] <- timer_info$toc - timer_info$tic
    m = m + 1
    j = j + 1
  }
  i = i + 1
}
  
runtime <- cbind(runtime, time)

# Plot your results ---------------------------------------------------------

#k
runtime_plot_k <- ggplot(runtime, aes(x = k, y = time, col = n)) +
  geom_jitter() + geom_smooth(colour = "black", alpha = 0.5, se = FALSE) +
  labs(title = "Number of nearest neighbors considered vs runtime", y = "runtime (seconds)")
runtime_plot_k

ggsave(filename="anthony_rentsch_k.png", width=16, height = 9)

#n
runtime_plot_n <- ggplot(runtime, aes(x = n, y = time, col = k)) +
  geom_jitter() + geom_smooth(aes(group = k, col = k), alpha = 0.5, se = FALSE) +
  labs(title = "Size of training set vs runtime", subtitle = "Lines included for all values of k", y = "runtime (seconds)")
runtime_plot_n

ggsave(filename="anthony_rentsch_n.png", width=16, height = 9)

#d
runtime_plot_d <- ggplot(runtime, aes(x = 3, y = time)) +
  geom_point(aes(col = k)) + 
  labs(title = "Number of predictors vs runtime", y = "runtime (seconds)", x = "Number of predictors")
runtime_plot_d

ggsave(filename="anthony_rentsch_d.png", width=16, height = 9)

# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:

# n: number of points in training set
# Answer: O(c)
# The runtime of the kNN model does not increase as the size of training set increases, 
# as long as the parameter k is held still.


# k: number of neighbors to consider
# Answer: O(k^2)
# The runtime of the kNN model increases exponentially as the parameter k increases, 
# meaning that the value of k has a big impact on the model's algorithmic complexity.  


# d: number of predictors used? In this case d is fixed at 3
# Answer: O(c)
# Hard to tell since we only consider the 3 predictor case in this problem set, but it would
# seem that the runtime does not increase as d increases, as the algorithmic complexity is determined
# more by how many computations it is forced to do (i.e., how many k neighbors to compare to) rather
# than the number of points n to train the model on or the number of predictors d to build the
# predictor space. 


