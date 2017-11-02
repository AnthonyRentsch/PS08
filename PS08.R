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
# I sample uniformly on k exists on [1,20] and on n exists on [1,000,000, 10,000,000].

n <- c(1e+06, 2e+06, 3e+06, 4e+06, 5e+06, 6e+06, 7e+06, 8e+06, 9e+06, 10e+6)
k <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)

runtime <- expand.grid(k, n) %>%
  as_tibble() %>%
  rename(k = Var1, n = Var2)

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

runtime_plot <- ggplot(runtime, aes(x = n, y = time, col = k)) +
  geom_point() + geom_smooth(colour = "black", alpha = 0.5, se = FALSE) +
  labs(title = "Number of nearest neighbors considered vs runtime", y = "runtime (seconds)")
runtime_plot

ggsave(filename="anthony_rentsch.png", width=16, height = 9)


# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:

# n: number of points in training set
# k: number of neighbors to consider
# d: number of predictors used? In this case d is fixed at 3

# Answer:
# f(n) = 0(d(n^2 + k))


