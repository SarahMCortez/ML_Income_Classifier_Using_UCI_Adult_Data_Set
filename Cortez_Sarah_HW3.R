##################################################
# ECON 418-518 Homework 3
# Sarah Cortez
# The University of Arizona
# scorte7056@arizona.edu 
# 5 December 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table)
install.packages("readr")
library(readr)
dt<- fread("data.csv")
dt<- as.data.table(data)


# Set sead# Set seadatad
set.seed(418518)


#####################
# Problem 1
#####################
# Question (i)
#################


##############
# Part (a)
##############

# Code

#data table
View(data)

head(dt)

names(dt)

##############
# Part (b)
##############

# Code

#dropping columns
dt[, c("fnlwgt", "occupation", "relationship", 
       "capital-gain", "capital-loss", "educational-num") := NULL]


#################
# Question (ii)
#################


##############
# Part (a)
##############
# Code

#Convert "income" to binary
dt[, income := ifelse(income == ">50K", 1, 0)]

##############
# Part (b)
##############

# Convert "race" to binary
dt[, race := ifelse(race == "White", 1, 0)]

##############
# Part (c)
##############

# Convert "gender" to binary 
dt[, gender := ifelse(gender == "Male", 1, 0)]

##############
# Part (d)
##############

# Convert "workclass" to binary 
dt[, workclass := ifelse(workclass == "Private", 1, 0)]

##############
# Part (e)
##############


# Convert "native-country" to binary
dt[, `native-country` := ifelse(`native-country` == "United-States", 1, 0)]

##############
# Part (f)
##############

# Convert "marital status" to binary 
dt[, `marital-status` := ifelse(`marital-status` == "Married-civ-spouse", 1, 0)]

##############
# Part (g)
##############

# Convert "education" to binary
dt[, education := ifelse(education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)]

##############
# Part (h)
##############

# Create "age_sq" as age squared
dt[, age_sq := age^2]

##############
# Part (i)
##############

# Standardize "age", "age_sq", and "hours_per_week"
dt[, age_std := (age - mean(age)) / sd(age)]
dt[, age_sq_std := (age_sq - mean(age_sq)) / sd(age_sq)]
dt[, `hours-per-week_std` := (`hours-per-week` - mean(`hours-per-week`)) / sd(`hours-per-week`)]



#################
# Question (iii)
#################


##############
# Part (a)
##############
# Code

# Calculate the proportion of individuals with income > 50K
prop_income_gt_50k <- dt[, mean(income)]
prop_income_gt_50k

##############
# Part (b)
##############
# Code

# Calculate the proportion of individuals in the private sector
prop_private_sector <- dt[, mean(workclass)]
prop_private_sector

##############
# Part (c)
##############

# Calculate the proportion of married individuals
prop_married <- dt[, mean(`marital-status`)]
prop_married

##############
# Part (d)
##############

# Calculate the proportion of females
prop_females <- dt[, mean(1 - gender)]
prop_females

##############
# Part (e)
##############

# Total number of rows
total_rows <- nrow(dt)

# Total number of NAs in the dataset
total_NAs <- dt[, sum(is.na(.SD))]

# Total observations with non-missing values
total_non_missing <- total_rows * ncol(dt) - total_NAs

list(total_rows = total_rows, total_NAs = total_NAs, total_non_missing = total_non_missing)

##############
# Part (f)
##############

# Convert "income" to a factor
dt[, income := factor(income, levels = c(0, 1), labels = c("<=50K", ">50K"))]


##############
# Question (iv)
##############

##############
# Part (a)
##############

# Find the last training set observation index (70% of the data)
train_index <- floor(nrow(dt) * 0.70)


##############
# Part (b)
##############

# Create the training data table (first 70% of rows)
train_dt <- dt[1:train_index]


##############
# Part (c)
##############

# Create the testing data table (observations after the last training observation)
test_dt <- dt[(train_index + 1):nrow(dt)]

##############
# Question (v)
##############

##############
# Part (b)
##############
# Install and load necessary libraries (if not already installed)
install.packages("caret")
install.packages("glmnet")

library(caret)
library(glmnet)


# Define the grid of lambda values
lambda_grid <- 10^(seq(2, -2, length = 50))

# Train a lasso regression model with 10-fold cross-validation
set.seed(418518)
lasso_model <- train(
  income ~ ., 
  data = train_dt, 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)
)

# Example: train the model (adjust this based on your actual model setup)
cv <- train(income ~ ., data = train_dt, method = "glmnet", 
            tuneGrid = expand.grid(alpha = 1, lambda = seq(10^5, 10^-2, length = 50)), 
            trControl = trainControl(method = "cv", number = 10))

##############
# Part (c)
##############

# Best value of lambda and classification accuracy

best_lambda <- lasso_model$bestTune$lambda
classification_accuracy <- max(lasso_model$results$Accuracy)

cat("Best lambda for Lasso:", best_lambda, "\n")
cat("Classification accuracy for Lasso:", classification_accuracy, "\n")


##############
# Part (d)
##############
# Variables with coefficients approximately zero

lasso_coefficients <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
zero_coeff_vars <- rownames(lasso_coefficients)[abs(as.vector(lasso_coefficients)) < 1e-4]

cat("Variables with zero coefficients in Lasso:", zero_coeff_vars, "\n")


##############
# Part (e)
##############

# Estimate Lasso and Ridge regression models with only non-zero coefficient variables

# Filter out non-zero variables from training data
non_zero_vars <- setdiff(colnames(train_dt), zero_coeff_vars)
filtered_training_data <- training_data[, c(non_zero_vars, "income")]

# Lasso Model with non-zero variables
set.seed(418518)
lasso_refined <- train(
  income ~ ., 
  data = filtered_training_data, 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_grid)
)

# Ridge Model with non-zero variables
set.seed(418518)
ridge_refined <- train(
  income ~ ., 
  data = filtered_training_data, 
  method = "glmnet",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_grid)
)

# Compare Classification Accuracies
lasso_accuracy <- max(lasso_refined$results$Accuracy)
ridge_accuracy <- max(ridge_refined$results$Accuracy)

cat("Lasso Classification Accuracy:", lasso_accuracy, "\n")
cat("Ridge Classification Accuracy:", ridge_accuracy, "\n")


##############
# Question (vi)
##############

install.packages("randomForest")

##############
# Part (b)
##############

cat("Starting Random Forest Model Evaluation...\n")

# Define the grid for tuning mtry (number of features to try at each split)
tree_grid <- expand.grid(mtry = c(2, 5, 9))  # Adjust as needed based on features

# Train Random Forest models with 5-fold cross-validation and different number of trees
rf_model_100 <- train(
  income ~ ., 
  data = train_dt, 
  method = "rf", 
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE), 
  tuneGrid = tree_grid, 
  ntree = 100  # 100 trees
)

rf_model_200 <- train(
  income ~ ., 
  data = train_dt, 
  method = "rf", 
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE), 
  tuneGrid = tree_grid, 
  ntree = 200  # 200 trees
)

rf_model_300 <- train(
  income ~ ., 
  data = train_dt, 
  method = "rf", 
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE), 
  tuneGrid = tree_grid, 
  ntree = 300  # 300 trees
)
cat("Random Forest Models Complete!\n")

##############
# Part (c)
##############

cat("Evaluating Model Accuracy...\n")

# Print the results to compare the models' accuracy
cat("Model 100 trees:", rf_model_100$results$Accuracy, "\n")
cat("Model 200 trees:", rf_model_200$results$Accuracy, "\n")
cat("Model 300 trees:", rf_model_300$results$Accuracy, "\n")

# Find the best model based on accuracy
best_rf_model <- max(rf_model_100$results$Accuracy, rf_model_200$results$Accuracy, rf_model_300$results$Accuracy)
cat("Best Model Accuracy:", best_rf_model, "\n")


##############
# Part (d)
##############

cat("Comparing Best Models...\n")

# Assuming you have the best Random Forest model accuracy from Part (vi) saved
best_rf_model <- 0.8570592  
# Define the best accuracy for the Lasso/Ridge model (from Part (v))
# Assuming 'cv' is your lasso/ridge model training object, extract the best accuracy:
best_lasso_ridge_accuracy <- max(cv$results$Accuracy)  

# Print comparison
cat("Best Random Forest Model Accuracy:", best_rf_model, "\n")
cat("Best Lasso/Ridge Model Accuracy from Part (v):", best_lasso_ridge_accuracy, "\n")
##############

##############
# Part (e)
##############

cat("Generating Confusion Matrix...\n")

# Make predictions using the best random forest model 
predictions_rf <- predict(rf_model_100, training_data)

# Generate the confusion matrix
cm_rf <- confusionMatrix(predictions_rf, training_data$income)

# Print confusion matrix
print(cm_rf)

# Check false positives and false negatives
cat("False Positives:", cm_rf$table[2,1], "\n")
cat("False Negatives:", cm_rf$table[1,2], "\n")


##############
# Question (vii)
##############
#code

cat("Evaluating Best Model on Testing Data...\n")

# Make predictions using the best random forest model (example: rf_model_100)
predictions_rf_test <- predict(rf_model_100, testing_data)

# Evaluate the classification accuracy on the testing set
test_accuracy_rf <- mean(predictions_rf_test == testing_data$income)
cat("Classification Accuracy on Testing Set:", test_accuracy_rf, "\n")




