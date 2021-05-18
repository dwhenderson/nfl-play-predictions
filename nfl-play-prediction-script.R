library(nflfastR)
library(tidyverse)
library(lubridate)
library(caret)
library(stringr)
library(ggthemes)
library(randomForest)
library(rpart)
library(gam)
library(knitr)
library(reshape2)

# Set global seed
global_seed <- 42

# Load data
future::plan("multisession")
game_years <- 2019:2020
pbp_original <- load_pbp(game_years)
col_index <- c(3,6,8,10,12,15,17,22,26,28,29,31,32,34,35,36,54,55,60,284,294,296,330,332,333)

master <- pbp_original[,col_index] %>%
  filter(season == 2019 & season_type == "REG") %>%
  select(-season, -season_type) %>%
  filter(play_type %in% c("pass", "run")) %>%
  mutate(posteam = factor(posteam),
         defteam = factor(defteam, levels = levels(posteam))) %>%
  rename(second_half = game_half) %>%
  mutate(second_half = ifelse(second_half == "Half2", 1, 0)) %>%
  filter(qb_kneel == 0 & qb_spike == 0) %>%
  select(-qb_kneel, -qb_spike) %>%
  mutate(play_type = case_when(
    play_type == "run" & qb_scramble == 1 ~ "pass",
    TRUE ~ play_type
    )) %>%
  mutate(pass_attempt = ifelse(play_type == "pass", 1, 0)) %>%
  select(-play_type, -qb_scramble) %>%
  mutate(play_clock = as.numeric(play_clock)) %>%
  filter(play_clock %in% 0:40) %>%
  select(old_game_id, desc, defteam, posteam, pass_attempt, down, ydstogo, yardline_100, second_half, half_seconds_remaining,
         play_clock, score_differential, defteam_timeouts_remaining, posteam_timeouts_remaining, shotgun, no_huddle) %>%
  na.omit()


# Now we create a partition for the final validation set
set.seed(global_seed, sample.kind = "Rounding")
validation_ind <- createDataPartition(master$pass_attempt, times = 1, p = 0.15, list = FALSE)
validation <- master[validation_ind,] %>% na.omit()
dat <- master[-validation_ind,] %>% na.omit()
rm(validation_ind)


# Then partition the remaining data into training and test sets
set.seed(global_seed, sample.kind = "Rounding")
test_ind <- createDataPartition(master$pass_attempt, times = 1, p = 0.15, list = FALSE)
dat_test <- dat[test_ind,] %>% na.omit()
dat_train <- dat[-test_ind,] %>% na.omit()
rm(test_ind)


# Setting parameters to improve speed of cross validation
k_folds <- 10
control = trainControl(method = "cv", number = k_folds, p = 0.9)

### BUILDING THE MODELS

# Model #1: guessing based on overall pass/run frequencies
pass_frequency <- mean(dat_train$pass_attempt == 1)

phat_naive <- rep(pass_frequency, nrow(dat_test))

rmse_naive <- RMSE(phat_naive, dat_test$pass_attempt)


# Model #2a: logistic regression - down + distance + shotgun
fit_logit_downdist <- glm(pass_attempt ~ down + ydstogo + shotgun, data = dat_train, family = "binomial")
                            
phat_logit_downdist <- predict(fit_logit_downdist, dat_test, type = "response")

rmse_logit_downdist <- RMSE(phat_logit_downdist, dat_test$pass_attempt)


# Model #2b: local weighted regression - down + distance + shotgun
fit_loess_downdist <- train(factor(pass_attempt, levels = 1:0) ~ down + ydstogo + shotgun, data = dat_train, method = "gamLoess",
                  tuneGrid = data.frame(span = seq(0.05, 0.55, 0.1), degree = 1))

phat_loess_downdist <- predict(fit_loess_downdist, dat_test, type = "prob") %>% as.data.frame() %>% pull(`1`)

rmse_loess_downdist <- RMSE(phat_loess_downdist, dat_test$pass_attempt)

####################################################################################
# Model #3: KNN 
# Now we use a more sophisticated approach to predict outcomes taking into account all the variables
fit_knn <- train(pass_attempt ~ down + ydstogo + half_seconds_remaining + score_differential + shotgun + second_half,
                 data = dat_train, method = "knn", tuneGrid = data.frame(k = seq(20,100,20)), preProcess = c("center", "scale"),
                 trControl = control)

phat_knn <- predict(fit_knn, dat_test)

rmse_knn <- RMSE(phat_knn, dat_test$pass_attempt)

# Model #4: regression tree
minsplit_values <- seq(2, 50, len = 20)
cp_values <- seq(0, 0.01, len = 10)

rmse_minsplit_cp <- sapply(minsplit_values, function(ms) {
  train(x = dat_train[,c(3:4,6:16)], y = dat_train$pass_attempt, method = "rpart",
        tuneGrid = data.frame(cp = cp_values), control = rpart.control(minsplit = ms), trControl = control)$results[["RMSE"]]
})

best_cp <- cp_values[which(rmse_minsplit_cp == min(rmse_minsplit_cp), arr.ind = TRUE)[,1]]
best_minsplit <- minsplit_values[which(rmse_minsplit_cp == min(rmse_minsplit_cp), arr.ind = TRUE)[,2]]

fit_rpart <- rpart(pass_attempt ~ defteam + posteam + down + ydstogo + yardline_100 + second_half + half_seconds_remaining + play_clock + 
                     score_differential + defteam_timeouts_remaining + posteam_timeouts_remaining + shotgun + no_huddle,
                   data = dat_train, control = rpart.control(minsplit = best_minsplit, cp = best_cp))

phat_rpart <- predict(fit_rpart, dat_test)

rmse_rpart <- RMSE(phat_rpart, dat_test$pass_attempt)

# Model #5: randomForest 
# First, we will try out different nodesizes and select the one resulting in the lowest RMSE
rf_nodesize = seq(50,150,25)

rmse_nodesize <- sapply(rf_nodesize, function(ns){
  train(x = dat_train[,c(3:4,6:16)], y = dat_train$pass_attempt, method = "rf", 
        nodesize = ns, tuneGrid = data.frame(mtry = 2:5), trControl = control, ntree = 300)$results[["RMSE"]]
})
best_nodesize <- rf_nodesize[which(rmse_nodesize == min(rmse_nodesize), arr.ind = TRUE)[2]]
best_mtry <- c(2:5)[which(rmse_nodesize == min(rmse_nodesize), arr.ind = TRUE)[1]]

fit_rf <- randomForest(x = dat_train[,c(3:4,6:16)], y = dat_train$pass_attempt, nodesize = best_nodesize, mtry = best_mtry,
                       ntree = 300)

phat_rf <- predict(fit_rf, dat_test)

rmse_rf <- RMSE(phat_rf, dat_test$pass_attempt)

# Summary of test set predictions and accuracy
test_results_df <- data.frame(model = c("naive", "logit_downdist", "loess_downdist", "knn", "rpart", "rf"),
                              RMSE = c(rmse_naive, rmse_logit_downdist, rmse_loess_downdist, rmse_knn, rmse_rpart, rmse_rf))


##################################################################################################
# Re-training on full dataset, then making the validation predictions

# Model #1: naive baseline
rmse_naive_validation <- RMSE(rep(pass_frequency, nrow(validation)), validation$pass_attempt)

# Model #2
fit_logit_downdist_final <- glm(pass_attempt ~ down + ydstogo + shotgun, data = dat, family = "binomial")
phat_logit_downdist_validation <- predict(fit_logit_downdist_final, validation, type = "response")
rmse_logit_downdist_validation <- RMSE(phat_logit_downdist_validation, validation$pass_attempt)


# Model #2b
fit_loess_downdist_final <- train(factor(pass_attempt, levels = 1:0) ~ down + ydstogo + shotgun, data = dat, method = "gamLoess",
                                  tuneGrid = data.frame(span = seq(0.05, 0.55, 0.1), degree = 1))
phat_loess_downdist_validation <- predict(fit_loess_downdist_final, validation, type = "prob") %>% as.data.frame() %>% pull(`1`)
rmse_loess_downdist_validation <- RMSE(phat_loess_downdist_validation, validation$pass_attempt)

# Model #3 
fit_knn_final <- train(pass_attempt ~ down + ydstogo + half_seconds_remaining + score_differential + shotgun + second_half,
                       data = dat, method = "knn", tuneGrid = data.frame(k = seq(20,140,20)), preProcess = c("center", "scale"),
                       trControl = control)
phat_knn_validation <- predict(fit_knn_final, validation)
rmse_knn_validation <- RMSE(phat_knn_validation, validation$pass_attempt)

# Model #4: regression tree
fit_rpart_final <- rpart(pass_attempt ~ defteam + posteam + down + ydstogo + yardline_100 + second_half + half_seconds_remaining + play_clock + 
                           score_differential + defteam_timeouts_remaining + posteam_timeouts_remaining + shotgun + no_huddle,
                         data = dat, control = rpart.control(minsplit = best_minsplit, cp = best_cp))
phat_rpart_validation <- predict(fit_rpart_final, validation)
rmse_rpart_validation <- RMSE(phat_rpart_validation, validation$pass_attempt)

# Model #5: random forest
fit_rf_final <- randomForest(x = dat[,c(3:4,6:16)], y = dat$pass_attempt, nodesize = best_nodesize, mtry = best_mtry,
                             ntree = 300)
phat_rf_validation <- predict(fit_rf_final, validation)
rmse_rf_validation <- RMSE(phat_rf_validation, validation$pass_attempt)

# SUMMARY OF FINAL PREDICTION RESULTS
validation_results_summary <- data.frame(model = c("naive", "logit_downdist", "loess_downdist", "knn", "rpart", "rf"),
                                         RMSE = c(rmse_naive_validation, rmse_logit_downdist_validation, rmse_loess_downdist_validation, 
                                                  rmse_knn_validation, rmse_rpart_validation, rmse_rf_validation))

print(validation_results_summary)
