






# Install BiocManager if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

# Install ComplexHeatmap
BiocManager::install("ComplexHeatmap")





# Load necessary package
if(!require(mice)) install.packages("mice")
library(mice)




#### 0. Required Packages ####
install.packages(c("tidyverse","caret","Boruta","smotefamily","randomForest",
                  "rpart","e1071","xgboost","nnet","pROC","mccr"))


library(tidyverse)
library(caret)
library(Boruta)
library(smotefamily)       # for SMOTE
library(pROC)       # for AUROC
library(mccr)       # for MCC





library(tidyverse)
library(caret)
library(Boruta)
library(smotefamily)  # for SMOTE
library(randomForest)
library(rpart)
library(e1071)
library(xgboost)
library(nnet)
library(pROC)
library(mccr)



library(haven)
library(tidyverse)
library(caret)
library(Boruta)
library(smotefamily)  # for SMOTE
library(randomForest)
library(rpart)
library(e1071)
library(xgboost)
library(nnet)
library(pROC)
library(mccr)
library(ComplexHeatmap)
library(circlize)


#### 1. Load Data ####
data <- read_dta("D:/Research/BDHS Research/ML Pregnancy loss/update/data/bdhs_ml_dataset.dta")


colnames(data)


table(data$preg_loss_cat, useNA = "ifany")



# Outcome as factor
outcome <- "preg_loss_cat"
data[[outcome]] <- factor(data[[outcome]],
                          levels = c(1,2),
                          labels = c("Single", "Multiple"))

# Check
table(data[[outcome]], useNA = "ifany")
colSums(is.na(data))




missing_percent <- colSums(is.na(data)) / nrow(data) * 100
round(missing_percent, 2)




#### 2. Missing Value Imputation ####

library(mice)

# Separate numeric and factor variables
num_vars <- names(data)[sapply(data, is.numeric)]
fact_vars <- names(data)[sapply(data, is.factor)]

# --- Quick median/mode imputation (simple approach) ---
# Median for numeric
for(v in num_vars){
  if(any(is.na(data[[v]]))){
    data[[v]][is.na(data[[v]])] <- median(data[[v]], na.rm = TRUE)
  }
}

# Mode for factor
getmode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
for(v in fact_vars){
  if(any(is.na(data[[v]]))){
    data[[v]][is.na(data[[v]])] <- getmode(data[[v]])
  }
}

# Check missing values
cat("Missing values after simple imputation:\n")
print(colSums(is.na(data)))

# --- Optional: MICE for more robust imputation ---
# Define methods for each variable type
method_vec <- sapply(data, function(x){
  if(is.numeric(x)) "pmm"       # predictive mean matching
  else if(is.factor(x) & length(levels(x)) == 2) "logreg"  # binary factor
  else if(is.factor(x)) "polyreg"  # categorical factor >2 levels
  else ""  # default
})

# Run MICE (5 imputations, reproducible)
imp <- mice(data, m = 5, method = method_vec, seed = 123)

# Complete data using first imputed dataset
data_imputed <- complete(imp, 1)

# Check again
cat("Missing values after MICE imputation:\n")
print(colSums(is.na(data_imputed)))





str(data_imputed)
sapply(data_imputed, class)
glimpse(data_imputed)







#### Categorical Encoding ####



library(dplyr)

data_ml <- data_imputed %>%
  mutate(
    age_cat = as.factor(age_cat),
    religion = as.factor(religion),
    hh_relation = as.factor(hh_relation),
    hh_age = as.factor(hh_age),
    children = as.factor(children),
    living_child = as.factor(living_child),
    preg_order = as.factor(preg_order),
    h_desire = as.factor(h_desire),
    contracept_dec = as.factor(contracept_dec),
    menstruated = as.factor(menstruated),
    pres_preg = as.factor(pres_preg),
    phy_affair = as.factor(phy_affair),
    abstaining = as.factor(abstaining),
    residing = as.factor(residing),
    empowerment = as.factor(empowerment),
    occupation = as.factor(occupation),
    hh_occupation = as.factor(hh_occupation),
    cohab = as.factor(cohab),
    care_roster = as.factor(care_roster),
    preterm = as.factor(preterm),
    hh_materials = as.factor(hh_materials),
    newspaper = as.factor(newspaper),
    media_radio = as.factor(media_radio),
    media_TV = as.factor(media_TV),
    phone_owner = as.factor(phone_owner),
    toilet = as.factor(toilet),
    water = as.factor(water)
  )








library(tidymodels)
library(themis)


library(caret)
library(smotefamily)  # For SMOTE









library(tidymodels)
library(themis)


library(caret)
library(smotefamily)  # For SMOTE




library(caret)
library(Boruta)
library(smotefamily)
library(dplyr)

set.seed(123)

# ============================
# 0. Setup
# ============================
outcome <- "preg_loss_cat"
data_imputed[[outcome]] <- as.factor(data_imputed[[outcome]])

features <- setdiff(names(data_imputed), outcome)

# ============================
# 1. Train–Test Split (70–30)
# ============================
train_index <- as.vector(createDataPartition(data_imputed[[outcome]], p = 0.7, list = FALSE))

train <- data_imputed[train_index, ]
test  <- data_imputed[-train_index, ]

cat("Train:", nrow(train), "| Test:", nrow(test), "\n")

# ============================
# 2. Boruta Feature Selection
# ============================
set.seed(123)

boruta_model <- Boruta(
  x = train[, features],
  y = train[[outcome]],
  doTrace = 1
)

# Confirm important variables
boruta_final <- TentativeRoughFix(boruta_model)
boruta_selected <- getSelectedAttributes(boruta_final, withTentative = FALSE)

cat("Selected features by Boruta:\n")
print(boruta_selected)

# Keep only selected predictors
train_boruta <- train[, c(boruta_selected, outcome)]
test_boruta  <- test[,  c(boruta_selected, outcome)]






plot(boruta_final, las = 2, cex.axis = 0.7, xlab = "", main = "Boruta Feature Importance")




# ============================
# 3. SMOTE Oversampling
# ============================
set.seed(123)

X <- train_boruta[, boruta_selected]
Y <- train_boruta[[outcome]]

# Borderline-SMOTE recommended
smote_out <- SMOTE(X, Y, K = 5, dup_size = 5)

train_smote <- smote_out$data
colnames(train_smote)[ncol(train_smote)] <- outcome
train_smote[[outcome]] <- as.factor(train_smote[[outcome]])

# ============================
# 4. Class Balance Check
# ============================
cat("\nBefore SMOTE:\n")
print(table(train_boruta[[outcome]]))

cat("\nAfter SMOTE:\n")
print(table(train_smote[[outcome]]))







#### 3. Train-Test Split (70-30) #### 




set.seed(123)

# Recipe
rec <- recipe(preg_loss_cat ~ ., data = train) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>%  # factor -> dummy
  step_smote(preg_loss_cat)

# Prepare the training data
train_smote <- bake(prep(rec), new_data = NULL)

# Check class balance
table(train_smote$preg_loss_cat)





#### 3. Train-Test Split (70-30) #### 

set.seed(123)
features <- setdiff(names(data), outcome)
train_index <- createDataPartition(data[[outcome]], p = 0.7, list = FALSE)
train <- data[train_index, ]
test  <- data[-train_index, ]

#### 4. SMOTE Oversampling #### 

set.seed(123)
smote_out <- SMOTE(train[, features], train[[outcome]], K = 5)
train_smote <- smote_out$data
colnames(train_smote)[ncol(train_smote)] <- outcome
train_smote[[outcome]] <- as.factor(train_smote[[outcome]])

# Check class balance
cat("Before SMOTE:\n")
print(table(train[[outcome]]))
cat("After SMOTE:\n")
print(table(train_smote[[outcome]]))








# Advanced Borderline-SMOTE using same variable names

library(smotefamily)

set.seed(123)

# X = Boruta predictors
X <- train_boruta[, boruta_selected]

# Y = outcome
Y <- train_boruta[[outcome]]

# Borderline-SMOTE (correct function is BLSMOTE)
smote_out <- BLSMOTE(X, Y, K = 5, dupSize = 5)

# output dataset
train_smote <- smote_out$data

# rename last column as outcome
colnames(train_smote)[ncol(train_smote)] <- outcome
train_smote[[outcome]] <- as.factor(train_smote[[outcome]])

# Class distribution
cat("\nBefore SMOTE:\n")
print(table(train_boruta[[outcome]]))

cat("\nAfter Borderline-SMOTE:\n")
print(table(train_smote[[outcome]]))







# Check no missing
colSums(is.na(data_imputed))



#### Class Distribution Before vs After SMOTE (Publication Standard) ####

library(ggplot2)
library(dplyr)
library(scales)



library(ggplot2)
library(dplyr)
library(scales)

# Prepare summary data
before <- train %>%
  count(!!sym(outcome)) %>%
  mutate(Stage = "Before SMOTE", Percent = n / sum(n) * 100)

after <- train_smote %>%
  count(!!sym(outcome)) %>%
  mutate(Stage = "After SMOTE", Percent = n / sum(n) * 100)

plot_data <- bind_rows(before, after)

# Ensure consistent factor order
plot_data[[outcome]] <- factor(plot_data[[outcome]], 
                               levels = unique(before[[outcome]]))

# Elegant colors (Lancet-style palette)
smote_colors <- c("Before SMOTE" = "#4472C4",  # professional blue
                  "After SMOTE"  = "#ED7D31")  # clean orange

# Polished bar plot
ggplot(plot_data, aes_string(x = "Stage", y = "Percent", fill = "Stage")) +
  geom_bar(stat = "identity", position = "dodge", width = 0.6, color = "white", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.1f%%", Percent)),
            position = position_dodge(width = 0.6), 
            vjust = -0.4, size = 4, family = "sans") +
  facet_wrap(as.formula(paste("~", outcome)), nrow = 1, scales = "free_x") +
  scale_fill_manual(values = smote_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title = "Effect of SMOTE on Class Distribution",
    subtitle = "Before vs After synthetic oversampling of outcome classes",
    x = NULL,
    y = "Percentage of Samples",
    fill = NULL
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5, color = "#222222"),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
    axis.text = element_text(size = 11, color = "#333333"),
    axis.title.y = element_text(size = 12, face = "bold"),
    legend.position = "top",
    legend.text = element_text(size = 11),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 12, color = "black"),
    strip.background = element_rect(fill = "gray90", color = NA)
  )




library(ggplot2)
library(dplyr)
library(scales)

# Prepare summary data
before <- train_boruta %>%
  count(!!sym(outcome)) %>%
  mutate(Stage = "Before SMOTE", Percent = n / sum(n) * 100)

after <- train_smote %>%
  count(!!sym(outcome)) %>%
  mutate(Stage = "After Borderline-SMOTE", Percent = n / sum(n) * 100)

plot_data <- bind_rows(before, after)

# Ensure factor order
plot_data[[outcome]] <- factor(plot_data[[outcome]],
                               levels = c("Single", "Multiple"))

# Colors (journal-standard palette)
smote_colors <- c("Single" = "#4472C4",  # professional blue
                  "Multiple" = "#ED7D31")  # clean orange

# Stacked bar plot
ggplot(plot_data, aes(x = Stage, y = Percent, fill = !!sym(outcome))) +
  geom_bar(stat = "identity", color = "white", width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", Percent)),
            position = position_stack(vjust = 0.5), size = 4, family = "sans") +
  scale_fill_manual(values = smote_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05)),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title = "Effect of Borderline-SMOTE on Outcome Class Distribution",
    subtitle = "Before vs After Synthetic Oversampling",
    x = NULL,
    y = "Percentage of Samples",
    fill = "Outcome"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray40"),
    axis.text = element_text(size = 11),
    axis.title.y = element_text(size = 12, face = "bold"),
    legend.position = "top",
    legend.text = element_text(size = 11)
  )





library(ggplot2)
library(dplyr)
library(scales)

# Prepare summary data
before <- train_boruta %>%
  count(!!sym(outcome)) %>%
  mutate(Stage = "Before SMOTE", Percent = n / sum(n) * 100)

after <- train_smote %>%
  count(!!sym(outcome)) %>%
  mutate(Stage = "After Borderline-SMOTE", Percent = n / sum(n) * 100)

plot_data <- bind_rows(before, after)

# Factor order
plot_data[[outcome]] <- factor(plot_data[[outcome]], levels = c("Single", "Multiple"))
plot_data$Stage <- factor(plot_data$Stage, levels = c("Before SMOTE", "After Borderline-SMOTE"))

# Modern, publication-ready color palette
smote_colors <- c("Single" = "#1f77b4",   # calm blue
                  "Multiple" = "#ff7f0e") # vibrant orange

# Horizontal stacked bar plot
ggplot(plot_data, aes(x = Stage, y = Percent, fill = !!sym(outcome))) +
  geom_bar(stat = "identity", color = "white", width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", Percent)),
            position = position_stack(vjust = 0.5), size = 4, family = "sans") +
  coord_flip() +  # horizontal bars
  scale_fill_manual(values = smote_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05)),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title = "Effect of Borderline-SMOTE on Outcome Class Distribution",
    subtitle = "Comparison of Single vs Multiple Pregnancy Loss Before and After Synthetic Oversampling",
    x = NULL,
    y = "Percentage of Samples",
    fill = "Outcome"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
    axis.text = element_text(size = 12),
    axis.title.y = element_text(size = 13, face = "bold"),
    legend.position = "top",
    legend.text = element_text(size = 12),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )







library(ggplot2)
library(dplyr)
library(scales)

# Prepare summary data
before <- train_boruta %>%
  count(!!sym(outcome)) %>%
  mutate(Stage = "Before SMOTE", Percent = n / sum(n) * 100)

after <- train_smote %>%
  count(!!sym(outcome)) %>%
  mutate(Stage = "After Borderline-SMOTE", Percent = n / sum(n) * 100)

plot_data <- bind_rows(before, after)

# Factor order
plot_data[[outcome]] <- factor(plot_data[[outcome]], levels = c("Single", "Multiple"))
plot_data$Stage <- factor(plot_data$Stage, levels = c("Before SMOTE", "After Borderline-SMOTE"))

# Alternative color palette: Pink & Firoza
smote_colors <- c("Single" = "#E377C2",   # Pink
                  "Multiple" = "#17BECF") # Firoza / turquoise

# Horizontal stacked bar plot
ggplot(plot_data, aes(x = Stage, y = Percent, fill = !!sym(outcome))) +
  geom_bar(stat = "identity", color = "white", width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", Percent)),
            position = position_stack(vjust = 0.5), size = 4, family = "sans") +
  coord_flip() +  # horizontal bars
  scale_fill_manual(values = smote_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05)),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title = "Effect of Borderline-SMOTE on Outcome Class Distribution",
    subtitle = "Comparison of Single vs Multiple Pregnancy Loss Before and After Synthetic Oversampling",
    x = NULL,
    y = "Percentage of Samples",
    fill = "Outcome"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
    axis.text = element_text(size = 12),
    axis.title.y = element_text(size = 13, face = "bold"),
    legend.position = "top",
    legend.text = element_text(size = 12),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )





# ===============================
# 9. Stacked Bar Plot: Class Distribution
# ===============================
before <- train_boruta %>%
  count(!!sym(outcome)) %>%
  mutate(Stage = "Before SMOTE", Percent = n / sum(n) * 100)

after <- train_smote %>%
  count(!!sym(outcome)) %>%
  mutate(Stage = "After Borderline-SMOTE", Percent = n / sum(n) * 100)

plot_data <- bind_rows(before, after)
plot_data[[outcome]] <- factor(plot_data[[outcome]], levels = c("Single","Multiple"))
plot_data$Stage <- factor(plot_data$Stage, levels = c("Before SMOTE","After Borderline-SMOTE"))

# Polished colors: Pink & Firoza
smote_colors <- c("Single" = "#E377C2", "Multiple" = "#17BECF")

ggplot(plot_data, aes(x = Stage, y = Percent, fill = !!sym(outcome))) +
  geom_bar(stat = "identity", color = "white", width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", Percent)),
            position = position_stack(vjust = 0.5), size = 4, family = "sans") +
  coord_flip() +
  scale_fill_manual(values = smote_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05)),
                     labels = function(x) paste0(x, "%")) +
  labs(
    title = "Effect of Borderline-SMOTE on Outcome Class Distribution",
    subtitle = "Comparison of Single vs Multiple Pregnancy Loss Before and After Synthetic Oversampling",
    x = NULL,
    y = "Percentage of Samples",
    fill = "Outcome"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray40"),
    axis.text = element_text(size = 12),
    axis.title.y = element_text(size = 13, face = "bold"),
    legend.position = "top",
    legend.text = element_text(size = 12),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )













colSums(is.na(data))



library(tidyverse)
library(caret)
library(Boruta)
library(randomForest)
library(rpart)
library(e1071)
library(xgboost)
library(nnet)
library(pROC)
library(mccr)

library(ComplexHeatmap)
library(circlize)








# ===============================
# 5. Recipe: Dummy, scaling, zero-variance removal
# ===============================
rec <- recipe(as.formula(paste(outcome, "~ .")), data = train_smote) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

train_processed <- bake(prep(rec), new_data = NULL)
test_processed  <- bake(prep(rec), new_data = test_boruta)

# Map Boruta-selected variables to dummy columns
select_dummy_cols <- function(dummy_df, original_vars) {
  selected <- c()
  for(v in original_vars) {
    matched <- grep(paste0("^", v, "_"), colnames(dummy_df), value = TRUE)
    if(v %in% colnames(dummy_df)) matched <- c(matched, v)
    selected <- c(selected, matched)
  }
  return(selected)
}

selected_cols <- select_dummy_cols(train_processed, boruta_selected)
selected_cols <- c(selected_cols, outcome)
train_processed <- train_processed[, selected_cols]
test_processed  <- test_processed[, selected_cols]

train_processed[[outcome]] <- factor(train_processed[[outcome]], levels = c("Single","Multiple"))
test_processed[[outcome]]  <- factor(test_processed[[outcome]], levels = c("Single","Multiple"))





# ===============================
# 6. Cross-validation setup
# ===============================
ctrl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = "final",
  allowParallel = TRUE
)

# ===============================
# 7. Models & Hyperparameter Tuning
# ===============================
models <- c(
  "Random Forest"       = "rf",
  "Decision Tree"       = "rpart",
  "KNN"                 = "knn",
  "Logistic Lasso"      = "glmnet",
  "Logistic Regression" = "glm",
  "SVM"                 = "svmRadial",
  "XGBoost"             = "xgbTree",
  "Neural Network"      = "nnet"
)

results <- list()
for(model_name in names(models)) {
  set.seed(123)
  formula_model <- as.formula(paste(outcome, "~ ."))
  
  if(models[[model_name]] == "glm") {
    model <- train(
      formula_model,
      data = train_processed,
      method = "glm",
      family = binomial(),
      trControl = ctrl,
      metric = "ROC"
    )
  } else {
    model <- train(
      formula_model,
      data = train_processed,
      method = models[[model_name]],
      trControl = ctrl,
      metric = "ROC",
      tuneLength = 5
    )
  }
  
  results[[model_name]] <- model
}


# ===============================
# 8. Final Evaluation on Test Set
# ===============================
performance <- purrr::map_dfr(names(results), function(name) {
  model <- results[[name]]
  pred <- predict(model, test_processed)
  prob <- predict(model, test_processed, type = "prob")[, "Multiple"]
  
  cm <- confusionMatrix(pred, test_processed[[outcome]], positive = "Multiple")
  MCC_val <- mccr::mccr(test_processed[[outcome]] == "Multiple", pred == "Multiple")
  
  data.frame(
    Model     = name,
    Accuracy  = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall    = cm$byClass["Recall"],
    F1        = cm$byClass["F1"],
    MCC       = MCC_val,
    Kappa     = cm$overall["Kappa"],
    AUROC     = as.numeric(pROC::roc(test_processed[[outcome]] == "Multiple", prob)$auc)
  )
}, .id = NULL)

print(performance)

# Save performance table to new path
write.csv(
  performance,
  "D:/Research/BDHS Research/ML Pregnancy loss/update/ML/Analysis/Update Table/ML_model_performance_final.csv",
  row.names = FALSE
)

cat("Pipeline executed successfully! Boruta plot generated and results saved.\n")








#### Tuning ####




# ===============================
# 6. Cross-validation setup
# ===============================
ctrl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = "final",
  allowParallel = TRUE
)

# ===============================
# 7. Models & Hyperparameter Grids
# ===============================
library(caret)
library(kernlab)  # for sigest()

# Estimate sigma for SVM
svm_sigma <- sigest(as.matrix(train_processed[, -which(names(train_processed) == outcome)]))[1]

model_grids <- list(
  "Random Forest" = expand.grid(mtry = c(2, 4, 6, 8)),
  
  "Decision Tree" = expand.grid(cp = seq(0.001, 0.05, by = 0.005)),
  
  "KNN" = expand.grid(k = seq(3, 15, by = 2)),
  
  "Logistic Lasso" = expand.grid(
    alpha = 1, 
    lambda = seq(0.0001, 0.1, length.out = 20)
  ),
  
  "SVM" = expand.grid(
    sigma = svm_sigma,
    C = 2^(-2:5)
  ),
  
  "XGBoost" = expand.grid(
    nrounds = c(100, 200),
    max_depth = c(3, 5, 7),
    eta = c(0.01, 0.1, 0.3),
    gamma = c(0, 1),
    colsample_bytree = c(0.6, 0.8),
    min_child_weight = c(1, 5),
    subsample = c(0.7, 1)
  ),
  
  "Neural Network" = expand.grid(
    size = c(3,5,7,9),
    decay = c(0, 0.001, 0.01, 0.1)
  )
)

# Logistic Regression does not require tuning
results <- list()

for(model_name in names(model_grids)) {
  set.seed(123)
  formula_model <- as.formula(paste(outcome, "~ ."))
  
  if(model_name == "Logistic Regression") {
    results[[model_name]] <- train(
      formula_model,
      data = train_processed,
      method = "glm",
      family = binomial(),
      trControl = ctrl,
      metric = "ROC"
    )
  } else {
    results[[model_name]] <- train(
      formula_model,
      data = train_processed,
      method = switch(model_name,
                      "Random Forest" = "rf",
                      "Decision Tree" = "rpart",
                      "KNN" = "knn",
                      "Logistic Lasso" = "glmnet",
                      "SVM" = "svmRadial",
                      "XGBoost" = "xgbTree",
                      "Neural Network" = "nnet"),
      trControl = ctrl,
      metric = "ROC",
      tuneGrid = model_grids[[model_name]]
    )
  }
}

# ===============================
# 8. Final Evaluation on Test Set
# ===============================
performance <- purrr::map_dfr(names(results), function(name) {
  model <- results[[name]]
  pred <- predict(model, test_processed)
  prob <- predict(model, test_processed, type = "prob")[, "Multiple"]
  
  cm <- confusionMatrix(pred, test_processed[[outcome]], positive = "Multiple")
  
  data.frame(
    Model     = name,
    Accuracy  = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall    = cm$byClass["Recall"],
    F1        = cm$byClass["F1"],
    AUROC     = as.numeric(pROC::roc(test_processed[[outcome]] == "Multiple", prob)$auc)
  )
}, .id = NULL)

print(performance)

# ===============================
# 9. Save performance table
# ===============================
write.csv(
  performance,
  "D:/Research/BDHS Research/ML Pregnancy loss/update/ML/Analysis/Update Table/ML_model_performance_final_tuning.csv",
  row.names = FALSE
)

cat("Pipeline executed successfully! Hyperparameter tuning completed and results saved.\n")










#### ভিসুয়ালিযাতিওন ####
library(pROC)
library(PRROC)
library(ggplot2)
library(dplyr)
library(tidyr)

# Initialize lists to store curves
roc_list <- list()
pr_list <- list()

for(model_name in names(results)) {
  
  model <- results[[model_name]]
  
  # Predicted probabilities for "Multiple"
  prob <- predict(model, test_processed, type = "prob")[, "Multiple"]
  
  # True labels: 1 = Multiple, 0 = Single
  true <- ifelse(test_processed[[outcome]] == "Multiple", 1, 0)
  
  # ---------- ROC Curve ----------
  roc_obj <- roc(true, prob)
  roc_list[[model_name]] <- data.frame(
    TPR = rev(roc_obj$sensitivities),
    FPR = rev(1 - roc_obj$specificities),
    Model = model_name
  )
  
  # ---------- Precision-Recall Curve ----------
  pr_obj <- pr.curve(scores.class0 = prob[true==1],
                     scores.class1 = prob[true==0],
                     curve = TRUE)
  pr_list[[model_name]] <- data.frame(
    Recall = pr_obj$curve[,1],
    Precision = pr_obj$curve[,2],
    Model = model_name
  )
}

# Combine all ROC curves
library(PRROC)
library(ggplot2)
library(dplyr)

pr_list <- list()

for(model_name in names(results)) {
  
  model <- results[[model_name]]
  
  # Predicted probabilities for "Multiple"
  prob <- predict(model, test_processed, type = "prob")[, "Multiple"]
  
  # True labels: 1 = Multiple, 0 = Single
  true <- ifelse(test_processed[[outcome]] == "Multiple", 1, 0)
  
  # Only compute if both classes present in predictions
  if(length(unique(true)) == 2 && length(unique(prob)) > 1) {
    pr_obj <- pr.curve(
      scores.class0 = prob[true==1],
      scores.class1 = prob[true==0],
      curve = TRUE
    )
    
    pr_list[[model_name]] <- data.frame(
      Recall = pr_obj$curve[,1],
      Precision = pr_obj$curve[,2],
      Model = model_name
    )
  } else {
    warning(paste("Skipping PR curve for", model_name, "– insufficient variation in predictions"))
  }
}

# Combine valid PR curves
pr_df <- bind_rows(pr_list)

# Plot Precision-Recall curves
ggplot(pr_df, aes(x = Recall, y = Precision, color = Model)) +
  geom_line(size = 1) +
  labs(title = "Precision-Recall Curves for All Models",
       x = "Recall",
       y = "Precision") +
  theme_minimal(base_size = 12)

























# ===============================
# Train standard Logistic Regression
# ===============================
set.seed(123)
best_model <- train(
  as.formula(paste(outcome, "~ .")),
  data = train_processed,
  method = "glm",
  family = binomial(),
  trControl = ctrl,
  metric = "ROC"
)

# Predicted probabilities
pred_probs <- predict(best_model, test_processed, type = "prob")[, "Multiple"]









# ===============================
# 10. SHAP for Logistic Regression
# ===============================
library(iml)
library(randomForest)  # needed for Predictor class
library(ggplot2)

# Prepare predictor object
X_test <- test_processed[, setdiff(names(test_processed), outcome)]
predictor <- Predictor$new(
  model = best_model,
  data = X_test,
  y = test_processed[[outcome]],
  type = "prob"
)

# Compute SHAP values
shapley <- Shapley$new(predictor, x.interest = X_test[1, ]) # example for first observation

# Global feature importance
shap_imp <- FeatureImp$new(predictor, loss = "ce")  # cross-entropy
plot(shap_imp) + ggtitle("SHAP Feature Importance - Logistic Regression")

# Optional: SHAP for multiple observations (summary plot)
shapley_summary <- Shapley$new(predictor, sample = X_test[1:100, ]) # use first 100 rows
plot(shapley_summary)










# Use the dummy-coded processed test data
X_test <- test_processed[, setdiff(names(test_processed), outcome)]

# Ensure factor levels are consistent
y_test <- test_processed[[outcome]]

# Create Predictor object for iml
predictor <- Predictor$new(
  model = best_model,
  data = X_test,
  y = y_test,
  type = "prob"
)

# Global feature importance
shap_imp <- FeatureImp$new(predictor, loss = "ce")
plot(shap_imp) + ggtitle("SHAP Feature Importance - Logistic Regression")

# SHAP for multiple observations (summary)
# Use first 50-100 rows to reduce computation
shapley_summary <- Shapley$new(predictor, x.interest = X_test[1:100, ])
plot(shapley_summary)








# ===============================
# Libraries
# ===============================
library(iml)
library(lime)
library(ggplot2)
library(dplyr)

# ===============================
# Prepare predictor object for SHAP
# ===============================
X_test <- test_processed[, setdiff(names(test_processed), outcome)]
y_test <- test_processed[[outcome]]

predictor <- Predictor$new(
  model = best_model,
  data = X_test,
  y = y_test,
  type = "prob"
)

# ===============================
# 1. SHAP Global Feature Importance
# ===============================
shap_imp <- FeatureImp$new(predictor, loss = "ce")  # cross-entropy loss

# Clean Lancet-style color
lancet_blue <- "#4472C4"

ggplot(shap_imp$results, aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity", fill = lancet_blue, color = "white", width = 0.7) +
  coord_flip() +
  labs(
    title = "SHAP Feature Importance - Logistic Regression",
    x = "Feature",
    y = "Importance (cross-entropy)"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    axis.text = element_text(size = 11),
    axis.title = element_text(face = "bold", size = 12),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )

# ===============================
# 2. SHAP Summary Plot for Sample Observations
# ===============================
shapley_summary <- Shapley$new(predictor, x.interest = X_test[1:100, ])

# Optional: base ggplot style
plot(shapley_summary) +
  theme_minimal(base_family = "sans") +
  labs(title = "SHAP Summary - First 100 Test Observations") +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    axis.text = element_text(size = 11)
  )







# ===============================
# 2. SHAP Summary Plot for Sample Observations (Polished)
# ===============================

# Use 500 observations for robust summary
n_obs <- min(500, nrow(X_test))  # ensures we don't exceed dataset size

shapley_summary <- Shapley$new(predictor, x.interest = X_test[1:n_obs, ])

# Extract data for ggplot
shap_data <- shapley_summary$results

# Create clean, polished SHAP summary plot
ggplot(shap_data, aes(x = reorder(feature, phi), y = phi, color = phi)) +
  geom_jitter(width = 0.25, height = 0, alpha = 0.8, size = 2) +
  scale_color_gradient2(low = "#1ABC9C", mid = "white", high = "#FF69B4", midpoint = 0) +  # Firoza -> white -> pink
  coord_flip() +
  labs(
    title = paste0("SHAP Summary - ", n_obs, " Test Observations"),
    x = "Feature",
    y = "SHAP Value",
    color = "Impact"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(face = "bold")
  )











# ===============================
# Libraries
# ===============================
library(caret)
library(fastshap)
library(ggplot2)
library(dplyr)
library(tidyr)

# ===============================
# Define outcome variable
# ===============================
outcome <- "preg_loss_cat"

# Ensure outcome is factor with correct levels
train_processed[[outcome]] <- factor(train_processed[[outcome]], levels = c("Single","Multiple"))
test_processed[[outcome]]  <- factor(test_processed[[outcome]], levels = c("Single","Multiple"))

# ===============================
# Cross-validation setup
# ===============================
ctrl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = "final",
  allowParallel = TRUE
)

# ===============================
# Train standard Logistic Regression
# ===============================
# ===============================
# Train standard Logistic Regression - correct syntax
# ===============================
set.seed(123)
logistic_model <- caret::train(
  as.formula(paste(outcome, "~ .")),   # formula object
  data = train_processed,
  method = "glm",
  family = binomial(),
  trControl = ctrl,
  metric = "ROC"
)


library(fastshap)
library(ggplot2)
library(dplyr)
library(tidyr)








library(fastshap)
library(ggplot2)
library(dplyr)
library(tidyr)

# -------------------------------
# Predictor function
# -------------------------------
pred_fun <- function(object, newdata) {
  predict(object, newdata = newdata, type = "response")
}

# -------------------------------
# Subset test data (up to 500 obs)
# -------------------------------
n_obs <- min(500, nrow(test_processed))
X_subset <- test_processed[1:n_obs, setdiff(names(test_processed), outcome)]
X_subset <- as.data.frame(X_subset)  # ensure clean data.frame

# -------------------------------
# Compute approximate SHAP values
# -------------------------------
set.seed(123)
shap_vals <- fastshap::explain(
  object = logistic_model$finalModel,
  X = X_subset,
  pred_wrapper = pred_fun,
  nsim = 100
)

# -------------------------------
# Convert to long format
# -------------------------------
shap_long <- shap_vals %>%
  as.data.frame() %>%
  mutate(obs = 1:n()) %>%
  pivot_longer(
    cols = -obs,
    names_to = "feature",
    values_to = "phi"
  ) %>%
  left_join(
    X_subset %>%
      mutate(obs = 1:n()) %>%
      pivot_longer(
        cols = everything(),
        names_to = "feature",
        values_to = "feature_value"
      ),
    by = c("obs", "feature")
  ) %>%
  mutate(feature = factor(feature, levels = rev(unique(feature))))

# -------------------------------
# Beeswarm / Brewsplot style SHAP plot
# -------------------------------
ggplot(shap_long, aes(x = phi, y = feature, color = feature_value)) +
  geom_jitter(width = 0.2, height = 0.2, alpha = 0.9, size = 2) +
  scale_color_gradient(low = "#1ABC9C", high = "#FF69B4") +  # Firoza -> Pink
  labs(
    title = paste0("SHAP Beeswarm Plot - ", n_obs, " Observations"),
    x = "SHAP Value",
    y = "Feature",
    color = "Feature Value"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(face = "bold")
  )













library(fastshap)
library(ggplot2)
library(dplyr)
library(tidyr)

# -------------------------------
# Predictor function
# -------------------------------
pred_fun <- function(object, newdata) {
  predict(object, newdata = newdata, type = "response")
}

# -------------------------------
# Subset test data (up to 500 obs)
# -------------------------------
n_obs <- min(500, nrow(test_processed))
X_subset <- test_processed[1:n_obs, setdiff(names(test_processed), outcome)]
X_subset <- as.data.frame(X_subset)  # ensure data.frame

# -------------------------------
# Compute approximate SHAP values
# -------------------------------
set.seed(123)
shap_vals <- fastshap::explain(
  object = logistic_model$finalModel,
  X = X_subset,
  pred_wrapper = pred_fun,
  nsim = 100
)

# -------------------------------
# Convert to long format safely
# -------------------------------
# Add observation index manually
shap_df <- as.data.frame(shap_vals)
shap_df$obs <- seq_len(nrow(shap_df))

X_subset$obs <- seq_len(nrow(X_subset))

# Pivot long for SHAP values
shap_long <- shap_df %>%
  pivot_longer(
    cols = -obs,
    names_to = "feature",
    values_to = "phi"
  ) %>%
  left_join(
    X_subset %>%
      pivot_longer(
        cols = -obs,
        names_to = "feature",
        values_to = "feature_value"
      ),
    by = c("obs", "feature")
  ) %>%
  mutate(feature = factor(feature, levels = rev(unique(feature))))

# -------------------------------
# SHAP Beeswarm / Brewsplot
# -------------------------------
ggplot(shap_long, aes(x = phi, y = feature, color = feature_value)) +
  geom_jitter(width = 0.2, height = 0.2, alpha = 0.9, size = 2) +
  scale_color_gradient(low = "#1ABC9C", high = "#FF69B4") +
  labs(
    title = paste0("SHAP Beeswarm Plot - ", n_obs, " Observations"),
    x = "SHAP Value",
    y = "Feature",
    color = "Feature Value"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(face = "bold")
  )










# -------------------------------
# SHAP Beeswarm / Brewsplot - Blue to Orange Gradient
# -------------------------------
ggplot(shap_long, aes(x = phi, y = feature, color = feature_value)) +
  geom_jitter(width = 0.2, height = 0.2, alpha = 0.8, size = 2.5) +
  scale_color_gradient(low = "#1E90FF", high = "#FF8C00") +  # Blue -> Orange
  labs(
    title = paste0("SHAP Beeswarm Plot - ", n_obs, " Observations"),
    x = "SHAP Value",
    y = "Feature",
    color = "Feature Value"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text.y = element_text(size = 11, face = "bold"),    # Feature names bold
    axis.text.x = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 10)
  ) +
  scale_y_discrete(labels = function(x) gsub("_", " ", x))  # Replace underscores with spaces
















# -------------------------------
# SHAP Beeswarm / Brewsplot - Polished for Q1 Journal
# -------------------------------
ggplot(shap_long, aes(x = phi, y = feature, color = feature_value)) +
  geom_jitter(width = 0.15, height = 0.25, alpha = 0.7, size = 3, shape = 16) +
  scale_color_gradient(low = "#1E90FF", high = "#FFA500") +  # Blue -> Orange
  labs(
    title = paste0("SHAP Beeswarm Plot - ", n_obs, " Observations"),
    x = "SHAP Value",
    y = "Feature",
    color = "Feature Value"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text.y = element_text(size = 11, face = "bold"),
    axis.text.x = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 10),
    plot.background = element_rect(fill = "white", color = NA)
  ) +
  scale_y_discrete(labels = function(x) gsub("_", " ", x)) +
  guides(color = guide_colorbar(barwidth = 1, barheight = 15))  # Sleeker legend






ggplot(shap_long, aes(x = phi, y = feature, color = feature_value)) +
  geom_jitter(width = 0.2, height = 0.25, alpha = 0.8, size = 2.5, shape = 16) +
  scale_color_gradient2(low = "#1E90FF", mid = "#FFFFFF", high = "#FF4500", midpoint = median(shap_long$feature_value)) +
  labs(
    title = paste0("SHAP Beeswarm Plot - ", n_obs, " Observations"),
    x = "SHAP Value",
    y = "Feature",
    color = "Feature Value"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text.y = element_text(size = 11, face = "bold"),
    axis.text.x = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 10)
  ) +
  scale_y_discrete(labels = function(x) gsub("_", " ", x))






library(viridis)

ggplot(shap_long, aes(x = phi, y = feature, color = feature_value)) +
  geom_jitter(width = 0.2, height = 0.25, alpha = 0.85, size = 3, shape = 16) +
  scale_color_viridis(option = "C") +
  labs(
    title = paste0("SHAP Beeswarm Plot - ", n_obs, " Observations"),
    x = "SHAP Value",
    y = "Feature",
    color = "Feature Value"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text.y = element_text(size = 11, face = "bold"),
    axis.text.x = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 10)
  ) +
  scale_y_discrete(labels = function(x) gsub("_", " ", x))







ggplot(shap_long, aes(x = phi, y = feature, color = feature_value)) +
  geom_jitter(width = 0.18, height = 0.2, alpha = 0.7, size = 3, shape = 16, stroke = 0.1) +
  scale_color_gradient(low = "#1E90FF", high = "#FFA500") +
  labs(
    title = paste0("SHAP Beeswarm Plot - ", n_obs, " Observations"),
    x = "SHAP Value",
    y = "Feature",
    color = "Feature Value"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5, color = "#2F4F4F"),
    axis.text.y = element_text(size = 11, face = "bold"),
    axis.text.x = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 10),
    plot.background = element_rect(fill = "#FCFCFC", color = NA)
  ) +
  scale_y_discrete(labels = function(x) gsub("_", " ", x))









# ===============================
# 3. LIME Explanations
# ===============================
explainer <- lime::lime(
  x = train_processed[, setdiff(names(train_processed), outcome)],
  model = best_model
)

# Explain first 5 test observations
explanation <- lime::explain(
  X_test[1:5, ],
  explainer = explainer,
  n_features = 10,
  n_labels = 1
)

# LIME Feature Contributions Plot
plot_features(explanation) +
  scale_fill_manual(values = c("#FF69B4", "#1ABC9C")) +  # Pink & Firoza
  labs(
    title = "LIME Feature Contributions - Logistic Regression",
    x = "Feature",
    y = "Contribution"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    axis.text = element_text(size = 11),
    legend.position = "top"
  )







# ===============================
# 3. LIME Explanations - Polished
# ===============================
library(lime)
library(ggplot2)
library(dplyr)

# Create explainer using training data
explainer <- lime::lime(
  x = train_processed[, setdiff(names(train_processed), outcome)],
  model = logistic_model  # using the logistic_model from caret
)

# Explain first 7 test observations (you can increase if needed)
explanation <- lime::explain(
  X_test[1:7, setdiff(names(X_test), outcome)],
  explainer = explainer,
  n_features = 10,
  n_labels = 1
)

# LIME Feature Contributions Plot - polished
plot_features(explanation) +
  scale_fill_manual(values = c("#1E90FF", "#FF8C00")) +  # Blue -> Orange
  labs(
    title = "LIME Feature Contributions - Logistic Regression",
    x = "Feature",
    y = "Contribution"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text = element_text(size = 11, face = "bold"),
    axis.title = element_text(size = 12, face = "bold"),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 10),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  scale_y_discrete(labels = function(x) gsub("_", " ", x))  # clean feature names



















# ===============================
# 3. LIME Explanations - Clean Faceted
# ===============================
library(lime)
library(ggplot2)
library(dplyr)
library(tidyr)

# Create explainer using training data
explainer <- lime::lime(
  x = train_processed[, setdiff(names(train_processed), outcome)],
  model = logistic_model  # using logistic_model from caret
)

# Explain first 7 test observations (increase if needed)
explanation <- lime::explain(
  X_test[1:7, setdiff(names(X_test), outcome)],
  explainer = explainer,
  n_features = 10,
  n_labels = 1
)

# Convert to ggplot-friendly format
lime_plot <- plot_features(explanation)

# Polished faceted plot
lime_plot +
  scale_fill_manual(values = c("#1E90FF", "#FF8C00")) +  # Blue -> Orange
  labs(
    title = "LIME Feature Contributions - Logistic Regression",
    x = "Feature",
    y = "Contribution"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text = element_text(size = 11, face = "bold"),
    axis.title = element_text(size = 12, face = "bold"),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 10),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  facet_wrap(~ case, scales = "free_y") +  # one panel per observation
  scale_y_discrete(labels = function(x) gsub("_", " ", x))  # clean feature names















# ===============================
# LIME Explanations - Lancet/JAMA style
# ===============================
library(lime)
library(ggplot2)
library(dplyr)
library(tidyr)

# Create explainer using training data
explainer <- lime::lime(
  x = train_processed[, setdiff(names(train_processed), outcome)],
  model = logistic_model
)

# Explain first 7 test observations
explanation <- lime::explain(
  X_test[1:7, setdiff(names(X_test), outcome)],
  explainer = explainer,
  n_features = 10,
  n_labels = 1
)

# Base plot
lime_plot <- plot_features(explanation)

# Lancet/JAMA style
lime_plot +
  scale_fill_manual(values = c("#0072B2", "#D55E00")) +  # Blue & Orange
  labs(
    title = "LIME Feature Contributions - Logistic Regression",
    x = "Feature",
    y = "Contribution"
  ) +
  theme_classic(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    axis.text = element_text(size = 7, face = "bold"),
    axis.title = element_text(size = 7, face = "bold"),
    strip.text = element_text(face = "bold", size = 12),  # facet titles
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 7),
    legend.text = element_text(size = 7),
    panel.border = element_rect(color = "black", fill = NA, size = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  facet_wrap(~ case, scales = "free_y") +
  scale_y_discrete(labels = function(x) gsub("_", " ", x))  # clean feature names













library(ggplot2)
library(dplyr)
library(caret)
library(pROC)






library(ggplot2)
library(pROC)
library(dplyr)











library(ggplot2)
library(dplyr)

# Predicted probabilities for "Multiple"
pred_probs <- predict(logistic_model, test_processed, type = "prob")[, "Multiple"]

# Convert observed outcome to numeric 0/1
obs_numeric <- ifelse(test_processed[[outcome]] == "Multiple", 1, 0)

# Create calibration data with 10 bins
calib_df <- data.frame(obs = obs_numeric, pred = pred_probs) %>%
  mutate(bin = ntile(pred, 10)) %>%
  group_by(bin) %>%
  summarise(
    mean_pred = mean(pred),
    mean_obs  = mean(obs),
    .groups = "drop"
  )

# Plot calibration curve
ggplot(calib_df, aes(x = mean_pred, y = mean_obs)) +
  geom_point(color = "#1E90FF", size = 3) +
  geom_line(color = "#FF8C00", size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  labs(
    title = "Calibration Curve - Logistic Regression",
    x = "Mean Predicted Probability",
    y = "Observed Proportion"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold")
  )




summary(pred_probs)






library(ggplot2)
library(dplyr)

# Convert outcome to numeric
obs_numeric <- ifelse(test_processed[[outcome]] == "Multiple", 1, 0)

# Create calibration bins (10 bins)
calib_df <- data.frame(obs = obs_numeric, pred = pred_probs) %>%
  mutate(bin = ntile(pred, 10)) %>%
  group_by(bin) %>%
  summarise(
    mean_pred = mean(pred),
    mean_obs  = mean(obs),
    .groups = "drop"
  )

# Calibration plot
ggplot(calib_df, aes(x = mean_pred, y = mean_obs)) +
  geom_point(color = "#1E90FF", size = 3) +           # points
  geom_line(color = "#FF8C00", size = 1) +           # connecting line
  geom_abline(slope = 1, intercept = 0,              # perfect calibration
              linetype = "dashed", color = "grey50") +
  labs(
    title = "Calibration Curve - Logistic Regression",
    x = "Mean Predicted Probability",
    y = "Observed Proportion"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold")
  )





library(ggplot2)
library(dplyr)

# Convert outcome to numeric: 1 = "Multiple", 0 = "Single"
obs_numeric <- ifelse(test_processed[[outcome]] == "Multiple", 1, 0)

# Create 10 fixed probability bins
bin_cut <- cut(
  pred_probs,
  breaks = seq(0, 1, by = 0.1),
  include.lowest = TRUE,
  right = FALSE
)

# Compute mean predicted probability and observed proportion per bin
calib_df <- data.frame(obs = obs_numeric, pred = pred_probs, bin = bin_cut) %>%
  group_by(bin) %>%
  summarise(
    mean_pred = mean(pred),
    mean_obs  = mean(obs),
    n_obs     = n(),
    .groups = "drop"
  )

# Only keep bins with at least 1 observation
calib_df <- calib_df %>% filter(!is.na(mean_pred))

# Calibration plot
ggplot(calib_df, aes(x = mean_pred, y = mean_obs)) +
  geom_point(aes(size = n_obs), color = "#1E90FF", alpha = 0.8) +  # point size = sample in bin
  geom_line(color = "#FF8C00", size = 1) +                        # connecting line
  geom_abline(slope = 1, intercept = 0,                            # perfect calibration
              linetype = "dashed", color = "grey50") +
  scale_size_continuous(name = "Number of Obs per Bin") +
  labs(
    title = "Calibration Curve - Logistic Regression",
    x = "Mean Predicted Probability",
    y = "Observed Proportion"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    legend.title = element_text(face = "bold")
  )
















# -------------------------------
# Predicted probabilities for test set
# -------------------------------
pred_probs <- predict(logistic_model, X_subset, type = "prob")[, "Multiple"]

# Convert outcome to numeric: 1 = "Multiple", 0 = "Single"
obs_numeric <- ifelse(test_processed[[outcome]][1:n_obs] == "Multiple", 1, 0)

# -------------------------------
# Create 10 fixed probability bins
# -------------------------------
bin_cut <- cut(
  pred_probs,
  breaks = seq(0, 1, by = 0.1),
  include.lowest = TRUE,
  right = FALSE
)

# -------------------------------
# Compute mean predicted probability and observed proportion per bin
# -------------------------------
calib_df <- data.frame(obs = obs_numeric, pred = pred_probs, bin = bin_cut) %>%
  group_by(bin) %>%
  summarise(
    mean_pred = mean(pred, na.rm = TRUE),
    mean_obs  = mean(obs, na.rm = TRUE),
    n_obs     = length(obs),
    .groups = "drop"
  ) %>%
  filter(!is.na(mean_pred))  # remove empty bins

# -------------------------------
# Calibration plot
# -------------------------------
ggplot(calib_df, aes(x = mean_pred, y = mean_obs)) +
  geom_point(aes(size = n_obs), color = "#1E90FF", alpha = 0.8) +
  geom_line(color = "#FF8C00", size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  scale_size_continuous(name = "Number of Obs per Bin") +
  labs(
    title = "Calibration Curve - Logistic Regression",
    x = "Mean Predicted Probability",
    y = "Observed Proportion"
  ) +
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12, face = "bold"),
    legend.title = element_text(face = "bold")
  )




















# Example: Compare Logistic Regression vs Random Forest
# Get predicted classes
pred_log <- predict(logistic_model, test_processed)
pred_rf  <- predict(results[["Random Forest"]], test_processed)

# Build 2x2 contingency table
mcnemar_table <- table(
  LR = pred_log,
  RF = pred_rf
)

print(mcnemar_table)

# McNemar test
mcnemar_test <- mcnemar.test(mcnemar_table)
print(mcnemar_test)














