# Split data into training and testing
## Here we partition the data such that there is 80% within the training dataset and 20% within the testing dataset. This step also involves preparing dummy variables.

## Subset variables
combined_df <- subset(combined_df,select=c(id,
                                            Status,
                                            age,
                                            race,
                                            sex_at_birth,
                                            fam_history,
                                            sys_bp,
                                            height,
                                            weight,
                                            waist_circumference,
                                            hip_circumference,
                                            ldl,
                                            hdl,
                                            triglycerides,
                                            smoke_final,
                                            alcohol_final,
                                            sodium,
                                            chloride,
                                            creatinine,
                                            hemoglobin,
                                            urea_nitrogen,
                                            mental_health,
                                            highest_grade,
                                            income))
combined_df <- combined_df[combined_df$race==1 | combined_df$race==2 | combined_df$race==3,] # Only save White, Black, or Hispanic
combined_df <- combined_df[complete.cases(combined_df),] # Save only complete cases

library(caret) # Used to split data
library(fastDummies) # Create dummy variables

# Create dummy columns for drinking and smoking, preserve all levels
final_combined <- dummy_cols(combined_df,
                             select_columns=c("smoke_final","alcohol_final"),
                             ignore_na=TRUE,
                             remove_first_dummy = FALSE,
                             remove_selected_columns = TRUE)

# Create dummy columns for sex at birth and family history, restrict to one level
final_combined <- dummy_cols(final_combined,
                             select_columns=c("sex_at_birth","fam_history"),
                             ignore_na=TRUE,
                             remove_first_dummy = TRUE,
                             remove_selected_columns = TRUE)

train_indexes <- createDataPartition(final_combined$Status,p=0.8,list=FALSE) # Split data, stratify by status to ensure consistency
train_df <- final_combined[train_indexes,] # Define training df
test_df <- final_combined[-train_indexes,] # Define testing df

# Build random forest model
library(randomForest)

test_df$Status <- as.factor(test_df$Status) # Convert Status to a factor
train_df$Status <- as.factor(train_df$Status) # Convert Status to a factor

t2d_model <- randomForest(Status ~ .,data=train_df %>% select(-c(id,race))) # Build model, remove id and race
t2d_model$confusion[,'class.error']
print(t2d_model)

# Evaluate model
library(pROC)
library(caret)

# Predict on the test set
predictions <- predict(t2d_model, test_df %>% select(-c(id,race,Status)))

# Confusion matrix
confMatrix <- confusionMatrix(predictions, as.factor(test_df$Status),positive="1")
print(confMatrix)

# Accuracy
accuracy <- confMatrix$overall['Accuracy']
print(paste("Accuracy:", round(accuracy, 4)))

# Precision, Recall, F1-Score
precision <- confMatrix$byClass['Precision']
recall <- confMatrix$byClass['Recall']
f1 <- 2 * ((precision * recall) / (precision + recall))
print(paste("Precision:", round(precision, 4)))
print(paste("Recall:", round(recall, 4)))
print(paste("F1 Score:", round(f1, 4)))

# AUC
# Note: For multi-class problems, AUC is generally computed for each class versus the rest
probabilities <- predict(t2d_model, test_df %>% select(-c(id,Status,race)), type = "prob")
rocCurve <- multiclass.roc(test_df$Status, as.matrix(probabilities))
auc <- auc(rocCurve)
print(paste("AUC:", round(auc, 4)))

roc_curve <- roc(test_df$Status, probabilities[, 2], levels = rev(levels(test_df$Status)))

roc_df <- data.frame(
  tpr = roc_curve$sensitivities,
  fpr = 1 - roc_curve$specificities,
  thresholds = roc_curve$thresholds
)

# Plot the ROC curve with ggplot2
plt <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "blue", size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "ROC Curve for Random Forest",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_classic() +
  annotate("text", x = 0.75, y = 0.25, 
           label = paste("AUC =", round(auc(roc_curve), 4)), 
           color = "blue", size = 5, hjust = 0) +
    scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0))

plt

# Calculate SHAP values
## We utilize the treeshap package to calculate SHAP values
library(treeshap)
library(shapviz)

unified <- unify(t2d_model,train_df %>% select(-c(id,Status,race))) # Build a unified treeshap object of model and training data

treeshap1 <- treeshap(unified,  test_df %>% select(-c(id,race,Status)), verbose = 0) # Calculate SHAP values for testing data

base_prediction <- mean(as.data.frame(predict(t2d_model,train_df %>% select(-c(Status,id,race)),type="prob"))[,2]) # Calculate base prediction
base_prediction

final_shaps <- treeshap1$shaps # Extract df of SHAP values

final_shaps$base_prediction <- base_prediction # Add base prediction to df
final_shaps$predictions <- as.vector(predict(t2d_model,test_df %>% select(-c(Status,id,race)),type="prob")[,2]) # Add final prediction to df

# Rename columns in testing df and SHAP df for visualization
colnames(test_df) <- c("id",
                          "Status",
                          "Age",
                          "race",
                          "Systolic BP",
                          "Height (cm)",
                          "Weight (kg)",
                          "Waist Circumference (cm)",
                          "Hip Circumference (cm)",
                          "LDL",
                          "HDL",
                          "Triglycerides",
                          "Chloride",
                          "Creatinine",
                          "Hemoglobin",
                          "Sodium",
                          "Urea Nitrogen",
                          "Mental Health",
                          "Income",
                          "Highest Grade",
                          "Smoking Habits (Never smoked)",
                          "Smoking Habits (Former smoker)",
                          "Smoking Habits (Sometimes smokes)",
                          "Smoking Habits (Daily smoker)",
                          "Drinking Habits (Never drank)",
                          "Drinking Habits (Not in the past year)",
                          "Drinking Habits (Monthly or less)",
                          "Drinking Habits (Two to four times a month)",
                          "Drinking Habits (Two to three a week)",
                          "Drinking Habits (Four or more a week)",
                          "Sex at Birth",
                          "Family History")

colnames(full_shap) <- c("Age",
                          "Systolic BP",
                          "Height (cm)",
                          "Weight (kg)",
                          "Waist Circumference (cm)",
                          "Hip Circumference (cm)",
                          "LDL",
                          "HDL",
                          "Triglycerides",
                          "Chloride",
                          "Creatinine",
                          "Hemoglobin",
                          "Sodium",
                          "Urea Nitrogen",
                          "Mental Health",
                          "Income",
                          "Highest Grade",
                          "Smoking Habits (Never smoked)",
                          "Smoking Habits (Former smoker)",
                          "Smoking Habits (Sometimes smokes)",
                          "Smoking Habits (Daily smoker)",
                          "Drinking Habits (Never drank)",
                          "Drinking Habits (Not in the past year)",
                          "Drinking Habits (Monthly or less)",
                          "Drinking Habits (Two to four times a month)",
                          "Drinking Habits (Two to three a week)",
                          "Drinking Habits (Four or more a week)",
                          "Sex at Birth",
                          "Family History",
                         "base_prediction",
                         "predictions")

full_shap$race <- test_df$race # Add race to SHAP df
full_shap$Status <- test_df$Status # Add T2D Status to SHAP df
base_prediction <- unique(full_shap$base_prediction) # Save base prediction from data
full_shap <- full_shap %>% select(-c(base_prediction,predictions)) # Remove base and final predictions

# Visualize SHAP values for White participants
library(shapviz)

shapley_race1 <- full_shap[full_shap$race == 1,] # SHAP df for white
shapley_race1 <- shapley_race1 %>%
                    select(-c(race,Status)) # Remove race and status

long_df <- as.data.frame(colMeans(shapley_race1)) # Get mean SHAP for each variable
long_df <- rownames_to_column(long_df, "Variable") # Convert row names to a column
colnames(long_df)[2] <- "SHAP" # Rename column 2
# head(long_df)
# nrow(long_df)

wide_df <- long_df %>%
                pivot_wider(names_from = Variable, values_from=SHAP) # Convert data to wide form
wide_df <- wide_df[,order(colnames(wide_df))] # Order alphabetically

(wide_df)

white_data <- test_df[test_df$race == 1,] # Save white participants from testing data
white_data <- white_data %>% select(-c(id,Status,race)) # Remove ID, Status, race
long_white <- as.data.frame(colMeans(white_data,na.rm=TRUE)) # Calculate means of each variable
long_white <- rownames_to_column(long_white, "Variable") # Rownames to a separate column
colnames(long_white)[2] <- "Mean" # Rename column
wide_white <- long_white %>%
                pivot_wider(names_from = Variable, values_from=Mean) # Convert to wide form data
wide_white <- wide_white[,order(colnames(wide_white))] # Order by column name

(wide_white)

sv <- shapviz(as.matrix(wide_df),X=wide_white,baseline = base_prediction) # shapviz object for plotting

# Generate waterfall plot
sv_waterfall(sv,row_id=1,max_display=Inf,order_fun=function(s) 1:length(s),fill_colors=c("#FF6B82","#6DB3FF")) + 
             theme_classic() +
             ggtitle("White SHAP Values")

shapley_race1 <- shapley_race1[,order(colnames(shapley_race1))] # Order SHAP df

white_vals <- test_df[test_df$race == 1,] # Save testing data for White participants
white_vals <- white_vals %>% select(-c(id,race,Status)) # Remove id, race, and status
white_vals <- white_vals[,order(colnames(white_vals))] # Order columns

sv <- shapviz(as.matrix(shapley_race1),X=white_vals,baseline = base_prediction) # shapviz object for plotting

sv_summary <- sv_importance(sv,kind="beeswarm",max_display=Inf) + 
        theme_classic() + 
        ggtitle("White SHAP Summary") +
        theme(text = element_text(family = "DejaVu Sans"),
              plot.title = element_text(hjust = 0.5),
              axis.line.x = element_line(color = "black", linewidth = 0.25),
              axis.line.y = element_line(color = "white", linewidth = 0.25))

sv_summary

# Visualize SHAP values for Black participants
library(shapviz)
             
shapley_race2 <- full_shap[full_shap$race == 2,] # SHAP df for black
shapley_race2 <- shapley_race2 %>%
                    select(-c(race,Status)) # Remove race and status
             
long_df <- as.data.frame(colMeans(shapley_race2)) # Get mean SHAP for each variable
long_df <- rownames_to_column(long_df, "Variable") # Convert row names to a column
colnames(long_df)[2] <- "SHAP" # Rename column 2
# head(long_df)
# nrow(long_df)

wide_df <- long_df %>%
                pivot_wider(names_from = Variable, values_from=SHAP) # Convert data to wide form
wide_df <- wide_df[,order(colnames(wide_df))] # Order alphabetically

(wide_df)

black_data <- test_df[test_df$race == 2,] # Save black participants from testing data
black_data <- black_data %>% select(-c(id,Status,race)) # Remove ID, Status, race
long_black <- as.data.frame(colMeans(black_data,na.rm=TRUE)) # Calculate means of each variable
long_black <- rownames_to_column(long_black, "Variable") # Rownames to a separate column
colnames(long_black)[2] <- "Mean" # Rename column
wide_black <- long_black %>%
                pivot_wider(names_from = Variable, values_from=Mean) # Convert to wide form data
wide_black <- wide_black[,order(colnames(wide_black))] # Order by column name

(wide_black)

sv <- shapviz(as.matrix(wide_df),X=wide_white,baseline = base_prediction) # shapviz object for plotting
             
# Generate waterfall plot
sv_waterfall(sv,row_id=1,max_display=Inf,order_fun=function(s) 1:length(s),fill_colors=c("#FF6B82","#6DB3FF")) + 
             theme_classic() +
             ggtitle("Black SHAP Values")

shapley_race2 <- shapley_race2[,order(colnames(shapley_race2))] # Order SHAP df
             
black_vals <- test_df[test_df$race == 2,] # Save testing data for Black participants
black_vals <- black_vals %>% select(-c(id,race,Status)) # Remove id, race, and status
black_vals <- black_vals[,order(colnames(black_vals))] # Order columns
             
sv <- shapviz(as.matrix(shapley_race2),X=black_vals,baseline = base_prediction) # shapviz object for plotting

sv_summary <- sv_importance(sv,kind="beeswarm",max_display=Inf) + 
        theme_classic() + 
        ggtitle("Black SHAP Summary") +
        theme(text = element_text(family = "DejaVu Sans"),
              plot.title = element_text(hjust = 0.5),
              axis.line.x = element_line(color = "black", linewidth = 0.25),
              axis.line.y = element_line(color = "white", linewidth = 0.25))

sv_summary

# Visualize SHAP values for Hispanic participants
library(shapviz)
             
shapley_race3 <- full_shap[full_shap$race == 3,] # SHAP df for Hispanic
shapley_race3 <- shapley_race3 %>%
                    select(-c(race,Status)) # Remove race and status
             
long_df <- as.data.frame(colMeans(shapley_race3)) # Get mean SHAP for each variable
long_df <- rownames_to_column(long_df, "Variable") # Convert row names to a column
colnames(long_df)[2] <- "SHAP" # Rename column 2
# head(long_df)
# nrow(long_df)

wide_df <- long_df %>%
                pivot_wider(names_from = Variable, values_from=SHAP) # Convert data to wide form
wide_df <- wide_df[,order(colnames(wide_df))] # Order alphabetically

             
(wide_df)

hispanic_data <- test_df[test_df$race == 3,] # Save hispanic participants from testing data
hispanic_data <- hispanic_data %>% select(-c(id,Status,race)) # Remove ID, Status, race
long_hispanic <- as.data.frame(colMeans(hispanic_data,na.rm=TRUE)) # Calculate means of each variable
long_hispanic <- rownames_to_column(long_hispanic, "Variable") # Rownames to a separate column
colnames(long_hispanic)[2] <- "Mean" # Rename column
wide_hispanic <- long_hispanic %>%
                pivot_wider(names_from = Variable, values_from=Mean) # Convert to wide form data
wide_hispanic <- wide_hispanic[,order(colnames(wide_hispanic))] # Order by column name

(wide_hispanic)

sv <- shapviz(as.matrix(wide_df),X=wide_hispanic,baseline = base_prediction) # shapviz object for plotting
             
# Generate waterfall plot
sv_waterfall(sv,row_id=1,max_display=Inf,order_fun=function(s) 1:length(s),fill_colors=c("#FF6B82","#6DB3FF")) + 
             theme_classic() +
             ggtitle("Hispanic SHAP Values")

shapley_race3 <- shapley_race3[,order(colnames(shapley_race3))] # Order SHAP df
             
hispanic_vals <- test_df[test_df$race == 3,] # Save testing data for hispanic participants
hispanic_vals <- hispanic_vals %>% select(-c(id,race,Status)) # Remove id, race, and status
hispanic_vals <- hispanic_vals[,order(colnames(hispanic_vals))] # Order columns
             
sv <- shapviz(as.matrix(shapley_race3),X=hispanic_vals,baseline = base_prediction) # shapviz object for plotting
             
sv_summary <- sv_importance(sv,kind="beeswarm",max_display=Inf) + 
        theme_classic() + 
        ggtitle("Hispanic SHAP Summary") +
        theme(text = element_text(family = "DejaVu Sans"),
              plot.title = element_text(hjust = 0.5),
              axis.line.x = element_line(color = "black", linewidth = 0.25),
              axis.line.y = element_line(color = "white", linewidth = 0.25))

sv_summary
