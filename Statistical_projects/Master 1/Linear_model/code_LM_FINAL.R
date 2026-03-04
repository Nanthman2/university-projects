
file_path <- "Sleep_health_and_lifestyle_dataset.csv"
dataset <- read.csv(file_path, header = TRUE, sep = ",")


# Needed libraries #############################################################

library(fastDummies)
library(ggplot2)
library(gridExtra)
library(tidyr)
library(dplyr)
library(glmnet)
library(car)
library(e1071)
library(olsrr)
library(skedastic)
library(sandwich)
library(lmtest)
library(performance)
library(tseries)

# Preprocessing ################################################################

dataset$BMI.Category <- gsub("Normal Weight", "Normal", dataset$BMI.Category)
dataset$Occupation <- gsub("Sales Representative","Salesperson",dataset$Occupation)


# 1 -------------------------------------------------------------------------
dataset$Systolic <- as.numeric(sub("/.*", "", dataset$Blood.Pressure)) 
dataset$Diastolic <- as.numeric(sub(".*/", "", dataset$Blood.Pressure)) 

dataset$Blood.Pressure <- NULL


# 2 -------------------------------------------------------------------------

categorical_columns <- c("Gender", "Occupation", "BMI.Category", "Sleep.Disorder") 
numerical_columns <- setdiff(names(dataset), categorical_columns) 
dataset <- dataset[, c(numerical_columns, categorical_columns)] 

# 3 -------------------------------------------------------------------------

dataset$Occupation_Type <- case_when(
  dataset$Occupation %in% c("Doctor", "Nurse") ~ "Health",
  dataset$Occupation %in% c("Software Engineer", "Engineer", "Scientist") ~ "Technical",
  dataset$Occupation %in% c("Salesperson", "Accountant", "Manager") ~ "Business",
  TRUE ~ "Other"  
)

dataset$Sleep.Disorder <- case_when(
  dataset$Sleep.Disorder %in% c("Insomnia","Sleep Apnea") ~ "Yes",
  TRUE ~ "No"
)

# 4 -------------------------------------------------------------------------

dataset <- dummy_cols(dataset, select_columns = c("Gender", "Occupation_Type", "BMI.Category", "Sleep.Disorder"))

names(dataset) <- gsub(" ", "_", names(dataset))


# Training et Validation sets #################################################

set.seed(2003)  # Pour la reproductibilité

validation_indices <- sample(nrow(dataset), size = 0.10 * nrow(dataset))
validation_set <- dataset[validation_indices, ]
training_set <- dataset[-validation_indices, ]


# Analyse Descriptive ##########################################################
var_quant <- c("Age", 
                      "Sleep.Duration", 
                      "Quality.of.Sleep", 
                      "Physical.Activity.Level", 
                      "Stress.Level", 
                      "Systolic", 
                      "Diastolic", 
                      "Heart.Rate", 
                      "Daily.Steps")

filtered_dataset <- dataset[, var_quant]

# Statistiques descriptives ----------------------------------------------------
descriptive_stats <- data.frame(
  Moyenne = sapply(filtered_dataset, mean, na.rm = TRUE),
  Ecart_Type = sapply(filtered_dataset, sd, na.rm = TRUE),
  Skewness = sapply(filtered_dataset, skewness, na.rm = TRUE),  # Appliquer directement skewness
  Kurtosis = sapply(filtered_dataset, kurtosis, na.rm = TRUE)     # Appliquer directement kurtosis
)

descriptive_stats <- round(descriptive_stats, 2)


descriptive_stats


# Résumé pour chaque variable catégorielle 

# Pour la variable 'Gender'
dataset %>%
  group_by(Gender) %>%
  summarise(moyenne = mean(Sleep.Duration), Ecart_type = sd(Sleep.Duration))

# Pour la variable 'occupation'
dataset %>%
  group_by(Occupation_Type) %>%
  summarise(moyenne = mean(Sleep.Duration), Ecart_type = sd(Sleep.Duration))

# Pour la variable 'BMI.Category'
dataset %>%
  group_by(BMI.Category) %>%
  summarise(moyenne = mean(Sleep.Duration),Ecart_type = sd(Sleep.Duration))

# Pour la variable 'Sleep.Disorder'
dataset %>%
  group_by(Sleep.Disorder) %>%
  summarise(moyenne = mean(Sleep.Duration),Ecart_type = sd(Sleep.Duration))


# Graphiques -------------------------------------------------------------------

plot_boxplot <- function(variable_name) {
  boxplot(
    dataset[[variable_name]],
    main = paste(variable_name),
    outcol = "red", 
    cex.lab = 1.2, 
    cex.axis = 0.8, 
    cex.main = 1.4 
  )
}

par(mfrow = c(3, 3), mar = c(5, 4, 3, 1)) # Ajuster la mise en page
for (var in var_quant) {
  plot_boxplot(var)
}

par(mfrow =c(1,1))
suppressPackageStartupMessages(require(corrplot))
corrplot(cor(dataset[2:10]), tl.col = "black",tl.cex=0.7, order = "hclust", addCoef.col = "darkgrey",method='number',diag=FALSE)


# Modelisation #################################################################

# Analyse de multicolinéarité --------------------------------------------------


reg_model <- lm(Sleep.Duration ~ Age + Quality.of.Sleep + Physical.Activity.Level + Stress.Level + Heart.Rate + Daily.Steps + Diastolic + Systolic + Gender_Male + Occupation_Type_Business +  Occupation_Type_Health +  Occupation_Type_Technical + BMI.Category_Normal + BMI.Category_Obese + Sleep.Disorder_Yes,data = training_set)
vif <-vif(reg_model)
vif
max(vif)


reg_model <- lm(Sleep.Duration ~ Age + Quality.of.Sleep + Physical.Activity.Level + Stress.Level + Heart.Rate + Daily.Steps + Systolic + Gender_Male + Occupation_Type_Business +  Occupation_Type_Health +  Occupation_Type_Technical + BMI.Category_Normal + BMI.Category_Obese + Sleep.Disorder_Yes,data = training_set)
vif <- vif(reg_model)
vif
max(vif)

reg_model <- lm(Sleep.Duration ~ Age + Physical.Activity.Level + Stress.Level + Heart.Rate + Daily.Steps + Systolic + Gender_Male + Occupation_Type_Business +  Occupation_Type_Health +  Occupation_Type_Technical + BMI.Category_Normal + BMI.Category_Obese + Sleep.Disorder_Yes,data = training_set)
vif <- vif(reg_model)
vif
max(vif)


# TIPE I MODEL SELECTION -------------------------------------------------------


#start <- Sys.time()
#each_model <- ols_step_all_possible(reg_model)
#end <- Sys.time()

#cat("Temps pour ols_step_all_possible :", end- start)

#plot(each_model) 

#best_subset <- ols_step_best_subset(reg_model)

#best_subset

#plot(best_subset)


# Modèle choisi par l'approche 'brute force'------------------------------------

brute_force <- lm(Sleep.Duration ~Age + Physical.Activity.Level + Stress.Level + Heart.Rate + Daily.Steps + Systolic + Gender_Male + Occupation_Type_Business
                  + Occupation_Type_Health + Occupation_Type_Technical + BMI.Category_Normal,data = training_set )
brute_force

# Calcul de la MSE
variables <- names(brute_force$coefficients)[-1] # [-1] pour ne pas prendre l'intercept
pred_bf <- predict(brute_force,validation_set[,variables])
mse_bf <- mean((validation_set$Sleep.Duration-pred_bf)^2)
cat("MSE pour forward Selection:", mse_bf, "\n")

# TYPE II MODEL SELECTION ------------------------------------------------------

forward <- ols_step_forward_adj_r2(reg_model)
forward

# Calcul de la MSE
variables <- names(forward$model$coefficients)[-1]
pred_forward <- predict(forward$model,validation_set[,variables])
mse_forward <- mean((validation_set$Sleep.Duration-pred_forward)^2)
cat("MSE pour forward Selection:", mse_forward, "\n")

backward <- ols_step_backward_p(reg_model)
backward 

variables <- names(backward$model$coefficients)[-1]
pred_backward <- predict(backward$model,validation_set[,variables])
mse_backward <- mean((validation_set$Sleep.Duration-pred_backward)^2)
cat("MSE pour Backward Selection:", mse_backward, "\n")


# TYPE III MODEL SELECTION -----------------------------------------------------

# On enlève l'identifiant 'Person.ID' (1), la variable dépendante 'Sleep.Duration' (3), 
# les deux variables qui causaient de la multicolinéarité 'Quality.of.Sleep' (4) et Diastolic (10),
# les variables catégorielles au forma texte : 'Gender' (11), Occupation (12), BMI.Category (13), Sleep.Disorder (14)
# la variable qui agrège les métiers par type : Occupation_Type (15), Une Dummy par catégorie Gender_Female (16), Occupation_Type_Other (20)
# BMI.Category_Overweight (24) et Sleep.Disorder_No (25)

colnames(dataset[,c(1,3,4,10,11,12,13,14,15,16,20,24,25)])

x_train <- as.matrix(training_set[,c(-1,-3,-4,-10,-11,-12,-13,-14,-15,-16,-20,-24,-25)])
y_train <- training_set$Sleep.Duration
lasso_train <- glmnet(x_train,y_train)


x_val <- as.matrix(validation_set[,c(-1,-3,-4,-10,-11,-12,-13,-14,-15,-16,-20,-24,-25)])
y_val <- validation_set$Sleep.Duration

lasso_cv <- cv.glmnet(x = x_train, y = y_train)   # par défaut 10-fold cross validation
lasso_cv
plot(lasso_cv)

best_lambda <- lasso_cv$lambda.min 
best_lasso <- predict(lasso_cv, type = "coefficients", s = best_lambda)
best_lasso

lambda_1se <- lasso_cv$lambda.1se
pred_simple <- predict(lasso_cv,  type = "coefficients", s = lambda_1se)
pred_simple

# MSE pour lasso_cv
pred_cv <- predict(lasso_cv, s = best_lambda, newx = x_val)
mse_cv <- mean((y_val - pred_cv)^2)

# MSE pour un modèle plus simple
pred_simple <- predict(lasso_cv, s = lambda_1se , newx = x_val)
mse_simple <- mean((y_val - pred_simple)^2)

cat("MSE pour lasso_cv:", mse_cv, "\nMSE pour simple lasso_cv:", mse_simple, "\n")


# Modèle de LASS0 --------------------------------------------------------------

best_lasso_matrix <- as.matrix(best_lasso)
selected_vars <- rownames(best_lasso_matrix)[best_lasso_matrix != 0]
selected_vars <- selected_vars[selected_vars != "(Intercept)"] 

formula_lm <- as.formula(paste("Sleep.Duration ~", paste(selected_vars, collapse = " + ")))
lm_model <- lm(formula_lm, data = training_set)

# Comparaison des MSE ----------------------------------------------------------
data.frame("MSE Best Subset" = mse_bf, "MSE Forward Slection" = mse_forward
           , "MSE Backward Elimination" = mse_backward
           , "MSE LASSO Lambda min" = mse_cv,"MSE LASSO Lambda1SE" = mse_simple)



# Vérification des hypothèses classiques #######################################

# Overview graphique -----------------------------------------------------------
par(mfrow=c(2,2))
plot(brute_force)


# Hypothèse de Linéarité -------------------------------------------------------

y_vars <- c("Age", "Physical.Activity.Level", "Stress.Level", "Heart.Rate", "Daily.Steps","Systolic")
plots <- list()

for (i in 1:length(y_vars)) {
  p <- ggplot(dataset, aes(x = Sleep.Duration, y = .data[[y_vars[i]]])) +
    geom_point(color = "black", alpha = 0.5) +  
    stat_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "red") +
    labs(x = "Sleep Duration", y = y_vars[i]) +
    theme_minimal()
  plots[[i]] <- p
}


grid.arrange(plots[[1]], plots[[2]], plots[[3]], plots[[4]], plots[[5]],plots[[6]], ncol = 3)


# Hypothèse de Multicolinéarité-------------------------------------------------

max(vif(brute_force))

# Hypothèse d'hétéroscédasticité -----------------------------------------------

fitted_values <- lm_model$fitted.values
residuals <- lm_model$residuals

plot(fitted_values, residuals, 
     xlab = "Valeurs ajustées", 
     ylab = "Résidus")
abline(h = 0, col = "red",lty=2 )


var <-  names(brute_force$coefficients)[-1]

par(mfrow=c(3,4))
for (variable in var) {
  plot(training_set[[variable]], brute_force$residuals,
       xlab = variable, 
       ylab = "Résidus")
  abline(h = 0, col = "red", lty = 2)  
}


white(brute_force,interactions = TRUE) # test formel

# Résoudre le problème d'hétéroscédasticité ? ----------------------------------

# Tranformation de variables ----------------------------------------------------

# Age en 1/Age
Age_non_transformé <- training_set$Age
training_set$Age <- (1/training_set$Age)

model_hom <- lm(Sleep.Duration ~Age + Physical.Activity.Level + Stress.Level + Heart.Rate + Daily.Steps + Systolic + Gender_Male + Occupation_Type_Business
                + Occupation_Type_Health + Occupation_Type_Technical + BMI.Category_Normal,data = training_set )
model_hom
AIC(model_hom)
par(mfrow=c(1,1))
plot(training_set$Age,model_hom$residuals)
abline(h = 0, col = "red", lty = 2)  

plot(training_set$Age,training_set$Sleep.Duration)

white(model_hom,interactions=TRUE)
training_set$Age <- Age_non_transformé


# Sleep.Duration en log(Sleep.Duration) et Heart.Rate en log(Heart.Rate)

Sleep_non_transofrmé <- training_set$Sleep.Duration
Heart_rate_non_transformé <- training_set$Heart.Rate

training_set$Sleep.Duration <- log(training_set$Sleep.Duration)
training_set$Heart.Rate <- log(training_set$Heart.Rate)

model_hom <- lm(Sleep.Duration ~Age + Physical.Activity.Level + Stress.Level + Heart.Rate + Daily.Steps + Systolic + Gender_Male + Occupation_Type_Business
                + Occupation_Type_Health + Occupation_Type_Technical + BMI.Category_Normal,data = training_set )
model_hom
AIC(model_hom)
par(mfrow=c(1,1))
plot(training_set$Heart.Rate,model_hom$residuals)
abline(h = 0, col = "red", lty = 2)  

plot(training_set$Heart.Rate,training_set$Sleep.Duration)

white(model_hom,interactions=TRUE)

training_set$Sleep.Duration <- Sleep_non_transofrmé
training_set$Heart.Rate <- Heart_rate_non_transformé

# Daily Steps en log Daily Steps
Daily_non_transformé <- training_set$Daily.Steps
training_set$Daily.Steps <- log(training_set$Daily.Steps)

model_hom <- lm(Sleep.Duration ~Age + Physical.Activity.Level + Stress.Level + Heart.Rate + Daily.Steps + Systolic + Gender_Male + Occupation_Type_Business
                + Occupation_Type_Health + Occupation_Type_Technical + BMI.Category_Normal,data = training_set )
model_hom
AIC(model_hom)
par(mfrow=c(1,1))
plot(training_set$Daily.Steps,model_hom$residuals)
abline(h = 0, col = "red", lty = 2)  
white(model_hom,interactions=TRUE)

# Combinaison des 2
training_set$Daily.Steps <- Daily_non_transforméHeart_rate_non_transformé <- training_set$Heart.Rate
training_set$Age <- (1/training_set$Age)
training_set$Daily.Steps <- log(training_set$Daily.Steps)

model_hom <- lm(Sleep.Duration ~Age + Physical.Activity.Level + Stress.Level + Heart.Rate + Daily.Steps + Systolic + Gender_Male + Occupation_Type_Business
                + Occupation_Type_Health + Occupation_Type_Technical + BMI.Category_Normal,data = training_set )
model_hom
white(model_hom,interactions=TRUE)

training_set$Daily.Steps <- Daily_non_transformé
training_set$Age <- Age_non_transformé

# FWLS -----------------------------------------------------------------------------

# # Calcul des résidus au carré
residuals2 <- brute_force$residuals^2

# # Calcul des carrés des variables explicatives
Age2 <- training_set$Age^2 
Phys2 <- training_set$Physical.Activity.Level^2 
Stress2 <- training_set$Stress.Level^2 
Rate2 <- training_set$Heart.Rate^2 
Daily2 <- training_set$Daily.Steps^2

 # Régression auxiliaire pour modéliser l'hétéroscédasticité
aux_model <- lm(residuals2 ~ (Age + Physical.Activity.Level + Stress.Level + Heart.Rate + 
         Daily.Steps + Gender_Male + Occupation_Type_Business + 
         Occupation_Type_Health + Occupation_Type_Technical + 
         BMI.Category_Normal + Systolic)^2 + Age2 + Phys2 + Stress2 + Rate2 + Daily2, data = training_set)

# # Afficher le résumé du modèle auxiliaire
summary(aux_model)

w <- abs(aux_model$fitted.value)

# # Régression pondérée avec les poids ajustés
FWLS_model <- lm(Sleep.Duration ~Age + Physical.Activity.Level + Stress.Level + Heart.Rate + Daily.Steps + Systolic + Gender_Male + Occupation_Type_Business
                   + Occupation_Type_Health + Occupation_Type_Technical + BMI.Category_Normal, weights = 1 / w, data = training_set)

# # Afficher le résumé du modèle FWLS
summary(FWLS_model)
white(FWLS_model,interactions = TRUE)


# Hypothèse d'Autocorrélation --------------------------------------------------

bgtest(brute_force,order=2)

# Hypothèse d'absence de valeurs influentes ------------------------------------

p <- length(coef(brute_force))-1  
n <- nrow(dataset)  

# Outliers wrt X ---------------------------------------------------------------

leverage <- hatvalues(brute_force)
plot(leverage, main = "Leverage Values")
abline(h = 2 * mean(leverage), col = "red")

rownames(dataset[which(leverage > (2*p)/n),])

# Outliers wrt Y ---------------------------------------------------------------

deleted_residuals <- rstudent(brute_force)

alpha <- 0.05

critical_value <- qt(1 - alpha /(2*n), df = n - p - 1, lower.tail = TRUE) # Seuil avec correction de Bonferroni 

rownames(dataset[which(abs(deleted_residuals) > critical_value),])


# Valeurs influentes -----------------------------------------------------------

# Calcul des distances de Cook pour chaque observation dans le modèle "brute_force"
cooks_distances <- cooks.distance(brute_force)
F_quantile <- qf(0.5, df1 = p, df2 = n - p)

# Identifier les observations influentes (celles dont D_i > F_quantile)
which(cooks_distances > F_quantile)

# Visuelement 
plot(cooks.distance(brute_force), main = "Cook's Distance", ylim =c(-0.1, 1))
abline(h = F_quantile, col = "red", lty = 2) 


# Hypothèse de normalité des résidus -------------------------------------------

jarque.bera.test(brute_force$residuals) #ne marche plus avec la nouvelle version de R.Studio


# Inférence robuste ############################################################

v_robust_2 <- vcovHAC(brute_force)

# Test de significativité ------------------------------------------------------

coeftest(brute_force,v_robust_2)

# Test de significativité d'un ensemble de coefficients ------------------------

linearHypothesis(brute_force, c("Occupation_Type_Business","Occupation_Type_Technical","Daily.Steps","Heart.Rate"),vcov. = v_robust_2)
linearHypothesis(brute_force, c("Occupation_Type_Technical","Daily.Steps"),vcov. = v_robust_2)

model <- lm(Sleep.Duration ~Age + Physical.Activity.Level + Stress.Level + Heart.Rate  + Systolic + Gender_Male + Occupation_Type_Business
            + Occupation_Type_Health + BMI.Category_Normal,data = training_set )

# Hypothèses pour ce modèle :

white(model) #hetéroscédasticité
v_robust_2 <- vcovHAC(model)
coeftest(model,v_robust_2)
bgtest(model) #autocorrélation
max(vif(model)) #multicolinéarité
jarque.bera.test(model$residuals) #normalité des résidus

# Qualité de ce modèle :
AIC(model)
variables <- names(model$coefficients)[-1]
pred_reduced <- predict(model,validation_set[,variables])
mse_reduced <- mean((validation_set$Sleep.Duration-pred_reduced)^2)
cat("MSE pour modèle réduit :", mse_reduced, "\n")



# Test d'une combinaison linéaire de deux coefficients -------------------------

v_robust_2 <- vcovHAC(brute_force)
hypothesis <- c("0.04*Heart.Rate -0.05*Age + -0.37*Stress.Level = 0")

linearHypothesis(brute_force, hypothesis, vcov. = v_robust_2)

# Qualité des prédictions modèle réduit-----------------------------------------


# Calcul des prédictions et des intervalles de prédiction à 95% pour le jeu de validation
predictions <- predict(model, newdata = validation_set, interval = "prediction", level = 0.95)

# Calcul de la couverture : vérification si la vraie valeur est dans l'intervalle de prédiction
coverage <- mean(validation_set$Sleep.Duration >= predictions[, "lwr"] & validation_set$Sleep.Duration <= predictions[, "upr"])

# Affichage de la couverture
coverage



plot1 <- ggplot(predictions, aes(x = fit)) +
  geom_ribbon(aes(ymin = lwr, ymax = upr), fill = "lightblue", alpha = 0.4) +
  geom_line(aes(y = fit), color = "blue", size = 1.2) +
  geom_point(aes(y = validation_set$Sleep.Duration), color = "darkgreen", shape = 16, size = 2) +
  labs(title = "Intervale de prédiction pour le modèle réduit",
       subtitle = "pour le set de validation",
       x = "Predicted Sleep Duration", 
       y = "True Sleep Duration") +
  theme_minimal() +
  scale_color_manual(name = "Legend", 
                     values = c("Predicted" = "blue", "True Values" = "darkgreen")) 

# Qualité des prédictions du modèle de base-------------------------------------

model <- brute_force

# Calcul des prédictions et des intervalles de prédiction à 95% pour le jeu de validation
predictions <- predict(model, newdata = validation_set, interval = "prediction", level = 0.95)

# Calcul de la couverture : vérification si la vraie valeur est dans l'intervalle de prédiction
coverage <- mean(validation_set$Sleep.Duration >= predictions[, "lwr"] & validation_set$Sleep.Duration <= predictions[, "upr"])

# Affichage de la couverture
coverage


plot2 <- ggplot(predictions, aes(x = fit)) +
  geom_ribbon(aes(ymin = lwr, ymax = upr), fill = "lightblue", alpha = 0.4) +
  geom_line(aes(y = fit), color = "blue", size = 1.2) +
  geom_point(aes(y = validation_set$Sleep.Duration), color = "darkgreen", shape = 16, size = 2) +
  labs(title = "Intervale de prédiction pour le modèle (I)",
       subtitle = "pour le set de validation",
       x = "Predicted Sleep Duration", 
       y = "True Sleep Duration") +
  theme_minimal() +
  scale_color_manual(name = "Legend", 
                     values = c("Predicted" = "blue", "True Values" = "darkgreen")) 

grid.arrange(plot1,plot2,ncol=2)
