# Personalized Healthcare Script

##############################################################

# Load Packages

rm(list=ls())
# install.packages("pacman") # install multiple package at one time
pacman::p_load(mlr3, mlr3proba, mlr3pipelines, 
               mlr3learners, mlr3viz,mlr3tuning, 
               mlr3benchmark, mlr3misc,mlr3extralearners,
               R6, data.table, lgr, uuid, mlbench, digest, 
               backports, checkmate, paradox, reticulate, 
               keras, devtools, survival, rms, summarytools, knitr) # takes approx 4 mins

install_github("binderh/CoxBoost")

library(mlr3extralearners)
install_learners('surv.coxboost') # TODO

library(mlr3learners) #ranger
library(mlr3proba) #coxph

##############################################################

# Import dataset German Breast Cancer Study

#?mlr3proba::gbcs # uncomment for dataset infos

gbcs <- mlr3proba::gbcs

gbcs2 <- gbcs[,c(5:12,15:16)]
summarytools::dfSummary(gbcs2, 
                        graph.col = F, 
                        valid.col = F
)

head(gbcs) # print example

##############################################################

# Data Cleaning

## Normalize/Scale Continuous variables

gbcs2$age <- scale(gbcs2$age)
gbcs2$menopause <- gbcs2$menopause-1
gbcs2$hormone <- gbcs2$hormone-1
gbcs2$size <- scale(gbcs2$size)
gbcs2$grade1 <- ifelse(gbcs2$grade==1, 1,0)
gbcs2$grade2 <- ifelse(gbcs2$grade==2, 1,0)
gbcs2$grade3 <- ifelse(gbcs2$grade==3, 1,0)
gbcs2$grade <- NULL
gbcs2$nodes <- scale(gbcs2$nodes)
gbcs2$prog_recp <- scale(gbcs2$prog_recp)
gbcs2$estrg_recp <- scale(gbcs2$estrg_recp)

##############################################################

# Train/Test Split

set.seed(123)
train_set = sample(nrow(gbcs2), 0.8 * nrow(gbcs2))
#str(train_set)
test_set = setdiff(seq_len(nrow(gbcs2)), train_set)

## train/test set initialization and summary

train_gbcs <- gbcs2[train_set, ]
summarytools::dfSummary(train_gbcs, 
                        graph.col = F, 
                        valid.col = F
)


test_gbcs <- gbcs2[test_set, ]
summarytools::dfSummary(test_gbcs, 
                        graph.col = F, 
                        valid.col = F
)


##############################################################

# Observe S(t) in Train/Test set

KM_train <- survfit(Surv(survtime, censdead) ~ 1, data = train_gbcs)
KM_train

KM_test <- survfit(Surv(survtime, censdead) ~ 1, data = test_gbcs)
KM_test

par(mfrow=c(1,2))
plot(KM_train, xlab = "Time to Death (days)", ylab = "Survival Probability", 
     main = "GBCS Train Data")
abline(h=.5)
plot(KM_test, xlab = "Time to Death (days)", ylab = "Survival Probability", 
     main = "GBCS Test Data")
abline(h=.5)

##############################################################

# Cox Model

fit <- coxph(Surv(survtime, censdead) ~ age + menopause + hormone + size + grade1 + grade2 + nodes + prog_recp + estrg_recp, data = train_gbcs)
summary(fit)

check_PH <- cox.zph(fit, transform = "km")
check_PH

ND <- data.frame(age = 0, menopause = 1, hormone = 2,
                 size = 0, grade1 = c(1,0,0), grade2=c(0,1,0), grade3=c(0,0,1), nodes = 0, prog_recp=0, estrg_recp=0)

surv_probs_Cox <- survfit(fit, newdata = ND)
surv_probs_Cox

summary(surv_probs_Cox, times = 500)

plot(surv_probs_Cox, col = c("red", "blue", "green"),
     xlab = "Follow-Up Time (days)", ylab = "Survival Probabilities")


