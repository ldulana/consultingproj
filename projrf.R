library(tidyverse)
library(randomForest)
library(caret)


# Data

full <- read_csv("final.csv")


# transform appeals to single column

full$appealed19[is.na(full$appealed19)] <- 0
full$appealed20[is.na(full$appealed20)] <- 0
full$appealed21[is.na(full$appealed21)] <- 0
full$appealed <- ifelse((full$appealed19 + full$appealed20 +
                           full$appealed21) > 0, 1, 0)


# separate sales

sales <- full %>% filter(!is.na(full$sale_price))


# adjust to factor variables select data for model

sales$bld_type <- as.factor(sales$bld_type)
sales$appraiser <- as.factor(sales$appraiser)
sales$nbhd <- as.factor(sales$nbhd)
sales$qual <- as.factor(sales$qual)
sales$cond <- as.factor(sales$cond)
sales$kitchen_rating <- as.factor(sales$kitchen_rating)
sales$full_bath_rating <- as.factor(sales$full_bath_rating)
sales$half_bath_rating <- as.factor(sales$half_bath_rating)


salesrfdat <- sales %>%
  select(condition, hb_rating, fb_rating, k_rating, qualscore,
         prop_id, bld_type, appraiser, kitchen_ct, full_bath_ct,
         half_bath_ct, year_built, finished_area, land_sf, 
         sale_price, zip, geo_tract, lat, long, appealed)


# partition data into train and test

set.seed(1111)

rftrainindex <- createDataPartition(salesrfdat$sale_price, p = 0.7, list = FALSE)
rf_train <- salesrfdat[rftrainindex,]
rf_test <- salesrfdat[-rftrainindex,]


# Random Forest model

set.seed(2222)
sales_rf <- randomForest(sale_price ~ ., data = rf_train, mtry = 6,
                         ntree = 1000, importance = TRUE)
sales_hat <- predict(sales_rf, newdata = rf_test)
rmserf <- sqrt(mean((rf_test$sale_price - sales_hat)^2))

varImpPlot(sales_rf)