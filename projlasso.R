library(tidyverse)
library(glmnet)
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


# select variables for LASSO model and change character to factor

salesrfdat <- sales %>%
  select(condition, hb_rating, fb_rating, k_rating, qualscore,
         prop_id, bld_type, appraiser, kitchen_ct, full_bath_ct,
         half_bath_ct, year_built, finished_area, land_sf, 
         sale_price, zip, geo_tract, lat, long, appealed)


salesrfdat <- salesrfdat %>% 
  filter(!is.na(sale_price)) %>% 
  mutate(across(where(is.character), factor))


# Partition sales data into train and test

set.seed(111)
trainindex <- createDataPartition(salesrfdat$sale_price, p = 0.7, list = FALSE)
sales_train <- salesrfdat[trainindex,]
sales_test <- salesrfdat[-trainindex,]

# create model matrix for LASSO

train.mat <- model.matrix(sales_train$sale_price ~ ., data = sales_train)[,-1]
test.mat <- model.matrix(sales_test$sale_price ~ ., data = sales_test)[,-1]
full.mat <- model.matrix(salesrfdat$sale_price ~ ., data = salesrfdat)[,-1]

# Cross Validation

grid <- exp(seq(-15, 9, .05))

lassocv <- cv.glmnet(train.mat, sales_train$sale_price, alpha = 1,
                     lambda = grid)

plot(lassocv)

# LASSO model

lassomod <- glmnet(train.mat, sales_train$sale_price, alpha = 1,
                   lambda = lassocv$lambda.min)
lassopred <- predict(lassomod, s = lassocv$lambda.min,
                     newx = test.mat)

lassormse <- sqrt(mean((sales_test$sale_price - lassopred)^2))

coef(lassocv, s = lassocv$lambda.min)


