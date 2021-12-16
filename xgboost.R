library(tidyverse)
library(tidymodels)

# prep

final <- read_csv("final.csv")

d <- final %>% 
  select(condition,
         hb_rating,
         fb_rating,
         k_rating,
         qualscore,
         bld_type,
         appraiser,
         nbhd,
         contains("_ct"),
         sale_price,
         year_built,
         finished_area,
         land_sf,
         zip,
         lat,
         long) %>% 
  mutate(zip = as.character(zip),
         zip = fct_lump_min(zip, min = 1000),
         bld_type = fct_lump_min(bld_type, min = 1000),
         across(where(is.character), factor))

d_train <- d %>%
  filter(!is.na(sale_price))

d_test <- d %>%
  filter(is.na(sale_price))

# Modeling

# partition

set.seed(1234)
xgboost_split <- initial_split(data = d_train,
                               prop = 0.7,
                               strata = sale_price)

xgboost_train <- training(xgboost_split)
xgboost_validation <- testing(xgboost_split)

# parsnip_addin()

# specification
xgboost_spec <- boost_tree(tree_depth = tune(),
                           trees = 1000,
                           learn_rate = 0.01,
                           min_n = tune(),
                           mtry = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('regression')


# recipe

xgboost_rec <- recipe(sale_price ~ ., data = xgboost_train) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# xgboost_rec %>% 
#   prep() %>% 
#   juice() %>% 
#   View()


# tune

xgboost_workflow <- workflow() %>% 
  add_recipe(xgboost_rec) %>% 
  add_model(xgboost_spec)

set.seed(123)
xgboost_cv <- vfold_cv(data = xgboost_train,
                       v = 5,
                       strata = sale_price)

# create grid separately
xgboost_grid <- crossing(mtry = c(15, 20, 25),
                         tree_depth = c(9, 12, 15),
                         min_n = c(12))

doParallel::registerDoParallel(cores = 7)
set.seed(12)
xgboost_tune <- tune_grid(object = xgboost_workflow,
                          resamples = xgboost_cv,
                          metrics = metric_set(rmse),
                          control = control_grid(save_pred = TRUE,
                                                 save_workflow = TRUE,
                                                 extract = extract_fit_engine),
                          grid = xgboost_grid)

autoplot(xgboost_tune)

xgboost_best <- xgboost_tune %>% 
  select_best("rmse")

xgboost_best

# finalize the model based on the selected best/optimal hyperparam values
xgboost_final <- xgboost_workflow %>% 
  finalize_workflow(xgboost_best)

# fit a final model on the entire train set
xgboost_last <- xgboost_final %>% 
  last_fit(xgboost_split)

# evaluate (on the test set)

xgboost_last %>% 
  collect_predictions() %>% 
  rmse(.pred, sale_price)

xgboost_last %>% 
  collect_predictions() %>% 
  rsq(.pred, sale_price)

xgboost_last %>% 
  collect_predictions() %>% 
  ggplot(aes(.pred, sale_price)) +
  geom_point(alpha = 0.7, color = "midnightblue", size = 2) +
  geom_abline(slope = 1, intercept = 0)


# predict on the holdout

xgboost_fit_final <- xgboost_final %>% 
  fit(d_train)


xgboost_preds <- xgboost_fit_final %>% 
  augment(d_test)

# variable importance

xgboost_mod <- xgboost_fit_final %>% 
  extract_fit_parsnip()

library(vip)
vip(xgboost_mod$fit, num_features = 15)








