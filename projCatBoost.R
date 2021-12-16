
# Ran on google Colab notebook

devtools::install_url('https://github.com/catboost/catboost/releases/download/v1.0.3/catboost-R-Linux-1.0.3.tgz')


library(catboost)
library(tidyverse)

final <- read_csv("final.csv")
#head(final)

df <- final %>%
  select(qualscore, bld_type, sale_price, contains("appealed"), lat, long, appraiser, zip, nbhd, condition, hb_rating, fb_rating, k_rating, kitchen_ct, full_bath_ct, half_bath_ct, year_built, finished_area, land_sf) %>%
  mutate(appealed19 = ifelse(is.na(appealed19), "no", "yes"),
         appealed20 = ifelse(is.na(appealed20), "no", "yes"),
         appealed21 = ifelse(is.na(appealed21), "no", "yes"),
         appealed = ifelse(appealed19 == "yes" | appealed20 == "yes" | appealed21 == "yes",
                           "yes", "no")) %>%
  select(-(appealed19:appealed21))

head(df)

train <- df %>% filter(!is.na(sale_price)) %>% mutate(across(where(is.character), factor))
test <- df %>% filter(is.na(sale_price)) %>% mutate(across(where(is.character), factor))



# fit a catboost on train


set.seed(2223)
sample <- sample.int(n = nrow(train), size = floor(.70*nrow(train)), replace = FALSE)
d_train <- train[sample,]
d_test <- train[-sample,]

y_train <- d_train$sale_price
y_test <- d_test$sale_price
x_train <- select(d_train, -sale_price)
x_test <- select(d_test, -sale_price)

#y_train <- select(train, sale_price)
#x_train <- select(train, -sale_price)
#x_test <- select(test, -sale_price)
#map(list(x_train, x_test), dim)

train_pool <- catboost.load_pool(data = x_train, label = y_train)
test_pool <- catboost.load_pool(data = x_test)

fit <- catboost.train(learn_pool = train_pool,
                      params = list(random_seed = 1234, learning_rate = 0.05))



# Variable importance

imp_mat <- catboost.get_feature_importance(fit)

data.frame(var = rownames(imp_mat),
           imp = imp_mat[,1]) %>%
  mutate(var = fct_reorder(var, imp)) %>%
  ggplot(aes(imp, var)) +
  geom_col() +
  theme_light()



# RMSE 

train_preds <- catboost.predict(fit, train_pool)
# length(train_preds)
# dim(d_train)
#train$pred_price <- train_preds

test_preds <- catboost.predict(fit, test_pool)
# length(test_preds)
# dim(d_test)
#test$pred_price <- test_preds

#train %>%
#  ggplot(aes(sale_price, pred_price)) +
#  geom_point() +
#  geom_smooth(method = "lm")

sqrt(mean((train_preds - d_train$sale_price)^2))
sqrt(mean((test_preds - d_test$sale_price)^2))