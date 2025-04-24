# Load required libraries
library(tidymodels)
library(tidyverse)
library(lubridate)
library(zoo)

# Data prep (simplified and clean)
Wood_Thrush <- Wood_Thrush |>
  select(COUNTRY, `COUNTRY CODE`, STATE, `OBSERVATION COUNT`, `OBSERVATION DATE`, LATITUDE, LONGITUDE) |>
  mutate(`OBSERVATION DATE` = as.Date(`OBSERVATION DATE`, format = "%m/%d/%Y"),
         Year = year(`OBSERVATION DATE`))

Annual_Weather <- AllWeatherData |>
  select(Year, Annual, STATE)

Annua_CO2_Emissions <- Annua_CO2_Emissions |>
  select(Year, `Annual CO₂ emissions`)

Main_Data <- Wood_Thrush |>
  inner_join(Annual_Weather, by = c("Year", "STATE")) |>
  inner_join(Annua_CO2_Emissions, by = "Year") |>
  mutate(DayOfYear = yday(`OBSERVATION DATE`)) |>
  drop_na(Annual, `Annual CO₂ emissions`, DayOfYear)

# Creating the Datasets

Spring_Arrival <- Main_Data |>
  group_by(year(`OBSERVATION DATE`)) |>
  slice_min(`OBSERVATION DATE`, with_ties = FALSE)

Fall_Departure <- Main_Data |>
  group_by(year(`OBSERVATION DATE`)) |>
  slice_max(`OBSERVATION DATE`, with_ties = FALSE)

# Splitting Data
set.seed(123)
WT_split <- initial_split(Main_Data, prop = 0.8)
WT_train <- training(WT_split)
WT_test  <- testing(WT_split)
WT_cv <- vfold_cv(WT_train, v = 10)

# Recipe 
CO2_rec <- recipe(formula = DayOfYear ~ `Annual CO₂ emissions`, data = WT_train) |>
  step_log(all_predictors()) |>
  step_naomit(all_predictors(), all_outcomes())

Temp_rec <- recipe(formula = DayOfYear ~ Annual, data = WT_train) |>
  step_log(all_predictors()) |>
  step_naomit(all_predictors(), all_outcomes())

# Define models
lm_model <- linear_reg() |> set_engine("lm") |> set_mode("regression")
rf_model <- rand_forest() |> set_engine("ranger") |> set_mode("regression")
xg_model <- boost_tree() |> set_engine("xgboost") |> set_mode("regression")

# CO2 workflows
wf_lm_CO2 <- workflow() |>
  add_model(lm_model) |>
  add_recipe(CO2_rec)

wf_rf_CO2 <- workflow() |>
  add_model(rf_model) |>
  add_recipe(CO2_rec)

wf_xg_CO2 <- workflow() |>
  add_model(xg_model) |>
  add_recipe(CO2_rec)

# Temperature workflows
wf_lm_Temp <- workflow() |>
  add_model(lm_model) |>
  add_recipe(Temp_rec)

wf_rf_Temp <- workflow() |>
  add_model(rf_model) |>
  add_recipe(Temp_rec)

wf_xg_Temp <- workflow() |>
  add_model(xg_model) |>
  add_recipe(Temp_rec)

# CO2: Linear Model
res_lm_CO2 <- fit_resamples(
  wf_lm_CO2,
  resamples = WT_cv,
  metrics = metric_set(rmse, rsq),
  control = control_resamples(save_pred = TRUE)
)

# CO2: Random Forest
res_rf_CO2 <- fit_resamples(
  wf_rf_CO2,
  resamples = WT_cv,
  metrics = metric_set(rmse, rsq),
  control = control_resamples(save_pred = TRUE)
)

# CO2: Boosted Tree
res_xg_CO2 <- fit_resamples(
  wf_xg_CO2,
  resamples = WT_cv,
  metrics = metric_set(rmse, rsq),
  control = control_resamples(save_pred = TRUE)
)

# Temp: Linear Model
res_lm_Temp <- fit_resamples(
  wf_lm_Temp,
  resamples = WT_cv,
  metrics = metric_set(rmse, rsq),
  control = control_resamples(save_pred = TRUE)
)

# Temp: Random Forest
res_rf_Temp <- fit_resamples(
  wf_rf_Temp,
  resamples = WT_cv,
  metrics = metric_set(rmse, rsq),
  control = control_resamples(save_pred = TRUE)
)

# Temp: Boosted Tree
res_xg_Temp <- fit_resamples(
  wf_xg_Temp,
  resamples = WT_cv,
  metrics = metric_set(rmse, rsq),
  control = control_resamples(save_pred = TRUE)
)

# Best Models: XG Boost and Random Forest

# Next steps, either tuning XG Boost or going for random forest
