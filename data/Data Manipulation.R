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

# Splitting Data
set.seed(123)
WT_split <- initial_split(Main_Data, prop = 0.8)
WT_train <- training(WT_split)
WT_test  <- testing(WT_split)
WT_cv <- vfold_cv(WT_train, v = 10)

# Define models
lm_model <- linear_reg() |> set_engine("lm") |> set_mode("regression")
rf_model <- rand_forest() |> set_engine("ranger") |> set_mode("regression")
xg_model <- boost_tree() |> set_engine("xgboost") |> set_mode("regression")

wf_set <- workflow_set(
  preproc = list(formula = DayOfYear ~ Annual + `Annual CO₂ emissions`),
  models = list(
    linear = lm_model,
    random_forest = rf_model,
    xgboost = xg_model
  )
)

wf_results <- wf_set |> 
  workflow_map("fit_resamples", resamples = WT_cv)

autoplot(wf_results)
collect_metrics(wf_results)

# Best Models: XG Boost and Random Forest

# Next steps, either tuning XG Boost or going for random forest
