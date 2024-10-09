source("./helping_functions.R")
# Formulas
formulas = list(pred = as.formula(SalePrice ~ PRom + BRA + lng + lat + Altitude  + NumberOfBedrooms + 
                                    Floor + YearsSinceBuilt + CoastDistance + LakeDistance + NumberOfUnitsOnAddress + 
                                    CityDistrict + HomesNearby + OtherBuildingsNearby + Balcony + Elevator),
                sigma_hat = sigma_hat = as.formula(abs_res ~ PRom + as.factor(CityDistrict) + as.factor(SaleMonth))) # + 



prediction_models = c("random forest", "lightgbm")
sigmas = c("cqr_lightgbm", "cqr_rf", "yhat", "lm", "one")
weights = c("none", "spatial", "spatial_nn", "mondrian")


alphas = 0.9
sim = 10


res_df = data.frame()
res_per_cd = data.frame()
test_res = data.frame()

for(a in alphas){
  for(s in 1:sim){
    cat(" ==== ITERATION ", s, " === \n")
    set.seed(s)
    rows = sample(1:NROW(df_oslo), size = 6000, replace = FALSE)
    df_subset = df_oslo %>% dplyr::slice(rows)
    
    df_original = DataSplit(df_subset)
    dfs = list(full = df_subset, train = df_original$train_df, calibration = df_original$calibration_df, test = df_original$test_df)
    
    one_sim = SplitCP_Wrapper_New(dfs = dfs,
                                  formulas = formulas, 
                                  alpha = a,
                                  prediction_models = prediction_models,  
                                  sigma_methods = sigmas, 
                                  weight_methods = weights, 
                                  my_seed = s)
    
    all_test_res = one_sim$test
    
    # Test res 
    test_res = rbind(test_res, EvaluatePrediction(all_test_res) %>% mutate(seed = s))
    
    # Coverage res
    this_res_df = EvaluateConfidenceIntervals(all_test_res, alphas = a) %>% mutate(seed = s)
    this_res_per_cd = EvaluateConfidenceIntervalsPerCityDistrict(all_test_res, alphas = a) %>% mutate(seed = s)
    
    res_df = rbind(res_df, this_res_df)
    res_per_cd = rbind(res_per_cd, this_res_per_cd)
    
  }
}


