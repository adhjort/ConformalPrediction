
TrainModels = function(train_df, 
                       calibration_df, 
                       test_df, 
                       model_formula, 
                       prediction_model){
  
  
  ### Initiate
  oracle_score_train = rep(NA, NROW(train_df))
  oracle_score_calibration = rep(NA, NROW(calibration_df))
  oracle_score_test = rep(NA, NROW(test_df))
  
  pvalue_train = rep(NA, NROW(train_df))
  pvalue_calibration = rep(NA, NROW(calibration_df))
  pvalue_test = rep(NA, NROW(test_df))
  
  
  if (prediction_model %in% c("xgb")){
    cat("STEP 1: Training XGB \n")
    
    train_sparse = sparse.model.matrix(model_formula, data = train_df)
    dtrain = xgb.DMatrix(data = train_sparse, label = train_df$SalePrice)
    
    calibration_sparse = sparse.model.matrix(model_formula, data = calibration_df)
    dcalibrate = xgb.DMatrix(data = calibration_sparse, label = calibration_df$SalePrice)
    
    test_sparse = sparse.model.matrix(model_formula, data = test_df)
    dtest = xgb.DMatrix(data = test_sparse)
    
    pred_model = xgb.train(data = dtrain,
                           params = list(eta = 0.03, max_depth = 4),
                           nrounds = 1000,
                           base_score = mean(train_df$SalePrice), 
                           verbose = FALSE)
    
    cat("STEP 2: Predicting on train, calibration and test \n")
    
    train_df$predictions = predict(pred_model, dtrain)
    calibration_df$predictions = predict(pred_model, dcalibrate) 
    test_df$predictions = predict(pred_model, dtest)
  } 
  else if(prediction_model %in% c("lm")){
    pred_model = lm(formula = model_formula, 
                    data = train_df)
    
    train_df$predictions = predict(pred_model, train_df)
    calibration_df$predictions = predict(pred_model, calibration_df)
    test_df$predictions = predict(pred_model, test_df)
  }
  else if (prediction_model %in% c("random forest")){
    cat("STEP 1: Training random forest \n")
    
    pred_model = ranger::ranger(formula = model_formula, 
                                data = train_df)
    
    
    cat("STEP 2: Predicting on train, calibration and test \n")
    
    train_df$predictions = predict(pred_model, train_df)$predictions
    calibration_df$predictions = predict(pred_model, calibration_df)$predictions
    test_df$predictions = predict(pred_model, test_df)$predictions
  }
  else if (prediction_model %in% c("spatial")){
    
    cat("Add City district dummy \n")
    train_dummy = caret::dummyVars(" ~ .", data = train_df %>% select(CityDistrict), fullRank = TRUE)
    calibration_dummy = caret::dummyVars(" ~ .", data = calibration_df %>% select(CityDistrict), fullRank = TRUE)
    test_dummy = caret::dummyVars(" ~ .", data = test_df %>% select(CityDistrict), fullRank = TRUE)
    
    train_df_dummy = data.frame(predict(train_dummy, train_df))
    calibration_df_dummy = data.frame(predict(calibration_dummy, calibration_df))
    test_df_dummy = data.frame(predict(test_dummy, test_df))
    
    # Get relevant column names
    relevant_columns = all.vars(model_formula)[-1]
    
    # Get X matrix
    X_train = train_df %>% dplyr::select(relevant_columns) %>% select(-CityDistrict, -lng, -lat) %>% cbind(train_df_dummy) %>% as.matrix()
    X_calibration = calibration_df %>% dplyr::select(relevant_columns) %>% select(-CityDistrict,  -lng, -lat) %>% cbind(calibration_df_dummy) %>% as.matrix()
    X_test = test_df %>% dplyr::select(relevant_columns) %>% select(-CityDistrict,-lng, -lat) %>% cbind(test_df_dummy) %>% as.matrix()
    
    N_train = NROW(X_train)
    N_calibration = NROW(X_calibration)
    N_test = NROW(X_test)
    
    # Get L matrix
    L_train = train_df %>% select(Longitude, Latitude) %>% as.matrix()
    L_calibration = calibration_df %>% select(Longitude, Latitude) %>% as.matrix()
    L_test = test_df %>% select(Longitude, Latitude) %>% as.matrix()
    
    # Get Y matrix
    Y_train = train_df %>% select(SalePrice)%>% as.matrix()
    Y_calibration = calibration_df %>% select(SalePrice)%>% as.matrix()
    Y_test = test_df %>% select(SalePrice)%>% as.matrix()
    
    H_train = as.matrix(dist(L_train, method = "manhattan"))/1000 # distance in kilometers
    
    # Get MLE 
    cat("\t Doing hyperparameter optimization on profiled log likelihood in 2D \n")
    mle_params = MLE_gridsearch(H = H_train, X = X_train, Y = Y_train, gridsize = 10, return_full_df = FALSE)
    rho_hat = exp(mle_params$estimate[1])
    sigma_eps_hat = exp(mle_params$estimate[2])
    
    cat("MLE found: \n")
    cat("\t rho_hat: ", rho_hat, "\n")
    cat("\t sigma_eps_hat: ", sigma_eps_hat, "\n")
    
    # Train
    train_list = GetBetaAndSigma(X = X_train, Y = Y_train, L = L_train, rho_mle = rho_hat, sigma_eps_mle = sigma_eps_hat)
    
    beta_mle = train_list$beta
    Sigma_train = train_list$Sigma
    sigma_hat = train_list$sigma
    
    # Get calibration Sigma
    H_calibration = as.matrix(dist(L_calibration, method = "manhattan"))/1000 # distance in kilometers
    Sigma_calibration <- sigma_hat*exp(-H_calibration*rho_hat) + sigma_eps_hat*diag(N_calibration)
    
    # Get test Sigma
    H_test = as.matrix(dist(L_test, method = "manhattan"))/1000 # distance in kilometers
    Sigma_test <- sigma_hat*exp(-H_test*rho_hat) + sigma_eps_hat*diag(N_test)
    
    cat("\t sigma_hat: ", sigma_hat, "\n")
    
    # Predict
    cat("STEP 2: Predicting on train, calibration and test \n")
    train_df$predictions = X_train%*%beta_mle
    calibration_df$predictions = X_calibration%*%beta_mle
    test_df$predictions = X_test%*%beta_mle
    
    # Create p values 
    Sigma_train_inv = solve(Sigma_train)
    Sigma_calibration_inv = solve(Sigma_calibration)
    Sigma_test_inv = solve(Sigma_test)
    
    oracle_score_train = abs(expm::sqrtm(Sigma_train_inv)%*%(Y_train - X_train%*%beta_mle))
    oracle_score_calibration = abs(expm::sqrtm(Sigma_calibration_inv)%*%(Y_calibration - X_calibration%*%beta_mle))
    oracle_score_test = abs(expm::sqrtm(Sigma_test_inv)%*%(Y_test - X_test%*%beta_mle))
    
    pvalue_test = rep(NA, NROW(test_df))
    
    for(pp in 1:NROW(test_df)){
      pvalue_test[pp] = mean(oracle_score_calibration <= oracle_score_test[pp])
    }
  } 
    

  ### Add predictions
  train_df$prediction_model = prediction_model
  calibration_df$prediction_model = prediction_model
  test_df$prediction_model = prediction_model
  
  ### Add oracle scores
  train_df$oracle_score = oracle_score_train
  calibration_df$oracle_score = oracle_score_calibration
  test_df$oracle_score = oracle_score_test
  
  
  ### Add p values
  train_df$pvalue = rep(NA, NROW(train_df))
  calibration_df$pvalue = rep(NA, NROW(calibration_df))
  test_df$pvalue = pvalue_test
  
  return(list(train_df = train_df, 
              calibration_df = calibration_df, 
              test_df = test_df))
}



profiled_loglik2 = function(param, H, X, Y){
  
  N = NROW(X)
  p = NCOL(X)
  
  #cat("N: ", N, "\n")
  rho = exp(param[1])
  sigma_eps = exp(param[2])
  
  # Schabenberger (2006): Sigma = sigma^2*SigmaStar
  # We have Sigma = sigma_eps^2 + sigma^2*SigmaStar = a*sigma^2*I + sigma^2*SigmaStar = sigma^2*(a*I + SigmaStar)
  Sigma = exp(-H*rho) + sigma_eps*diag(N)
  Sigma_inv <- chol2inv(chol(Sigma + 1e-9*diag(N)))
  
  r <- Y - X%*%solve(t(X)%*%Sigma_inv%*%X + 1e-6*diag(p))%*%t(X)%*%Sigma_inv%*%Y
  sigma_hat_2 <- (1/N) * t(r)%*%Sigma_inv%*%r
  
  # Approximate log(det(Sigma))
  # https://stats.stackexchange.com/questions/71381/estimating-parameters-in-multivariate-classification-resulting-zero-determinant
  
  eigenvalues = eigen(Sigma, only.values = TRUE)$values
  eigenvalues_cutoff = eigenvalues[which(eigenvalues > 1e-9)]
  log_det_Sigma_estimate = sum(log(eigenvalues_cutoff))
  
  
  # Eq. 5.44: 
  #mle = log(det(Sigma)) + N*log(sigma_hat_2) + N*(log(2*pi) - 1)
  mle = log_det_Sigma_estimate + N*log(sigma_hat_2) + N*(log(2*pi) - 1)
  
  return(mle)
}


MLE_gridsearch = function(H, X, Y, gridsize = 10, return_full_df = FALSE){
  
  param_grid1 = seq(from = 0, to = 5, length.out = gridsize)
  param_grid2 = seq(from = -2, to = 1, length.out = gridsize)
  mle_eval = expand.grid(p1 = param_grid1, p2 = param_grid2) 
  mle_eval$eval = NA
  
  for(j in 1:NROW(mle_eval)){
    
    if(j %% 10 == 0){ cat("j: ", j, "\n")}
    mle_eval$eval[j] = profiled_loglik2(param = as.numeric(mle_eval[j, 1:2]), 
                                        H = H, X = X, Y = Y)
  }
  
  # Find
  best_row = which(mle_eval$eval == min(mle_eval$eval))
  
  if(return_full_df){
    return(mle_eval)
  }else{
    return(list(estimate = c(mle_eval$p1[best_row], mle_eval$p2[best_row])))
  }
  
}

MLE_gridsearch_sigma = function(H, X, Y, gridsize = 10, rho, return_full_df = FALSE){
  
  param_grid1 = seq(from = -2, to = 1, length.out = gridsize)
  mle_eval = data.frame(p1 = param_grid1)
  mle_eval$eval = NA
  
  for(j in 1:NROW(mle_eval)){
    cat("j: ", j, "\n")
    mle_eval$eval[j] = profiled_loglik2(param = as.numeric(c(rho, mle_eval$p1[j])), 
                                        H = H, X = X, Y = Y)
  }
  
  # Find
  best_row = which(mle_eval$eval == min(mle_eval$eval))
  rho_hat = exp(rho)
  sigma_eps = exp(mle_eval$p1[best_row])
  
  if(return_full_df){
    return(mle_eval)
  }else{
    return(list(estimate = mle_eval$p1[best_row]))
  }
  
}

CreateSyntheticData = function(df_oslo, model_formula, data_generating_model, seed){
  
  dfs = DataSplit(df_oslo = df_oslo, seed = seed)
  
  train_df = dfs$train_df
  calibration_df = dfs$calibration_df
  test_df = dfs$test_df
  
  # Transform to dummy
  # https://stackoverflow.com/questions/48649443/how-to-one-hot-encode-several-categorical-variables-in-r
  train_dummy = caret::dummyVars(" ~ .", data = train_df %>% select(CityDistrict), fullRank = TRUE)
  calibration_dummy = caret::dummyVars(" ~ .", data = calibration_df %>% select(CityDistrict), fullRank = TRUE)
  test_dummy = caret::dummyVars(" ~ .", data = test_df %>% select(CityDistrict), fullRank = TRUE)
  
  
  train_df_dummy = data.frame(predict(train_dummy, train_df))
  calibration_df_dummy = data.frame(predict(calibration_dummy, calibration_df))
  test_df_dummy = data.frame(predict(test_dummy, test_df))
  
  # Get relevant column names
  relevant_columns = all.vars(model_formula)[-1]
  
  # Get X matrix
  X_train = train_df %>% dplyr::select(relevant_columns) %>% select(-CityDistrict, -lng, -lat) %>% 
    cbind(train_df_dummy) %>% as.matrix()
  
  X_calibration = calibration_df %>% dplyr::select(relevant_columns) %>% select(-CityDistrict,  -lng, -lat) %>% 
    cbind(calibration_df_dummy) %>% as.matrix()
  
  X_test = test_df %>% dplyr::select(relevant_columns) %>% select(-CityDistrict,-lng, -lat) %>% 
    cbind(test_df_dummy) %>% as.matrix()
  
  N_train = NROW(X_train)
  N_calibration = NROW(X_calibration)
  N_test = NROW(X_test)
  
  # Get L matrix
  L_train = train_df %>% select(Longitude, Latitude) %>% as.matrix()
  L_calibration = calibration_df %>% select(Longitude, Latitude) %>% as.matrix()
  L_test = test_df %>% select(Longitude, Latitude) %>% as.matrix()
  
  H_train = as.matrix(dist(L_train, method = "manhattan"))/1000 # distance in kilometers
  H_calibration = as.matrix(dist(L_calibration, method = "manhattan"))/1000 # distance in kilometers
  H_test = as.matrix(dist(L_test, method = "manhattan"))/1000 # distance in kilometers
  
  # Get Y matrix
  Y_train = train_df %>% select(SalePrice) %>% as.matrix()
  Y_calibration = calibration_df %>% select(SalePrice) %>% as.matrix()
  Y_test = test_df %>% select(SalePrice) %>% as.matrix()
  
  if(data_generating_model == "spatial"){
    
    # Generate data
    
    mle_params = MLE_gridsearch(H = H_train, X = X_train, Y = Y_train, gridsize = 10, return_full_df = FALSE)
    rho_hat = exp(mle_params$estimate[1])
    sigma_eps_hat = exp(mle_params$estimate[2])
    
    cat("MLE found: \n")
    cat("\t rho: ", rho_hat, "\n")
    cat("\t sigma_eps: ", sigma_eps_hat, "\n")
    
    # Train
    train_list = GetBetaAndSigma(X = X_train, Y = Y_train, L = L_train, rho_mle = rho_hat, sigma_eps_mle = sigma_eps_hat)
    calibration_list = GetBetaAndSigma(X = X_calibration, Y = Y_calibration, L = L_calibration, rho_mle = rho_hat, sigma_eps_mle = sigma_eps_hat)
    test_list = GetBetaAndSigma(X = X_test, Y = Y_test, L = L_test, rho_mle = rho_hat, sigma_eps_mle = sigma_eps_hat)
    
    beta_mle = train_list$beta
    Sigma_train = train_list$Sigma
    Sigma_calibration = calibration_list$Sigma
    Sigma_test = test_list$Sigma
    
    sigma_hat = train_list$sigma
    sigma_calibration = calibration_list$sigma
    sigma_test = test_list$sigma
    
    cat(" \t sigma_hat: ", sigma_hat, "\n")
    
  } 
  else if(data_generating_model == "spatial_custom_rho_and_sigma_eps"){
    
    # Generate data with found MLE
    sigma_eps_hat = 1.1
    rho_hat = 4.0 
    
    # Train
    train_list = GetBetaAndSigma(X = X_train, Y = Y_train, L = L_train, rho_mle = rho_hat, sigma_eps_mle = sigma_eps_hat)
    beta_mle = train_list$beta
    Sigma_train = train_list$Sigma
    sigma_hat = train_list$sigma
    
    # Get calibration Sigma
    H_calibration = as.matrix(dist(L_calibration, method = "manhattan"))/1000 # distance in kilometers
    Sigma_calibration <- sigma_hat*exp(-H_calibration*rho_hat) + sigma_eps_hat*diag(N_calibration)
    
    # Get test Sigma
    H_test = as.matrix(dist(L_test, method = "manhattan"))/1000 # distance in kilometers
    Sigma_test <- sigma_hat*exp(-H_test*rho_hat) + sigma_eps_hat*diag(N_test)
    
    cat(" \t sigma_hat: ", sigma_hat, "\n")
    
  }
  
  y_sampled_train = mvtnorm::rmvnorm(n = 1, mean = X_train%*%beta_mle, sigma = Sigma_train)
  y_sampled_calibration = mvtnorm::rmvnorm(n = 1, mean = X_calibration%*%beta_mle, sigma = Sigma_calibration)
  y_sampled_test = mvtnorm::rmvnorm(n = 1, mean = X_test%*%beta_mle, sigma = Sigma_test)
  
  train_df$SalePrice =  t(y_sampled_train)
  calibration_df$SalePrice = t(y_sampled_calibration)
  test_df$SalePrice = t(y_sampled_test)
  
  
  return(list(full_df = rbind(train_df, calibration_df, test_df), 
              train_df = train_df, 
              calibration_df = calibration_df, 
              test_df = test_df, 
              mle = c(rho_hat, sigma_eps_hat, sigma_hat)))
}

GetBetaAndSigma = function(X, Y, L, rho_mle, sigma_eps_mle){
  N = NROW(X)
  
  X = as.matrix(X)
  Y = as.matrix(Y)
  L = as.matrix(L)
  
  H = as.matrix(dist(L, method = "manhattan"))/1000 # distance in kilometers
  Sigma_star_hat <- exp(-H*rho_mle) + sigma_eps_mle*diag(N)
  Sigma_star_hat_inv <- solve(Sigma_star_hat + 0.001*diag(N))
  beta_hat_spatial <- solve(t(X)%*%Sigma_star_hat_inv%*%X)%*%t(X)%*%Sigma_star_hat_inv%*%Y
  
  r = Y - X%*%beta_hat_spatial
  sigma_hat_2 <- as.numeric((1/N) * t(r)%*%Sigma_star_hat_inv%*%r)
  
  Sigma_hat = sigma_hat_2*Sigma_star_hat + (sigma_eps_mle*sigma_hat_2)*diag(N)
  
  return(list(beta = beta_hat_spatial, sigma = sigma_hat_2, Sigma = Sigma_hat))
}


SplitCP_Wrapper_New = function(dfs, 
                               formulas, 
                               alpha_level = 0.9,
                               prediction_models = c("xgb"),  
                               sigma_methods = "one", 
                               weight_methods = "none", 
                               my_seed = 999){
  ### Part 0: Untangle
  
  set.seed(my_seed)
  dfs = AddCityDistrictInfo(dfs)
  
  train_df = dfs$train
  calibration_df = dfs$calibration
  test_df = dfs$test
  full_df = rbind(train_df, calibration_df, test_df)
  
  N_train = NROW(train_df)
  N_calib = NROW(calibration_df)
  N_test = NROW(test_df)
  
  model_formula = formulas$pred
  sigma_hat_formula = formulas$sigma_hat
  
  ### Populate some columns 
  train_df$alpha_lo = NA
  train_df$alpha_hi = NA
  calibration_df$alpha_lo = NA
  calibration_df$alpha_hi = NA
  test_df$alpha_lo = NA
  test_df$alpha_hi = NA
  
  train_df$model_sd = NA
  calibration_df$model_sd = NA
  test_df$model_sd = NA
  
  
  train_df$SalePriceOriginal = train_df$SalePrice
  calibration_df$SalePriceOriginal = calibration_df$SalePrice
  test_df$SalePriceOriginal = test_df$SalePrice
  
  ### Prepare
  N_prediction_models = length(prediction_models)
  N_score_functions = length(sigma_methods)
  N_weight_methods = length(weight_methods)
  
  train_df_all_models = data.frame()
  calibration_df_all_models = data.frame()
  test_df_all_models = data.frame()
  
  
  for(i in 1:N_prediction_models){
    this_model = TrainModels(train_df = train_df, 
                             calibration_df = calibration_df, 
                             test_df = test_df, 
                             model_formula = model_formula, 
                             prediction_model = prediction_models[i])
    
    train_df_all_models = rbind(train_df_all_models, this_model$train_df)
    calibration_df_all_models = rbind(calibration_df_all_models, this_model$calibration_df)
    test_df_all_models = rbind(test_df_all_models, this_model$test_df)
    
  }
  
  
  ### Step 2: Get all the non-conformity scores
  train_df_all_sigmas = data.frame()
  calibration_df_all_sigmas = data.frame()
  test_df_all_sigmas = data.frame()
  
  for(i in 1:N_score_functions){
    
    ### Add sigma
    cat("\t Non-conformity score: ", sigma_methods[i], "\n")
    
    dfs_with_sigma = GetSigma(train_df = train_df_all_models, 
                              calibration_df = calibration_df_all_models, 
                              test_df = test_df_all_models, 
                              sigma_method = sigma_methods[i], 
                              sigma_hat_formula = sigma_hat_formula, 
                              model_formula = model_formula, 
                              alpha = alpha_level)
    
    train_df_all_sigmas = rbind(train_df_all_sigmas, dfs_with_sigma$train_df)
    calibration_df_all_sigmas = rbind(calibration_df_all_sigmas, dfs_with_sigma$calibration_df)
    test_df_all_sigmas = rbind(test_df_all_sigmas, dfs_with_sigma$test_df)
  }
  
  
  ### Step 3: Get all the weightings
  all_runs_train = data.frame()
  all_runs_calibration = data.frame()
  all_runs_test = data.frame()
  
  
  for(j in 1:N_weight_methods){
    
    cat("Weight method: ", weight_methods[j], "\n")
    
    if(weight_methods[j] == "proximity"){
      cat("Training proximity matrix ... \n")
      ranger_formula = model_formula
      # ranger_formula = as.formula(SalePrice ~ lng + lat)
      ranger_mtry = length(all.vars(ranger_formula))-1
      ranger_model = ranger::ranger(formula = ranger_formula, 
                                    data = train_df,
                                    max.depth = 8,
                                    mtry = ranger_mtry,
                                    num.trees = 200)
      
      calibration_leafs = predict(object = ranger_model, calibration_df, type = "terminalNodes")$prediction
      
    } else{
      ranger_model = NA
      calibration_leafs = NA
    }
    
    start_time = Sys.time()
    one_run = SplitCP3_MultipleSigmas(dfs = list(train = train_df_all_sigmas, 
                                                 calibration = calibration_df_all_sigmas, 
                                                 test = test_df_all_sigmas), 
                                      formulas = formulas, 
                                      alpha_level = alpha_level, 
                                      prediction_model = prediction_model, 
                                      weight_method = weight_methods[j], 
                                      calibration_leafs = calibration_leafs, 
                                      ranger_model = ranger_model)
    end_time = Sys.time()
    
    all_runs_train = rbind(all_runs_train, one_run$train)
    all_runs_calibration = rbind(all_runs_calibration, one_run$calibration)
    all_runs_test = rbind(all_runs_test, one_run$test)
    
  }
  
  ### Check for Spatial Oracle method
  if("spatial" %in% prediction_models){
    
    # Make a copy and change sigma_method to "Oracle"
    train_df_copy = all_runs_train %>% 
      filter(prediction_model == "spatial", 
             score_function == "one", 
             weight_method == "none") %>%
      mutate(score_function = "Oracle")
    
    calibration_df_copy = all_runs_calibration %>% 
      filter(prediction_model == "spatial", 
             score_function == "one", 
             weight_method == "none") %>% 
      mutate(score_function = "Oracle")
    
    test_df_copy = all_runs_test %>% 
      filter(prediction_model == "spatial", 
             score_function == "one", 
             weight_method == "none") %>% 
      mutate(score_function = "Oracle")
    
    
    # Add copy to 
    all_runs_train = rbind(all_runs_train, train_df_copy)
    all_runs_calibration = rbind(all_runs_calibration, one_run$calibration)
    all_runs_test = rbind(all_runs_test, test_df_copy)
    
    
  }
  
  return(list(train = all_runs_train, 
              calibration = all_runs_calibration, 
              test = all_runs_test))
}

GetProximityOneRow = function(ranger_model, 
                              calibration_leafs, 
                              test_row){
  test_leafs = predict(object = ranger_model, test_row, type = "terminalNodes")$prediction %>% as.numeric()
  return(rowMeans(calibration_leafs == test_leafs))
}


GetSigma = function(train_df, calibration_df, test_df, sigma_method, sigma_hat_formula, model_formula, alpha){
  
  train_df = train_df %>% 
    mutate(res = SalePrice - predictions, 
           abs_res = abs(res), 
           score_function = sigma_method) %>% 
    filter(!(prediction_model != "random forest" & score_function == "cqr"))
  
  calibration_df = calibration_df %>%
    mutate(res = SalePrice - predictions, 
           abs_res = abs(res), 
           score_function = sigma_method)  %>% 
    filter(!(prediction_model != "random forest" & score_function == "cqr"))
  
  test_df = test_df %>%
    mutate(res = NA, 
           abs_res = NA, 
           score_function = sigma_method) %>% 
    filter(!(prediction_model != "random forest" & score_function == "cqr"))
  
  
  if(sigma_method == "one"){
    train_df = train_df %>% mutate(sigma_hat = 1)
    calibration_df = calibration_df %>% mutate(sigma_hat = 1)
    test_df = test_df %>% mutate(sigma_hat = 1)
    
  } else if(sigma_method == "CityDistrictMean"){
    train_df$sigma_hat = train_df$CityDistrictPriceLevel 
    calibration_df$sigma_hat = calibration_df$CityDistrictPriceLevel 
    test_df$sigma_hat = test_df$CityDistrictPriceLevel 
    
  } else if(sigma_method == "yhat"){
    train_df$sigma_hat = train_df$predictions
    calibration_df$sigma_hat =  calibration_df$predictions 
    test_df$sigma_hat = test_df$predictions 
    
  } else if (sigma_method == "lm"){
    difficulty_model = lm(formula = sigma_hat_formula, data = train_df)
    train_df$sigma_hat = predict(difficulty_model, train_df)
    calibration_df$sigma_hat = predict(difficulty_model, calibration_df)
    test_df$sigma_hat = predict(difficulty_model, test_df)
    
  } else if(sigma_method == "cqr"){
    
    ### Step 1: Train model
    train_x = train_df %>% dplyr::select(all.vars(model_formula)[-1])
    calibration_x = calibration_df %>% dplyr::select(all.vars(model_formula)[-1])
    test_x = test_df %>% dplyr::select(all.vars(model_formula)[-1])
    
    train_y = train_df %>% dplyr::pull(all.vars(model_formula)[1])
    calibration_y = calibration_df %>% dplyr::pull(all.vars(model_formula)[1])
    test_y = test_df %>% dplyr::pull(all.vars(model_formula)[1])
    
    
    library(quantregForest)
    qrf_model = quantregForest(x = train_x,
                               y = as.numeric(train_y),
                               nthreads = 1)
    
    ### Step 2: Predict
    
    train_qrf = predict(object = qrf_model, 
                        newdata = train_x,
                        what = c(0.5-alpha/2, 0.5, 0.5 + alpha/2))
    
    calibration_qrf = predict(object = qrf_model, 
                              newdata = calibration_x,
                              what = c(0.5-alpha/2, 0.5, 0.5 + alpha/2))
    
    test_qrf = predict(object = qrf_model, 
                       newdata = test_x,
                       #what = c(alpha/2, 0.5, 1 - alpha/2))
                       what = c(0.5-alpha/2, 0.5, 0.5 + alpha/2))
    
    train_df$alpha_lo = train_qrf[,1]
    train_df$predictions = train_qrf[,2]
    train_df$alpha_hi = train_qrf[,3]
    
    calibration_df$alpha_lo = calibration_qrf[,1]
    calibration_df$predictions = calibration_qrf[,2]
    calibration_df$alpha_hi = calibration_qrf[,3]
    
    test_df$alpha_lo = test_qrf[,1]
    test_df$predictions = test_qrf[,2]
    test_df$alpha_hi = test_qrf[,3] 
    
    train_df$sigma_hat = 1
    calibration_df$sigma_hat = 1
    test_df$sigma_hat = 1
    
  } else if(sigma_method == "spatial_oracle"){
    train_df$sigma_hat = 1
    calibration_df$sigma_hat = 1
    test_df$sigma_hat = 1
    
    train_df$Ri = train_df$oracle_score
    calibration_df$Ri = calibration_df$oracle_score
    test_df$Ri = test_df$oracle_score
  }
  
  
  ### Ri
  train_df = train_df %>% mutate(Ri = abs_res/sigma_hat) 
  calibration_df = calibration_df %>% mutate(Ri = abs_res/sigma_hat) 
  test_df = test_df %>% mutate(Ri = NA) 
  
  ### Special case: Conformalized Quantile Regression
  if(sigma_method == "cqr"){
    
    # This is eq. (9) in Romano et al (2019)
    train_df = train_df %>%
      mutate(Ri = pmax(train_df$alpha_lo - train_df$SalePrice, 
                       train_df$SalePrice - train_df$alpha_hi))
    
    calibration_df = calibration_df %>% 
      mutate(Ri = pmax(calibration_df$alpha_lo - calibration_df$SalePrice, 
                       calibration_df$SalePrice - calibration_df$alpha_hi)) 
    
    test_df = test_df %>% mutate(Ri = NA) 
    
    
  } 
  
  
  return(list(train_df = train_df, 
              calibration_df = calibration_df, 
              test_df = test_df))
}



SplitCP3_MultipleSigmas = function(dfs, 
                                   formulas,
                                   alpha_level,
                                   prediction_model,
                                   weight_method, 
                                   calibration_leafs, 
                                   ranger_model){
  
  ## Part 0: Untangle
  train_df = dfs$train
  calibration_df = dfs$calibration
  test_df = dfs$test
  
  N_train = NROW(train_df)
  N_calib = NROW(calibration_df)
  N_test = NROW(test_df)
  
  model_formula = formulas$pred
  sigma_hat_formula = formulas$sigma_hat
  
  # Do some preparation on train_df and calibration_df (can be done previously?)
  train_df = train_df %>% mutate(weight_method = weight_method)
  calibration_df = calibration_df %>% mutate(weight_method = weight_method)
  test_df = test_df %>% mutate(weight_method = weight_method, q90 = as.numeric(NA)) 
  
  
  ### Prepare for multiple rows 
  unique_test_ids = unique(test_df$TransactionID)
  N_test_unique = length(unique_test_ids)
  test_df_all = data.frame()
  
  
  # New method
  train_df = train_df %>% mutate(model_and_score = paste(prediction_model, score_function, sep = "_"))
  calibration_df = calibration_df %>% mutate(model_and_score = paste(prediction_model, score_function, sep = "_"))
  test_df = test_df %>% mutate(model_and_score = paste(prediction_model, score_function, sep = "_"))
  
  
  unique_scores = unique(test_df$model_and_score)
  N_scores = length(unique_scores)
  
  
  ### REGULAR WEIGHTS
  for(i in 1:N_test_unique){
    test_subset = test_df %>% filter(TransactionID %in% unique_test_ids[i])
    N_subset = NROW(test_subset)
    
    if(weight_method == "none"){
      break
    } 
    else if(weight_method == "spatial"){
      dist = sqrt((test_subset$Longitude[1] - calibration_df$Longitude)^2 + (test_subset$Latitude[1] - calibration_df$Latitude)^2)
      b = 10^6
      ww = exp(-dist^2/b)
      
    } else if(weight_method == "spatial_nn"){
      dist = sqrt((test_subset$Longitude[1] - calibration_df$Longitude)^2 + (test_subset$Latitude[1] - calibration_df$Latitude)^2)
      ww = ifelse(dist < 1000, 1, 0)
      if(sum(ww) == 0){
        cat("No neighbors within 1 km!\n")
        #WW = rep(1, NROW(calibration_df))
        ww = ifelse(dist < 5000, 1, 0)
      }
      
    } else if(weight_method == "proximity"){
      ww = GetProximityOneRow(ranger_model = ranger_model, 
                              calibration_leafs = calibration_leafs, 
                              test_row = test_subset[1,])
      ww = exp(ww) - 1
    } 
    
    else if(weight_method == "mondrian"){
      break 
    }
    
    
    # Normalize weights
    ww_normalized = ww/sum(ww)
    
    # This step must be done for all rows in test_subset because we use different calibration set each time!
    for(s in 1:N_scores){
      calibration_subset = calibration_df %>% filter(model_and_score == unique_scores[s])
      
      #cat("alpha_level: ", alpha_level, "\")
      test_subset$q90[s] = as.numeric(modi::weighted.quantile(x = calibration_subset$Ri, 
                                                              w = ww_normalized, 
                                                              prob = alpha_level)) 
      
    }
    test_df_all = rbind(test_df_all, test_subset)
    
    
  } # for loop
  
  
  ### SPECIAL TREATMENT 1: Regular CP
  if(weight_method == "none"){
    test_df_all = data.frame()
    
    for(s in 1:N_scores){
      calibration_subset = calibration_df %>% filter(model_and_score == unique_scores[s])
      test_subset = test_df %>% 
        filter(model_and_score == unique_scores[s]) %>% 
        mutate(q90 = as.numeric(quantile(x = calibration_subset$Ri, prob = alpha_level)))
      test_df_all = rbind(test_df_all, test_subset)
    } # for
  } # if 
  
  
  ### SPECIAL TREATMENT 2: Mondrian CP
  
  if(weight_method == "mondrian"){
    test_df_all = data.frame()
    unique_cd = unique(test_df$CityDistrict)
    N_cd = length(unique_cd)
    
    for(s in 1:N_scores){
      for(cd in 1:N_cd){
        # Subset
        calibration_subset = calibration_df %>% filter(model_and_score == unique_scores[s],
                                                       CityDistrict == unique_cd[cd])
        
        test_subset = test_df %>% filter(model_and_score == unique_scores[s], 
                                         CityDistrict == unique_cd[cd])
        
        test_subset = test_subset %>% mutate(q90 = as.numeric(quantile(x = calibration_subset$Ri,
                                                                       prob = alpha_level)))
        test_df_all = rbind(test_df_all, test_subset)
      } # for city districts
      
    } # for scores
  } # if 
  test_df_with_confidence = test_df_all %>% 
    mutate(
      Pq05 = case_when(score_function == "cqr" ~ alpha_lo - q90, 
                       score_function != "cqr" ~ predictions - sigma_hat*q90), 
      Pq95 = case_when(score_function == "cqr" ~ alpha_hi + q90, 
                       score_function != "cqr" ~ predictions + sigma_hat*q90))
  
  
  new_dfs = list(train = train_df, calibration = calibration_df, test = test_df_with_confidence)
  
  return(new_dfs)
}





AddCityDistrictInfo = function(dfs){
  
  # Unwrap
  train_df = dfs$train
  calibration_df = dfs$calibration
  test_df = dfs$test
  
  empirical_mean_origial = train_df %>% 
    group_by(CityDistrict) %>% 
    summarize(mean_price = mean(SalePrice))
  
  train_df$CityDistrictPriceLevel = train_df %>% left_join(empirical_mean_origial, by = "CityDistrict") %>% pull(mean_price)
  calibration_df$CityDistrictPriceLevel = calibration_df %>% left_join(empirical_mean_origial, by = "CityDistrict") %>% pull(mean_price)
  test_df$CityDistrictPriceLevel = test_df %>% left_join(empirical_mean_origial, by = "CityDistrict") %>% pull(mean_price)
  
  
  
  return(list(full = rbind(train_df, calibration_df, test_df), 
              train = train_df, 
              calibration = calibration_df, 
              test = test_df))
  
  
}






EvaluateOneAlpha = function(df, alpha_level){
  
  upper_var_name = "Pq95"
  lower_var_name = "Pq05"
  
  colnames(df)[which(colnames(df) == upper_var_name)] = "upper"
  colnames(df)[which(colnames(df) == lower_var_name)] = "lower"
  
  
  res = df %>% 
    mutate(
      alpha_level = alpha_level, 
      covered = (SalePrice >= lower & SalePrice <= upper), 
      covered_pvalue = (pvalue <= alpha_level),
      interval_size = (upper - lower), 
      relative_interval_size = (upper-lower)/SalePrice) %>% 
    group_by(prediction_model, 
             score_function, 
             weight_method, 
             alpha_level) %>%
    dplyr::summarize(
      Coverage = 100*mean(covered),
      CoveragePvalue = 100*mean(covered_pvalue), 
      CoverageGap = 100*mean(alpha_level) - Coverage,
      CoverageGapPvalue = 100*mean(alpha_level) - CoveragePvalue,
      M_IntSize = mean(interval_size),
      Md_IntSize = median(interval_size), 
      M_RelIntSize = mean(relative_interval_size), 
      Md_RelIntSize = median(relative_interval_size)) #%>%arrange(prediction_model, score_function, weight_method)
  
  res$Coverage = case_when((res$prediction_model == "spatial" & res$score_function == "Oracle") ~ res$CoveragePvalue, 
                           TRUE ~ res$Coverage)
  res$CoverageGap = case_when((res$prediction_model == "spatial" & res$score_function == "Oracle") ~ res$CoverageGapPvalue, 
                              TRUE ~ res$CoverageGap)
  
  res$CoveragePvalue = case_when((res$prediction_model == "spatial" & res$score_function == "Oracle") ~ res$CoveragePvalue, 
                                 TRUE ~ NA)
  res$CoverageGapPvalue = case_when((res$prediction_model == "spatial" & res$score_function == "Oracle") ~ res$CoverageGapPvalue, 
                                    TRUE ~ NA)
  
  
  return(res)
}


EvaluateOneAlphaPerCityDistrict = function(df, alpha_level){
  
  upper_var_name = "Pq95"
  lower_var_name = "Pq05"
  
  colnames(df)[which(colnames(df) == upper_var_name)] = "upper"
  colnames(df)[which(colnames(df) == lower_var_name)] = "lower"
  
  res = df %>%
    mutate(
      alpha_level = alpha_level, 
      covered = (SalePrice >= lower & SalePrice <= upper), 
      covered_pvalue = (pvalue <= alpha_level),
      interval_size = (upper - lower), 
      relative_interval_size = (upper-lower)/SalePrice) %>% 
    group_by(prediction_model, 
             score_function, 
             weight_method, 
             CityDistrict, alpha_level) %>%
    dplyr::summarize(
      Coverage = 100*mean(covered),
      CoveragePvalue = 100*mean(covered_pvalue), 
      #CoverageGap = Coverage - 100*mean(alpha_level),
      CoverageGap = 100*mean(alpha_level) - Coverage,
      CoverageGapPvalue = 100*mean(alpha_level) - CoveragePvalue,
      M_IntSize = mean(interval_size),
      Md_IntSize = median(interval_size), 
      M_RelIntSize = mean(relative_interval_size), 
      Md_RelIntSize = median(relative_interval_size)) #%>%arrange(prediction_model, score_function, weight_method)
  
  res$Coverage = case_when((res$prediction_model == "spatial" & res$score_function == "Oracle") ~ res$CoveragePvalue, 
                           TRUE ~ res$Coverage)
  res$CoverageGap = case_when((res$prediction_model == "spatial" & res$score_function == "Oracle") ~ res$CoverageGapPvalue, 
                              TRUE ~ res$CoverageGap)
  
  res$CoveragePvalue = case_when((res$prediction_model == "spatial" & res$score_function == "Oracle") ~ res$CoveragePvalue, 
                                 TRUE ~ NA)
  res$CoverageGapPvalue = case_when((res$prediction_model == "spatial" & res$score_function == "Oracle") ~ res$CoverageGapPvalue, 
                                    TRUE ~ NA)
  
  res = res %>% select(-CoveragePvalue, -CoverageGapPvalue)
  return(res)
}

EvaluateConfidenceIntervals = function(df, alphas){
  
  results_df = data.frame()
  for(a in 1:length(alphas)){
    this_res = EvaluateOneAlpha(df, alpha_level = alphas[a])
    results_df = rbind(results_df, this_res)
  }
  
  results_df = results_df  %>% 
    arrange(alpha_level, prediction_model, score_function, weight_method) 
  
  return(results_df)
}

EvaluateConfidenceIntervalsPerCityDistrict = function(df, alphas){
  
  
  results_df = data.frame()
  for(a in 1:length(alphas)){
    this_res = EvaluateOneAlphaPerCityDistrict(df, alpha_level = alphas[a])
    results_df = rbind(results_df, this_res)
  }
  
  results_df = results_df  %>% 
    arrange(alpha_level, prediction_model, score_function, weight_method) 
  return(results_df)
}




EvaluatePrediction = function(test_df){
  results_df = test_df %>%
    mutate(res = SalePrice - predictions, 
           rel_res = res/SalePrice, 
           abs_res = abs(res), 
           abs_rel_res = abs(rel_res)) %>% 
    group_by(prediction_model) %>%
    summarize(
      RMSE = sqrt(mean(res^2)),
      MdAE = median(abs_res), 
      MdAE_rel = median(abs_rel_res),
      PP10 = mean(as.numeric(abs(rel_res) < 0.1)),
      PP20 = mean(as.numeric(abs(rel_res) < 0.2)) 
    )
  
  return(results_df)
}


DataSplit = function(df_oslo, split = c(1/3, 1/3, 1/3), seed = 123){
  
  set.seed(seed)
  
  ind = sample(1:3, size = NROW(df_oslo), 
               replace = TRUE, 
               prob = split)
  
  return(list(train_df = df_oslo[which(ind == 1),], 
              calibration_df = df_oslo[which(ind == 2),], 
              test_df = df_oslo[which(ind == 3),]))
}

