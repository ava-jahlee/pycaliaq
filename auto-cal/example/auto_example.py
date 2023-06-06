import cls_func as cf

# prepare train (and test) data
userdata = cf.prepare_traindata(filename='example_data1000', location='/caliaq/_data/', model_formula='cvrmse ~.', train_data_sampling=False)
train_data_set = userdata['train_data']

# build Kriging surrogate model with train data (only for rbf kernel - not flexible)
krig = cf.Kriging_rbf(train_data_set = train_data_set, test_data_set = train_data_set, length_scale=0.0005)
krig_surrogate_model = krig.fitted_gpr

# L-BFGS-B optimizer, searching for "miniumum value" of response variable value, and find optimal combination of explanatory variable value for "the very value"
optimizer = cf.L_BFGS_B(input_data = train_data_set, surrogate_model=krig_surrogate_model)
optimized_explanat = optimizer.optimize_result['optimal_explanat']
optimized_response = optimizer.optimize_result['optimal_response']

print(userdata['model_explanat'], userdata['model_response'])
print(optimized_explanat, optimized_response)