import cls_func as cf
from cls_func import ModelParser

# install r package "partykit" for ctree model
cf.rpy2_install_rpackges("partykit")

# build 'ctree model' with rscripting and export ctree model to text file format
r_model = cf.ScriptingRModel(filename='example_data1000', path='/caliaq/_data/')
r_model_txt = r_model.r_model_txt_name

# parse the r-model and organize in hashtable, based on r_model_txt data
ctree_model = ModelParser(model_txt_file = r_model_txt)

# traversing and evaluating the test data with the hashtable model
r_model_data = r_model.raw_data
ctree_model.model_traversing(testmodel_dataframe = r_model_data, export = True, export_file_name = 'result')