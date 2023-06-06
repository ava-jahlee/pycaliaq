 pycaliaq
  : An IAQ simulation baseline model calibration toolkit for practitioners : automated and semi-automated calibration
--------------------------------------------------------------------------------------------------------------------------------------------------
▶ pycaliaq is a python open-source toolkit designed to assist practitioners who conduct Indoor Air Quality (IAQ) simulation such as CONTAM, by helping determine the input values of environmental variables (control variables) for the simulation baseline model. It offers two calibration methods: automated calibration and semi-automated calibration based on practitioners' real-measured data. 

▶The project's structure:
  caliaq
   |- _data
    |- example_data1000.csv *

   |- _documentation

   |- auto-cal
    |- example
     |- auto_example.py *
     |- result.csv *
    |- cls.func.py
    |- Kriging_LBFGSB.py

   |- semi-auto-cal
    |- example
     |- result.csv *
     |- rmodel_out.txt *
     |- semiauto_example.py *
    |- cls_func.py
    |- ConditionalInferenceTree.py

 ▶  This repository contains code for:
 1. Automatic Calibration:
  The automatic calibration method utilizes the Kriging regression combined with the L-BFGS-B optimization algorithm. 
  1) cls_func.py: This module contains the implementation of various functions and classes required for automated calibration with Kriging and L-BFGS-B algorithm.
  2) auto_example.py: This script demonstrates the usage of the automated calibration method with example data (caliaq/_data/example_data1000.csv). It showcases how to prepare the training data, build a Kriging surrogate model, and perform the L-BFGS-B optimization to find the optimal combination of explanatory variable values.
   ※ Usage:
  'Kriging_LBFGSB.py' is an empty module for users. 
  Import the cls_func module.
  Prepare the training data using the prepare_traindata function.
  Build the Kriging surrogate model with the desired parameters using the Kriging_rbf class.
  Perform the L-BFGS-B optimization using the L_BFGS_B class with the input data and the surrogate model.
  Get the optimized combination of explanatory variable values.

 2. Semi-Automatic Calibration:
  The semi-automatic calibration method involves utilizing the R package 'partykit' and R scripting within Python to extract and parse Conditional Inference Tree (CIT) models. 
  1) cls_func.py: This module contains the implementation of various functions and classes required for semi-automated calibration with Conditional Inference Tree model.
  2) auto_example.py: This script demonstrates the usage of the semi-automated calibration method with example data (caliaq/_data/example_data1000.csv).
   ※ Usage:
  'ConditionalInferenceTree.py' is an empty module for users. 
  Import the necessary functions and classes.
  Install the required R packages using rpy2_install_rpackages if they are not already installed.
  Read the CSV file using read_txt_file and obtain the raw data.
  Create an instance of ScriptingRModel with the filename and path.
  Use the r_model_txt method to extract the R model and save it as a text file.
  Create an instance of NodeParser for each model line and access the parsed information.
 !! Please refer to the code examples('auto_example.py', 'semiauto_example.py') provided in the package for a more detailed understanding of how to use the 'caliaq' open source for automated and semi-automated calibration.

 ▶ This open-source is developed as part of:
  "Development of a Numerical Analysis-Based Comprehensive Diagnosis System for Air Environment in Existing School Buildings by Type" research project, 
  conducted at Built Environment and Building Service Engineering(BEBSE) Lab, Seoul National University of Science and Technology.

 ▶ The prior research conducted for the development of this open-source includes the following:
  Sung, H. J., Kim, S. H., & Kim, H. (2023). Analysis of Building Retrofit, Ventilation, and Filtration Measures for Indoor Air Quality in a Real School Context: A Case Study in Korea. Buildings, 13(4), 1033.
  Sung, H. J., Kim, S. H., & Choi, S. Y. (2022). Preparation of an Indoor Air Quality Baseline Model for School Retrofitting Using Automated and Semi-Automated Calibrations: The Case Study in South Korea. Buildings, 12(9), 1449.

 ▶ Please contact the email below for any inquiries you may have:
  jahlee.ava@gmail.com
