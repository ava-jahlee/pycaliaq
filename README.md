# pycaliaq
An IAQ simulation baseline model calibration toolkit
pycaliaq: IAQ Simulation Calibration Toolkit

pycaliaq is an open-source Python toolkit for IAQ simulation calibration, offering automated and semi-automated calibration methods. It helps practitioners determine input values for environmental variables in the simulation baseline model.
Automatic Calibration:
 *cls_func.py: Module with functions and classes for automated calibration using Kriging and L-BFGS-B algorithm.
 *auto_example.py: Script demonstrating automated calibration using example data (caliaq/_data/example_data1000.csv).
Semi-Automatic Calibration:
 *cls_func.py: Module with functions and classes for semi-automated calibration with Conditional Inference Tree model.
 *auto_example.py: Script demonstrating semi-automated calibration using example data (caliaq/_data/example_data1000.csv).
 
▶ Project Structure:
 _data: Example data file (example_data1000.csv)
 _documentation: Documentation directory
 auto-cal: Automated calibration modules and scripts
 semi-auto-cal: Semi-automated calibration modules and scripts

▶ This open-source project is developed based on the automated and semi-automated calibration methods proposed in the following research paper:
Sung, H. J., Kim, S. H., & Choi, S. Y. (2022). "Preparation of an Indoor Air Quality Baseline Model for School Retrofitting Using Automated and Semi-Automated Calibrations: The Case Study in South Korea." Buildings, 12(9), 1449.

▶ The research project, titled "Development of a Numerical Analysis-Based Comprehensive Diagnosis System for Air Environment in Existing School Buildings by Type," was conducted at the Built Environment and Building Service Engineering (BEBSE) Lab, Seoul National University of Science and Technology.

▶ The developer of this open-source project:
 Jung Ah, Lee (BEBSE, Seoul National University of Science and Technology)
 For any inquiries, please contact jahlee.ava@gmail.com.
