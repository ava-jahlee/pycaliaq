# pycaliaq
This suite of tools generate a classroom IAQ baseline, which includes standardized diagnostic scenarios based on common retrofitting practices and measurement protocols of classroom IAQs; the diagnostic scenarios intend to quantify the dilution and filtration capabilities of classrooms through deposition, infiltration, and natural/mechanical ventilations when a high concentration is observed; the first principle model is developed to normalize the measurement, which is fitted against the measurement by adjusting its parameter values. In order to save time and effort for practitioners, automated and semi-automated calibrations that run in a short time are also developed.

▶ Project Structure: \n
 / _data: Example data file (example_data1000.csv) \n
 / _documentation: Documentation directory \n
 / auto-cal: Automated calibration modules and scripts \n
   *cls_func.py: Module with functions and classes for automated calibration using Kriging and L-BFGS-B algorithm. \n
   *auto_example.py: Script demonstrating automated calibration using example data (pycaliaq/_data/example_data1000.csv). \n
 / semi-auto-cal: Semi-automated calibration modules and scripts \n
   *cls_func.py: Module with functions and classes for semi-automated calibration with Conditional Inference Tree model. \n
   *auto_example.py: Script demonstrating semi-automated calibration using example data (pycaliaq/_data/example_data1000.csv). \n

▶ This open-source project is developed based on the automated and semi-automated calibration methods proposed in the following research papers:
1. Sung HJ, Kim SH, Choi SY. Preparation of an Indoor Air Quality Baseline Model for School Retrofitting Using Automated and Semi-Automated Calibrations: The Case Study in South Korea. Buildings. 2022; 12(9):1449.
2. Sung HJ, Kim SH, Kim H. Analysis of Building Retrofit, Ventilation, and Filtration Measures for Indoor Air Quality in a Real School Context: A Case Study in Korea. Buildings. 2023; 13(4):1033.

▶ The research project - Development of a Numerical Analysis-Based Comprehensive Diagnosis System for Air Environment in Existing School Buildings by Type (supported by the National Research Foundation of Korea: No. 2019M3E7A1113091) - was conducted by the Built Environment and Building Service Engineering (BEBSE) Lab of Seoul National University of Science and Technology.

▶ Credits: \n
The code for pycaliaq was based on code originally written by Jung Ah, Lee (@BEBSE of Seoul National University of Science and Technology). \n
For any inquiries, please contact jahlee.ava@gmail.com. \n
