# How to handle error when installing rpackages with function, "rpy2_install_rpackages"

▶ To install the R package "partykit" for implementing the conditional inference tree, a function called "rpy2_install_rpackages" has been defined in the "cls_func.py" module in the "semi-auto-cal" directory. <br>

▶ When executing this function, there may be errors due to permission issues in the library installation folder. If the installation fails, you can try the following steps to resolve the error: <br>
 1. Identify the location of the R library folder based on the error message. <br>
 2. Open the properties window of the R library folder. <br>
 3. Change the permissions to allow full access for the user. <br>