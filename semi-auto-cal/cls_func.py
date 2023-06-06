import sys
import sympy as sp
import pandas as pd
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects import pandas2ri
import rpy2.robjects as rob

def _agg_select(cet):
    cet_inlist = cet.values.tolist()
    if len(cet_inlist) == cet_inlist.count(cet_inlist[0]):
        return cet_inlist[0]

def rpy2_install_rpackges(package: str):
    # 설치가 안될 경우, 오류 해결하는 방법도 함께 올릴것.
    # r 라이브러리 경로에 접근해서, 폴더 속성 --> 사용자의 관리권한을 모두 허용으로 변경

    utils = importr("utils")
    if not isinstalled(package):
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(package)

def read_txt_file(filename: str) -> list:
    f = open(filename, 'r')
    return f.readlines()

class ScriptingRModel:
    def __init__(self, filename: str ='data', path:str ='/'):
        userdata_url = path + filename
        if '.csv' not in userdata_url:
            userdata_url = userdata_url + '.csv'

        utDF = self._usercsv_to_DataFrame(userdata_url = userdata_url)
        self.path_and_file = utDF['file']
        self.raw_data = utDF['data']

        self.rmodel = self._r_scripting_ctree()
        self.r_model_txt(r_var_name= self.rmodel)
        self.r_model_txt_name = self.rmodel + "_out.txt"

    def r_model_txt(self, r_var_name: str):
        """extracting txt file of R' variable by R scripting in python"""


        f = open(r_var_name + "_out.txt", "w")
        sys.stdout = f
        rob.r('print(' + r_var_name + ')')
        sys.stdout = sys.__stdout__
        f.close()

    def _usercsv_to_DataFrame(self, userdata_url: str) -> dict:

        userdata_df = pd.read_csv(userdata_url)
        userdata_cdf = userdata_df[:]
        userdata_column = userdata_cdf.columns.values.tolist()

        if 'Unnamed: 0' in userdata_column and 'observations' not in userdata_column:
            userdata_cdf.rename(columns={'Unnamed: 0': 'observations'}, inplace=True)

        return {'file': userdata_url, 'data': userdata_cdf}

    def _r_scripting_ctree(self, response: str = None, explanat: list = None, formula: str = 'cvrmse ~.', rmodelname: str = 'rmodel'):
        column = self.raw_data.columns.values.tolist()
        # check the formula
        if response is None and explanat is None:

            if '~' in formula:
                self.model_formula = formula

        elif response != None and explanat == None:
            if response not in column:
                ValueError("Check response variable. Not in the raw data column.")
            explanat = column.remove(response)
            explanat_formula = "+".join(explanat)
            self.model_formula = response + '~' + explanat_formula

        elif response != None and explanat != None:
            if response not in column:
                ValueError("Check response variable. Not in the raw data column.")
            if len(set(explanat) & set(column)) == 0:
                ValueError("Check explanatory variables. Not in the raw data column.")

            explanat_formula = "+".join(explanat)
            self.model_formula = response + '~' + explanat_formula

        # rscripting with rpy2
        pandas2ri.activate()
        rob.r('rurl <- "'+str(self.path_and_file)+'"')
        if 'observations' in column:
            column.remove('observations')
        rdataframe = pandas2ri.py2rpy(self.raw_data[column])

        rob.r.assign('rdata', rdataframe)
        rob.r(rmodelname + '<- partykit::ctree(formula = ' + self.model_formula + ',data= rdata)')

        return rmodelname

class NodeParser:
    """
A class for parsing "search node" of Conditional Inference Tree model written in text, and organizing into hashtable. \n
This class would be automatically use in class 'ModelParser' in DEFAULT. \ns
You could read the explanation of Conditional Inference Tree(CIT) HERE > (https://cran.r-project.org/web/packages/partykit/vignettes/ctree.pdf,
ref: Hothorn, T., & Zeileis, A. (2015). partykit: A modular toolkit for recursive partytioning in R. The Journal of Machine Learning Research, 16(1), 3905-3909.). \n

Below is the part of an example CIT model in text: \n
--------------------------------------------------------------------------------        \n
1   Model formula: \n
2   ctree_cvrmse ~ door_in + door_out + window_in + window_out \n
3 \n
4   Fitted party: \n
5   [1] root \n
6   |   [2] door_out <= 0.00102 \n
7   |   |   [3] door_in <= 0.00324 \n
--------------------------------------------------------------------------------        \n
--> One text line of CIT model contains information of: \n
'|' : (a node depth),
'[2]' : a node number(could be a node name),
'door_out' : (spliting criteria: explanatory variable name),
<= : (spliting criteria: inequality sign),
0.00102 : (spliting criteria: value) \n
--------------------------------------------------------------------------------        \n
/* - usage - */ \n
1 explanat_vars = ['door_in', 'door_out', 'window_in', 'window_out'] \n
2 node = NodeParser('|   [2] door_out <= 0.00102', exp_var = explanat_vars) \n
3 hashed_node = node.hashed() \n
4 \n
5 print(node.model_line)          # i.e. print(node.model_line), print(node.variable), print(node.split_equation), print(node.split_value) \n
6 print(hashed_node.keys())       # output: dict_keys(['level', 'variable', 'split_equation', 'split_value', 'terminal', 'terminal_result']) \n
--------------------------------------------------------------------------------        \n"""

    def __init__(self, model_line: str, explanat_var: list, depth_comparing: list = None):
        """Initializing an instance of class 'NodeParser' \n
        :param model_line: str : "a" node line of CIT "text" model
        :param explanat_var: list : explanatory variable of CIT model
        :param depth_comparing: list = None : list of current node's and next node's depth(level).
            If user want to use NodeParser independently, depth_comparing argument should be considered deliberately.
            (ex) [1, 2]
        """
        # model_line text object
        self.model_line = model_line
        split_model_line = list(model_line.split(' '))
        # node_level ( = tree depth)
        self.level = model_line.count('|')
        # explanatory variables
        self.variable = list(set(split_model_line) & set(explanat_var))[0]
        # split_equation
        if '<=' in split_model_line:
            self.split_equation = '<='
        elif '>' in split_model_line:
            self.split_equation = '>'

        # for terminal node
        if len(depth_comparing) == 0:  # no element in the list if the node is terminal node(leaf node)
            # split_value
            idx_value = model_line.split(':')[0].split(' ')[-1]
            self.split_value = idx_value
            self.terminal = True
            self.terminal_result = float(model_line.split(':')[1].split('(')[0])

        elif depth_comparing[0] >= depth_comparing[1]:
            # split_value
            idx_value = model_line.split(':')[0].split(' ')[-1]
            self.split_value = idx_value
            self.terminal = True
            self.terminal_result = float(model_line.split(':')[1].split('(')[0])

        else:
            # split_value
            idx_value = split_model_line[-1]
            self.split_value = idx_value
            self.terminal = False
            self.terminal_result = str().split('(')[0]

    def hashing(self) -> dict:
        """key(s): level, variable, split_equation, split_value"""
        hashed = {'level': self.level,
                  'variable': self.variable,
                  'split_equation': self.split_equation,
                  'split_value': self.split_value,
                  'terminal': self.terminal,
                  'terminal_result': self.terminal_result}

        return hashed

class ModelParser:
    """
A class for parsing the Conditional Inference Tree model written in text. \n
This class requires the tree structure written in txt file.
    """

    def __init__(self, model_txt_file: str, location: str = None):
        """
        :param model_txt_file: str : name of model's text file.
        :param location: str : location of the model's text file. Default value is 'None' (It means the file assumed to be in the same location with this module.).
        """
        r_model_txt_list = read_txt_file(model_txt_file)
        if location != None:
            r_model_txt_list = read_txt_file(location + model_txt_file)

        model_txt_list, number_of_nodes_list = [], []
        for line in r_model_txt_list:
            stp_line = line.strip()  # to remove '\n' text in each line. (line is element of r_model_txt_list)
            model_txt_list.append(stp_line)

            if 'Number of inner nodes:' in stp_line or 'Number of terminal nodes:' in stp_line:
                stp_number_of_nodes = int(stp_line.split(':')[1].strip())
                number_of_nodes_list.append(stp_number_of_nodes)

        self.number_of_nodes = sum(number_of_nodes_list)
        self.number_of_inner_nodes = number_of_nodes_list[0]
        self.number_of_terminal_nodes = number_of_nodes_list[1]

        model_formula = model_txt_list[model_txt_list.index('Model formula:') + 1] # i.e. ctree_cvrmse ~ door_in + door_out + window_in + window_out
        self.model_formula = self._model_formula(model_formula=model_formula)[0]
        self.model_response = self._model_formula(model_formula=model_formula)[1]
        self.model_explanatory = self._model_formula(model_formula=model_formula)[2]

        idx_fitted_model = int(model_txt_list.index('Fitted party:'))
        model_extracted = [j for j in model_txt_list[idx_fitted_model + 1: len(model_txt_list)] if '[' and ']' in j]
        if len(model_extracted) != self.number_of_nodes:
            ValueError("Must check the nodes of tree selected well")
        else:
            pass
        self.model_dynamic_node = self._generate_model_node(model_extracted=model_extracted)

    def call_dynamic_node(self, call_number: int, start_number: int = 1):
        return globals()['node{}'.format(call_number + start_number)]

    def model_traversing(self, testmodel_dataframe, observation_start_number: int = 1, export: bool = True, export_file_name: str = 'testresult'):
        nodes_data_dict = {}
        model_dataframe = testmodel_dataframe[:]

        for i in range(len(model_dataframe)):
            current_obs_number = i + observation_start_number
            current_data = model_dataframe[model_dataframe['observations'] == current_obs_number]

            path = {}   # original code var: terminal_path
            data_path = []  # original code var: path

            current_node_number = 1
            start_node_number = 1
            while self.call_dynamic_node(call_number=current_node_number, start_number=start_node_number)['terminal'] == False:
                current_node = self.call_dynamic_node(call_number=current_node_number)
                ### !Warning! since txt format of tree model shows the rounded value at digits 5 (this value used in ctree print function)
                ### so, I ADD THE BELOW CODE but not sure for versatility
                current_data_split_value = round(current_data[current_node['variable']].values[0], 5)
                evaluating_value_in_node = eval(str(current_data_split_value) + str(current_node['split_equation']+str(current_node['split_value'])))

                if evaluating_value_in_node == False:
                    current_node_level = current_node['level']
                    idx_current_node = current_node_number  # current_node_idx
                    idxing_node_depth_list = self.model_depth_list[idx_current_node: ]  # poping_cv

                    idxing_node_depth_list.pop(idxing_node_depth_list.index(current_node_level))
                    idx_another_node = idxing_node_depth_list.index(current_node_level) + 1
                    current_node_number = idx_current_node + idx_another_node
                    continue

                else: # globals()['CITnode_{}'.format(node_i+1)].get('terminal') == True
                    data_path.append('node{}'.format(current_node_number+start_node_number))
                    current_node_number += 1
                    continue

            else:
                current_node = self.call_dynamic_node(call_number=current_node_number)
                current_data_split_value = round(current_data[current_node['variable']].values[0], 5)
                evaluating_value_in_node = eval(str(current_data_split_value)+str(current_node['split_equation']+str(current_node['split_value'])))
                if evaluating_value_in_node == True:
                    data_path.append('node{}'.format(current_node_number+start_node_number))
                    path['terminal'] = 'node{}'.format(current_node_number+start_node_number)
                else:
                    current_node_number += 1
                    data_path.append('node{}'.format(current_node_number+start_node_number))
                    path['terminal'] = 'node{}'.format(current_node_number+start_node_number)

            path['path'] = data_path
            nodes_data_dict[current_data['observations'].values[0]] = path

        self.nodes_data_dict = nodes_data_dict
        evaluate_data_model = self._eval_data_model(nodes_data_dict = self.nodes_data_dict)
        self.model_node_splitted_by = [[[e for e in j if k in e] for k in self.model_explanatory] for j in evaluate_data_model[2]]
        model_dataframe = model_dataframe.join(self._organizing_result()[1:])

        for variable in range(len(self.model_explanatory)):
            model_dataframe = model_dataframe.join(pd.DataFrame({'path: variable '+self.model_explanatory[variable]+'': [e[variable] for e in self._organizing_result()[0]]}))


        result_data = model_dataframe[:]
        result_column = result_data.columns.values.tolist()
        drop_column = ['Unnamed: 0', ' '] + [e for e in result_column if e in self.model_explanatory]
        agg_column = [self.model_response, 'observations']
        rest_column = [e for e in result_column if e not in drop_column + agg_column]

        result_data_dropped = result_data.drop(columns=list(set(drop_column)&set(result_column)))[:]

        pivot_ctree_range = result_data_dropped.pivot_table(index = 'leaf: node number', values = self.model_response, aggfunc = ['min', 'max'])
        pivot_observation = result_data_dropped.pivot_table(index = 'leaf: node number', values = 'observations', aggfunc = 'count')
        pivot_rest_column = result_data_dropped[rest_column].groupby(['leaf: node number']).agg(_agg_select)
        pivoted_data = pd.concat([pivot_ctree_range, pivot_rest_column, pivot_observation], axis=1).sort_values(by='leaf: '+self.model_response, ascending=True)
        pivoted_data.to_csv(path_or_buf = export_file_name+'.csv')

    def _eval_data_model(self, nodes_data_dict: dict):
        nodes_data_dict_length = len(nodes_data_dict)
        eval_terminal, eval_terminal_result, eval_inequality_equation = [], [], []
        for anode in range(nodes_data_dict_length+1)[1:]:
            # terminal node lisst
            eval_terminal.append(nodes_data_dict[anode]['terminal'])
            # terminal node data list
            local_terminal = eval(nodes_data_dict[anode]['terminal'])['terminal_result']
            eval_terminal_result.append(local_terminal)
            # to evaluate path node - split criteria
            anode_path_eval = []
            for anode_apath in range(len(nodes_data_dict[anode]['path'])):
                apath = eval(nodes_data_dict[anode]['path'][anode_apath])
                apath_variable = apath['variable']
                apath_equation = apath['split_equation']
                apath_value = apath['split_value']
                anode_path_eval.append(str(apath_variable)+str(apath_equation)+str(apath_value))

            eval_inequality_equation.append(anode_path_eval)

        self.model_node_terminal = eval_terminal
        self.model_node_response_result = eval_terminal_result

        return [eval_terminal, eval_terminal_result, eval_inequality_equation]

    def _organizing_result(self):

        reduce_inequalities = []
        for k in self.model_node_splitted_by:
            cons_constants = []
            for j in range(len(k)):
                eq = sp.reduce_inequalities(k[j], self.model_explanatory[j])
                constants = []
                for args in eq.args:
                    if str(args.lhs) == self.model_explanatory[j]:
                        constants.append(float(args.rhs))
                    else:
                        constants.append(float(args.lhs))

                if len(constants) != 0:
                    smaller = min(constants)
                    greater = max(constants)

                    if smaller == float('-inf'):
                        smaller = int(0)
                    if greater == float('inf'):
                        greater = ' '

                    constants = [smaller, greater]

                cons_constants.append(constants)
            reduce_inequalities.append(cons_constants)


        # organize ex_result in dataframe
        df_empty_col = pd.DataFrame({' ': list(' ')*1000})
        df_terminal_node_number = pd.DataFrame({'leaf: node number': self.model_node_terminal})
        df_response_result = pd.DataFrame({'leaf: '+self.model_response : self.model_node_response_result})

        return [reduce_inequalities, df_empty_col, df_terminal_node_number, df_response_result]

    def _write_dataframe_csv(self, result_dataframe, filename: str = 'ex_result.csv', location: str = None):
        """
        :param result_dataframe: pandas.DataFrame : dataframe variable name to save     # dependency : library "pandas"
        :param filename: str : name of the file
        :param location: str : the location to save the file (default = None, same location with the module)
        """
        if location != None:
            result_dataframe.to_csv(path_or_buf= str(location+str(result_dataframe))+'.csv')
        result_dataframe.to_csv(path_or_buf=str(result_dataframe)+'.csv')

    def _generate_model_node(self, model_extracted: list) -> list:
        """
        :param model_extracted:list : tree model list
        :return: if user used this function, it returns True(bool)
        """
        model_dynamic_node = []
        counting_depth = [e.count('|') for e in model_extracted]  # since the number of '|' means the depth in model
        self.model_depth_list = counting_depth

        for node in range(len(model_extracted)):
            depth_comparing = []

            if node == 0:
                globals()['node{}'.format(node + 1)] = model_extracted[node]
            if (node > 0) and (node < len(model_extracted)):
                if node < len(model_extracted) - 1:
                    depth_comparing = [counting_depth[node], counting_depth[node + 1]]

                model_line = model_extracted[node]
                globals()['node{}'.format(node + 1)] = NodeParser(model_line, self.model_explanatory, depth_comparing).hashing()
                globals()['node{}'.format(node + 1)]['node_name'] = 'node{}'.format(node + 1)

            model_dynamic_node.append(globals()['node{}'.format(node + 1)])

        return model_dynamic_node

    def _model_formula(self, model_formula: str):
        model_formula = model_formula
        model_response = model_formula.split('~')[0].strip()
        model_explanatory = [i.strip() for i in model_formula.split('~')[1].split('+')]

        return [model_formula, model_response, model_explanatory]