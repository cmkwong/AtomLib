import inspect
from myUtils import fileModel
from dataclasses import dataclass


def paramPreprocess(input_data, param):
    if param.annotation == list:
        required_input_data = input_data.split(' ')
        if len(input_data) == 0: required_input_data = []
    elif type(input_data) != param.annotation:
        required_input_data = param.annotation(input_data)  # __new__, refer: https://www.pythontutorial.net/python-oop/python-__new__/
    else:
        required_input_data = input_data
    return required_input_data


def read_default_param(strategy_name, main_path, paramFile):
    text = fileModel.read_text(main_path, paramFile)
    strategy_param_text = [t for t in text.split('~') if len(t) > 0 and t.find(strategy_name) >= 0][0].strip()
    strategy_param_dict = {}
    for param_text in strategy_param_text.split('\n')[1:]:
        param_name, value = param_text.split(':')
        strategy_param_dict[param_name.strip()] = value.strip()
    return strategy_param_dict


def input_param(param, strategy_param_dict):
    input_data = input("{}({})\nDefault: {}: ".format(param.name, param.annotation.__name__, strategy_param_dict[param.name]))
    if len(input_data) == 0:
        input_data = strategy_param_dict[param.name]
    return input_data


def ask_params(class_object, main_path: str, paramFile: str):
    # read the default params text
    strategy_param_dict = read_default_param(class_object.__name__, main_path, paramFile)

    # params details from object
    sig = inspect.signature(class_object)
    params = {}
    for param in sig.parameters.values():
        if (param.kind == param.KEYWORD_ONLY) and (param.default == param.empty):
            input_data = input_param(param, strategy_param_dict)  # asking params
            input_data = paramPreprocess(input_data, param)
            params[param.name] = input_data
    return params


def insert_params(class_object, input_datas: list):
    """
    inputParams and class_object must have same order and length
    :param class_object: class
    :param inputParams: list of input parameters
    :return: {params}
    """
    # params details from object
    sig = inspect.signature(class_object)
    params = {}
    for i, param in enumerate(sig.parameters.values()):
        if (param.kind == param.KEYWORD_ONLY):  # and (param.default == param.empty)
            input_data = paramPreprocess(input_datas[i], param)
            params[param.name] = input_data
    return params
