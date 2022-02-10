from utils import listModel

def changeCase(dic, case='l'):
    old_keys = list(dic.keys())
    new_keys = listModel.changeCase(old_keys, case)
    for i, o_key in enumerate(old_keys):
        dic[new_keys[i]] = dic.pop(o_key)
    return dic