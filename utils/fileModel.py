import pandas as pd
import os
from utils import listModel

def transfer_all_xlsx_to_csv(main_path):
    """
    note 84d
    :param main_path: str, the xlsx files directory
    :return:
    """
    files = getFileList(main_path, reverse=False)
    for file in files:
        # read excel file
        excel_full_path = os.path.join(main_path, file)
        print("Reading the {}".format(file))
        df = pd.read_excel(excel_full_path, header=None)

        # csv file name
        csv_file = file.split('.')[0] + '.csv'
        csv_full_path = os.path.join(main_path, csv_file)
        print("Writing the {}".format(csv_file))
        df.to_csv(csv_full_path, encoding='utf-8', index=False, header=False)
    return True

def getFileList(pathDir, reverse=False):
    required_fileNames = []
    listFiles = os.listdir(pathDir)
    for fileName in listFiles:
        if fileName[0] != '~': # discard the temp file
            required_fileNames.append(fileName)
    required_fileNames = sorted(required_fileNames, reverse=reverse)
    return required_fileNames

def clearFiles(pathDir, pattern=None):
    """
    pattern None means clear all files in the pathDir
    """
    files = getFileList(pathDir)
    if pattern:
        files = listModel.filterList(files, pattern)
    for file in files:
        os.remove(os.path.join(pathDir, file))
        print("The file {} has been removed.".format(file))

def createDir(main_path, dir_name, readme=None):
    """
    Create directory with readme.txt
    """
    path = os.path.join(main_path, dir_name)
    if not os.path.isfile(path):
        os.mkdir(path)
    if readme:
        with open(os.path.join(path, 'readme.txt'), 'a') as f:
            f.write(readme)

def read_text(main_path, file_name):
    with open(os.path.join(main_path, file_name), 'r') as f:
        txt = f.read()
    return txt

def readAllTxtFiles(fileDir):
    """
    :param fileDir: str
    :return: {}
    """
    texts = {}
    listFiles = os.listdir(fileDir)
    for fileName in listFiles:
        with open(os.path.join(fileDir, fileName), 'r') as f:
            texts[fileName] = f.read()
    return texts

def writeAllTxtFiles(main_path, texts):
    """
    :param texts: dic
    :param path: str
    :return:
    """
    for fileName, code in texts.items():
        if fileName[0] != '_':
            with open(os.path.join(main_path, fileName), 'w') as f:
                f.write(code)
            print("Written {}".format(fileName))