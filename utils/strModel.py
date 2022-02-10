import re

def concatTxt(originalTxt, newTxt, headTail=True):
    """
    :param originalTxt:
    :param newTxt:
    :param targetCode:
    :return:
    """
    targetCodes = {}
    for fileName, code in originalTxt.copy().items():
        if fileName != newTxt:
            if headTail:
                targetCodes[fileName] = originalTxt[newTxt] + '\n' + originalTxt[fileName]
            else:
                targetCodes[fileName] = originalTxt[fileName] + '\n' + originalTxt[newTxt]
    return targetCodes

def replaceAllTxt(txt, replacedTable):
    for replacedObj, replacedValue in replacedTable.items():
        txt = txt.replace(replacedObj, replacedValue)
    return txt

def textReplaceWithPattern(txt, textReplacePattern, insertedText, count=1):
    """
    :param txt: text being process
    :param textReplacePattern: pattern being replaced
    :param insertedText: text to insert
    :return: text
    """
    replacedText = re.sub(textReplacePattern, insertedText + '\n', txt, count)
    return replacedText