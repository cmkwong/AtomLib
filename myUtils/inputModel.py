from datetime import datetime

from myUtils import timeModel


def enter(placeholder='Input: '):
    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
    user_input = input(placeholder)  # waiting user input
    return user_input


def askNum(placeholder="Please enter a number: ", outType=int):
    """
    change the type for user input, float, int etc
    """
    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
    usr_input = input(placeholder)
    if not usr_input.isnumeric():
        print("Wrong input. \nPlease input again.\n")
        return None
    usr_input = outType(usr_input)
    return usr_input


def askDate(placeholder='Please input the date ', dateFormat="YYYY-MM-DD HH:mm:ss"):
    """
    Ask for the date: (2022, 10, 30, 22:21)
    return: tuple (2022, 1, 20, 5, 45)
    """
    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
    usr_input = input(f"{placeholder} ({dateFormat})\nInput: ")
    # if user input empty, set into default date
    if not usr_input:
        now = datetime.now()
        return (now.year, now.month, now.day, now.hour, now.minute)
    return timeModel.getTimeT(usr_input, dateFormat)


def askConfirm(question=''):
    if question: print(question)
    placeholder = 'Input [y]es to confirm OR others to cancel: '
    confirm_input = input(placeholder)
    if confirm_input == 'y' or confirm_input == "yes":
        return True
    else:
        return False
