import xlrd
import xlwt


def read_excel(*path):
    if path:
        path = path[0]
        print("read_excel path={}".format(path))
        workbook = xlrd.open_workbook(path)
    else:
        workbook = xlrd.open_workbook(r'C:\Users\wangzh\Desktop\question.xlsx')
    # print("sheet name={}".format(workbook.sheet_names()))
    sheet = workbook.sheet_by_name('Sheet1')
    nrows = sheet.nrows
    ncols = sheet.ncols
    print("nrows={}, ncols={}".format(nrows, ncols))
    array = [[0] * ncols for _ in range(nrows-1)]
    question_type = ""
    answer = ""
    for i in range(1, nrows):
        # print(sheet.row_values(i, 0, 3))
        row_values = sheet.row_values(i, 0, 3)
        question_type_temp = row_values[0]
        if question_type_temp:
            question_type = question_type_temp
        question = row_values[1]
        answer_temp = row_values[2]
        answer = answer_temp if answer_temp is not '' else answer
        # if answer_temp:
            # answer = answer_temp
        # print('question_type={}, question={}, answer={}'.format(question_type, question, answer))
        array[i-1] = [question_type, question, answer]
    return array


if __name__ == '__main__':
    result = read_excel(r'F:\pyworkspace\text2vec\web\question.xlsx')
    print(result)
