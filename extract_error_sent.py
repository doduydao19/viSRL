import os

def check_full_pred(data, no_pred):
    pred_counter = 0

    for row in data:
        # print(row)
        for i in row:
            if i == 'Y':
                pred_counter += 1
    # print(data)
    # print(no_pred)
    # print(pred_counter)
    # print()

    if no_pred != pred_counter:
        return False
    else:
        return True



def ananlysis_error(sent):
    anchor = 12
    checker = []
    no_pred = 0
    for row in sent:
        # print(row)
        len_row = len(row)
        no_pred = len_row - anchor
        if no_pred > 0:
            checker.append(row[anchor:])

    if no_pred != 0:
        state = check_full_pred(checker, no_pred-2)
        return state
    else:
        return True



def read_data(path):
    data = []
    sent = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == '' and len(sent) != 0:
                data.append(sent)
                sent = []
            else:
                dataline = line.split('\t')
                sent.append(dataline)

        if len(sent) > 0:
            data.append(sent)
    return data



def load_anno_file(path):
    files = os.listdir(path)
    files = sorted(files, key=lambda x:x[0])
    # print(files)
    return files

if __name__ == '__main__':
    path = 'Propbank_tiengViet/PropBank-VTB-1000'
    files = load_anno_file(path)
    data = []
    for file in files:
        print(file)

        file_path = os.path.join(path, file)
        annot_file = load_anno_file(file_path)[0]
        annot_file = os.path.join(file_path, annot_file)
        # print(annot_file)
        content = read_data(annot_file)
        # print(len(content))

        errors_file = []
        for i in range(len(content)):
            sent_id = i + 1
            error_state = ananlysis_error(content[i])
            if error_state == False:
                errors_file.append(sent_id)
                # print(sent_id)
            # break
        print(errors_file)
        # print(len(errors_file))
        print()
        # break
