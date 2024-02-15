import os
from sklearn.model_selection import train_test_split

def read_data(filename):
    sentences = []
    sentence = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line != "":
                sentence.append(line)
            else:
                sentences.append(sentence)
                sentence = []
    return sentences

def load_data(path):
    files = os.listdir(path)
    data = []
    for file in files:
        filename = os.path.join(path, file)
        sentences = read_data(filename)
        data += sentences

    return data

def write_out(filename, data):
    with open(filename, 'w') as f:
        for d in data:
            sent = "\n".join(d)
            f.write(sent)
            f.write("\n\n")


def split_data(X, a, b, c, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Tính số lượng mẫu cho từng phần
    total_samples = len(X)
    a_count = int(a * total_samples)
    b_count = int(b * total_samples)
    print("train:", a_count)
    print("dev:", b_count)
    print("test:", len(X)-a_count-b_count)
    # Chia dữ liệu
    train = X[:a_count]
    dev = X[a_count:a_count + b_count]
    test = X[a_count + b_count:]

    # Ghi vào các thư mục tương ứng
    train_path = os.path.join(outdir, 'train.conll')
    dev_path = os.path.join(outdir, 'dev.conll')
    test_path = os.path.join(outdir, 'test.conll')

    write_out(train_path, train)
    write_out(dev_path, dev)
    write_out(test_path, test)
    print("Done")

if __name__ == '__main__':
    data_dir = "Propbank_tiengViet/PropBank-VTB-1000/VTB-SRL"
    data = load_data(data_dir)
    for d in data:
        print(d)
    # print("Split data to 3 set: train, dev, test by ratio a:b:c; a + b + c = 1")
    # a = float(input("a = "))
    # b = float(input("b = "))
    # c = float(input("c = "))
    a = 0.7
    b = 0.15
    c = 0.15
    outdir = "dataset"
    split_data(data, a, b, c, outdir)






