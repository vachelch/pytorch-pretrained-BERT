import csv, os, random


class Dataset(object):
    @classmethod
    def generator(cls, data, batch_size, shuffle = True):
        if shuffle: random.shuffle(data)

        for i in range(0, len(data), batch_size):
            yield list(zip(*data[i: i+batch_size]))

    def read_data(cls, path, skip_header = True, delimiter = "\t"):
        with open(path, 'r', encoding = 'utf8') as f:
            reader = csv.reader(f, delimiter = delimiter)
            if skip_header: next(reader)

            data = []
            for row in reader:
                data.append(row)

        return data

# dir, name, batch_size, head, label_list
class Stance_Dataset(Dataset):
    def __init__(self, args, train_name = None, dev_name = None, test_name = None):
        self.args = args
        self.train_name = train_name
        self.dev_name = dev_name
        self.test_name = test_name
        if train_name:
            path = os.path.join(args.stance_dir, train_name)
            self.train_data = self.read_data(path)
        if dev_name:
            path = os.path.join(args.stance_dir, dev_name)
            self.dev_data = self.read_data(path)
        if test_name:
            path = os.path.join(args.stance_dir, test_name)
            self.test_data = self.read_data(path)
        

    def Iterator(self):
        iters = []
        if self.train_name:
            train_iter = self.generator(self.train_data, self.args.train_batch_size)
            iters.append(train_iter)
        if self.dev_name:
            dev_iter = self.generator(self.dev_data, self.args.eval_batch_size)
            iters.append(dev_iter)
        if self.test_name:
            test_iter = self.generator(self.test_data, self.args.eval_batch_size, shuffle = False)
            iters.append(test_iter)

        return iters


class IR_Dataset(Dataset):
    def __init__(self, args, train_name = None, dev_name = None, test_name = None):
        self.args = args
        self.train_name = train_name
        self.dev_name = dev_name
        self.test_name = test_name

        if train_name:
            path = os.path.join(args.ir_dir, train_name)
            self.train_data = self.read_data(path)
        if dev_name:
            path = os.path.join(args.ir_dir, dev_name)
            self.dev_data = self.read_data(path)
        if test_name:
            path = os.path.join(args.ir_dir, test_name)
            self.test_data = self.read_data(path)
        

    def Iterator(self):
        iters = []
        if self.train_name:
            train_iter = self.generator(self.train_data, self.args.train_batch_size)
            iters.append(train_iter)
        if self.dev_name:
            dev_iter = self.generator(self.dev_data, self.args.eval_batch_size)
            iters.append(dev_iter)
        if self.test_name:
            test_iter = self.generator(self.test_data, self.args.eval_batch_size, shuffle = False)
            iters.append(test_iter)

        return iters



