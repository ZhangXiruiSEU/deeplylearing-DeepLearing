from torch.utils.data import Dataset
import os


class MyDataset(Dataset):
    def __init__(self, root_directory, transforms, train_set=True):
        self.root_dir = os.path.join(root_directory, "")
        # 数据集根目录
        # join 函数把所有的字符串参数合并成路径
        self.img_root_dir = os.path.join(self.root_dir, "Images")
        # 图片目录
        self.annotation_root_dir = os.path.join(self.root_dir, "Annotations")
        # 标注目录

        if train_set:
            txt_list = os.path.join(self.root_dir, "train.txt")
        else:
            txt_list = os.path.join(self.root_dir, "val.txt")
        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotation_root_dir, line.strip() + ".xml")
                             for line in read.readlines()]
            # 调用readline()可以每次读取一行内容; 调用readlines()一次读取所有内容并按行返回list。
