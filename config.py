from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
# 设置图片的大小
imgsz = 224
# 定义数据预处理的转换操作，包括训练集和测试集的预处理方式
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),  # 将图像缩放到224x224大小
            transforms.ToTensor(),  # 将PIL图片转换为Tensor格式
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 对图像进行归一化处理
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),  # 将图像缩放到224x224大小
            transforms.ToTensor(),  # 将PIL图片转换为Tensor格式
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 对图像进行归一化处理
        ]
    ),
}

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        """
        初始化函数，接收测试集路径和数据转换（预处理）函数
        :param test_dir: 测试集数据所在的目录
        :param transform: 数据预处理方法（例如数据增强，归一化等）
        """
        self.test_dir = test_dir  # 存储测试集的根目录
        self.transform = transform  # 存储数据预处理方法
        self.image_paths = []  # 存储所有图片的路径
        # 遍历目录中的所有文件，找到图片文件的路径并加入到image_paths中
        for img_name in os.listdir(test_dir):
            img_path = os.path.join(test_dir, img_name)
            self.image_paths.append(img_path)

    def __len__(self):
        # 返回数据集中图片的数量
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # 获取图片的路径
        # 使用PIL库打开图片并将其转换为RGB模式
        image = Image.open(img_path).convert("RGB")
        # 如果有指定的转换方法，则应用于图像
        if self.transform:
            image = self.transform(image)
        # 获取图片的文件名（不带扩展名）作为图像的ID
        id = os.path.splitext(os.path.basename(img_path))[0]
        # 返回处理后的图片和对应的ID
        return image, id

# 自定义训练集数据集类
class AIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 数据集的根目录
        self.transform = transform  # 存储数据预处理方法
        # 标签字典，'ai' 类为1，'real' 类为0
        self.labels = {"ai": 1, "real": 0}
        self.image_paths = []  # 存储图片路径
        self.labels_list = []  # 存储图片对应的标签
        for split in ["train"]:
            for label in ["ai", "real"]:  # 遍历两种标签
                # 构建每个类别文件夹的路径
                folder_path = os.path.join(root_dir, split, label)
                # 遍历每个文件夹中的图片文件
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    # 将图片路径和标签添加到对应的列表中
                    self.image_paths.append(img_path)
                    self.labels_list.append(self.labels[label])

    def __len__(self):
        # 返回数据集中的图片数量
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # 获取图片路径
        # 使用PIL库打开图片并将其转换为RGB模式
        image = Image.open(img_path).convert("RGB")
        # 如果有指定的转换方法，则应用于图像
        if self.transform:
            image = self.transform(image)
        # 获取图片对应的标签
        label = self.labels_list[idx]
        # 返回处理后的图片和对应的标签
        return image, label
