import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import os.path
from PIL import Image
from PIL import ImageEnhance
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from data import data_transforms
import mmcv
import cv2

"""torch.utils.data模块是子类化数据，transforms 库对数据进行预处理"""

# class MyTrainData(torch.utils.data.Dataset):
#
#     # 初始化函数，得到数据
#     def __init__(self, root, img_ids, train=True, img_scale=0.5, transform=True):
#         self.train = train
#         self.root = root
#         self.images_folder_path = os.path.join(self.root, "imgs")
#         self.masks_folder_path = os.path.join(self.root, "masks")
#         self.images_path_list = self.read_files_path(self.images_folder_path)
#         self.masks_path_list = self.read_files_path(self.masks_folder_path)
#         self.scale = img_scale
#         self.transform = transform
#         self.img_ids = img_ids
#
#     # index 是根据batchsize划分数据后得到索引，最后将data和对应的masks一起返回
#     def __getitem__(self, idx):
#         # print(idx)
#         img_id = self.img_ids[idx]
#         image = cv2.imread(os.path.join(self.images_folder_path, img_id + '.jpg'))
#         mask = cv2.imread(os.path.join(self.masks_folder_path, img_id + '_segmentation' + '.png'), cv2.IMREAD_GRAYSCALE)
#         # image = Image.open(image_path).convert('RGB')
#         # mask = Image.open(mask_path)
#         # image = self.preprocess(image, self.scale)
#         # mask = self.preprocess(mask, self.scale)
#
#         if self.transform is True:
#             image, mask = self.train_transform(image, mask)
#         if self.train is False:
#             image, mask = self.val_transform(image, mask)
#             return (image, mask, img_id)
#         return (image, mask)
#
#     # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
#     def __len__(self):
#         return len(self.img_ids)
#
#     @classmethod
#     def preprocess(cls, pil_img, scale):
#         # print(pil_img.shape)
#         # w, h,  = pil_img.shape
#         # newW, newH = int(scale * w), int(scale * h)
#         # assert newW > 0 and newH > 0, 'Scale is too small'
#         # pil_img = pil_img((256, 256))
#         # img_nd = np.array(pil_img)
#         # if len(pil_img.shape) == 2:
#         #     img_nd = np.expand_dims(pil_img, axis=2)
#         #
#         # # HWC to CHW
#         # img_trans = pil_img.transpose((2, 0, 1))
#         img_nd = pil_img.astype('float32') / 255
#         # print(img_nd.shape)#3,256,256
#
#         return img_nd
#
#     def train_transform(self, image, mask):
#         image_transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((256, 256)),
#             # transforms.CenterCrop(256),
#             # transforms.RandomHorizontalFlip(0.5),
#             # transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         mask_transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((256, 256)),
#             # transforms.RandomHorizontalFlip(0.5),
#             # transforms.CenterCrop(256),
#             transforms.ToTensor()
#         ])
#         image = image_transforms(image)
#         mask = mask_transforms(mask)
#         data_trans = data_transforms.Compose([
#             # data_transforms.CenterCrop(256),  # 结肠息肉数据集中心裁剪大小288
#             data_transforms.RandomHorizontalFlip(0.5)
#             # data_transforms.RandomVerticalFlip(0.5)
#         ])
#         sample = data_trans(image, mask)
#         image = sample['image']
#         mask = sample['mask']
#         # data_trans = data_transforms.Compose([
#         #     data_transforms.RandomHorizontalFlip(0.5),
#         # ])
#         # sample = data_trans(image, mask)
#         # image = sample['image']
#         # mask = sample['mask']
#         # print(image.shape)
#         # image = image.dtype('float32') / 255
#         # image = image.transpose(2, 0, 1)
#         # mask = mask.dtype('float32') / 255
#         # mask = mask.transpose(2, 0, 1)
#         return image, mask
#
#     def val_transform(self, image, mask):
#         image_transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((256, 256)),
#             # transforms.CenterCrop(256),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         mask_transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((256, 256)),
#             # transforms.CenterCrop(256),
#             transforms.ToTensor()
#         ])
#         image = image_transforms(image)
#         mask = mask_transforms(mask)
#         # image = image.astype('float32') / 255
#         # image = image.transpose(2, 0, 1)
#         # mask = mask.astype('float32') / 255
#         # mask = mask.transpose(2, 0, 1)
#         return image, mask
#
#     def read_files_path(self, folder_path):  # 截图1
#         files_names_list = os.listdir(folder_path)  # os.listdir()返回folder_path所指向的文件夹下所有文件的名称
#         files_paths_list = [os.path.join(folder_path, file_name) for file_name in files_names_list]  # 路径拼接
#         return files_paths_list
#
# if __name__ == '__main__':
#     images_folder_path = os.path.join("/root/autodl-tmp/lxf/Mine/polyp/cell/", "cell")
#     img_path = glob(os.path.join(images_folder_path, "images/*"))
#     msk_path = glob(os.path.join(images_folder_path, "masks/*"))

# 定义MyTrainData类，继承Dataset方法，并重写__getitem__()和__len__()方法
class MyTrainData(torch.utils.data.Dataset):

    # 初始化函数，得到数据
    def __init__(self, root, train=True, img_scale=0.5, transform=True):
        self.train = train
        self.root = root
        self.images_folder_path = os.path.join(self.root, "imgs")
        self.masks_folder_path = os.path.join(self.root, "masks")
        self.images_path_list = self.read_files_path(self.images_folder_path)
        self.masks_path_list = self.read_files_path(self.masks_folder_path)
        self.scale = img_scale
        self.transform = transform

    # index 是根据batchsize划分数据后得到索引，最后将data和对应的masks一起返回
    def __getitem__(self, index):
        self.images_path_list.sort()
        self.masks_path_list.sort()
        # print/(self.images_path_list)
        # print(self.masks_path_list)
        image_path = self.images_path_list[index]
        mask_path = self.masks_path_list[index]
        # file_client = mmcv.FileClient(backend='disk')
        # img_bytes = file_client.get(image_path)
        # image = mmcv.imfrombytes(img_bytes, flag='color', backend='tifffile')
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # print("imgpath:",image_path)
        image = Image.open(image_path).convert('RGB')
        # image = cv2.imread(image_path, 1)
        image = image.resize((256, 256))
        # ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        # # 将YCrCb图像通道分离
        # channels = cv2.split(ycrcb)
        # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        # clahe.apply(channels[0], channels[0])
        #
        # ycrcb = cv2.merge(channels)
        # image = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.array(255 * (image / 255) ** 1.2, dtype='uint8')
        # 亮度
        # enh_bri = ImageEnhance.Brightness(image)
        # brightness = 1.5
        # image = enh_bri.enhance(brightness)
        # # 色度
        # enh_col = ImageEnhance.Color(image)
        # color = 1.5
        # image = enh_col.enhance(color)
        # 对比度
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # # image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # image = Image.fromarray(image)
        # enh_con = ImageEnhance.Contrast(image)
        # contrast = 1.5
        # image = enh_con.enhance(contrast)
        # image = np.array(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        # # # 将YCrCb图像通道分离
        # channels = cv2.split(ycrcb)
        # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        # clahe.apply(channels[0], channels[0])
        #
        # ycrcb = cv2.merge(channels)
        # image = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.array(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # (b, g, r) = cv2.split(image)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # bH = clahe.apply(b)
        # gH = clahe.apply(g)
        # rH = clahe.apply(r)
        # image = cv2.merge((bH, gH, rH))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.equalizeHist(image)

        # img_nd = np.array(255 * (image / 255) ** 0.5, dtype='uint8')
        # image = Image.fromarray((img_nd * 255).astype(np.uint8))
        # (b, g, r) = cv2.split(img_nd)
        # bH = cv2.equalizeHist(b)
        # gH = cv2.equalizeHist(g)
        # rH = cv2.equalizeHist(r)
        # # 合并每一个通道
        # image = cv2.merge((bH, gH, rH))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image2 = image.copy()
        # image = self.hisEqulColor2(image2)
        mask = Image.open(mask_path).convert('L')
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.resize((256, 256))
        if self.transform is True:
            # image = np.array(255 * (image / 255) ** 0.5, dtype='uint8')
            # 亮度
            # enh_bri = ImageEnhance.Brightness(image)
            # brightness = 1.5
            # image = enh_bri.enhance(brightness)
            # # 色度
            # enh_col = ImageEnhance.Color(image)
            # color = 1.5
            # image = enh_col.enhance(color)
            # # 对比度
            # enh_con = ImageEnhance.Contrast(image)
            # contrast = 1.5
            # image = enh_con.enhance(contrast)
            image = self.preprocessa(image, self.scale)
            mask = self.preprocessa(mask, self.scale)
            image, mask = self.train_transform(image, mask)
        if self.train is False:
            # image = np.dot(image, [0.299, 0.587, 0.114])
            # image = np.array(255 * (image / 255) ** 2.5, dtype='uint8')#refuge
            # image = np.array(255 * (image / 255) ** 1, dtype='uint8')
            image = self.preprocess(image, self.scale)
            mask = self.preprocess(mask, self.scale)
            image, mask = self.val_transform(image, mask)
            # files_names_list = [v for v in os.listdir(self.images_folder_path)]
            # file_list = []
            # files_names_list = os.listdir(self.images_folder_path)
            # for file in files_names_list:
            #     file_list.append(file)
            return (image, mask, image_path)
        # print("image", image.shape)
        return (image, mask)
        # return {'image': image, 'mask': mask}
        # return (image, mask)

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.images_path_list)

    @classmethod
    def preprocess(cls, pil_img, scale):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((256, 256))
        # pil_img = pil_img.crop((0, 520, 750, 1290))
        img_nd = np.array(pil_img)
        # img_nd = np.array(255 * (pil_img / 255) ** 0.5, dtype='uint8')
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # HWC to CHW
        # img_trans = img_nd.transpose((2, 0, 1))
        # img_nd = img_trans / 255
        return img_nd

    @classmethod
    def preprocessa(cls, pil_img, scale):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((256, 256))
        # pil_img = pil_img.crop((100, 450, 550, 1090))
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        return img_nd
    # def hisEqulColor2(img):
    #     # 将RGB图像转换到YCrCb空间中
    #     ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    #     # 将YCrCb图像通道分离
    #     channels = cv2.split(ycrcb)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     clahe.apply(channels[0], channels[0])
    #
    #     cv2.merge(channels, ycrcb)
    #     cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    #     return img

    def train_transform(self, image, mask):
        image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((256, 256)),
            # transforms.CenterCrop(160),
            # transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.ToTensor(),
            # transforms.Normalize([0.34258035, 0.34258035, 0.34258035], [0.18014383, 0.18014383, 0.18014383])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        mask_transforms = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((256, 256)),
            # transforms.CenterCrop(160),
            transforms.ToTensor()
        ])
        image = image_transforms(image)
        mask = mask_transforms(mask)
        if self.transform is True:
            data_trans = data_transforms.Compose([
                data_transforms.RandomHorizontalFlip(0.5),
            ])
            sample = data_trans(image, mask)
            image = sample['image']
            mask = sample['mask']

        return image, mask

    def val_transform(self, image, mask):
        image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((256, 256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        mask_transforms = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((256, 256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        image = image_transforms(image)
        mask = mask_transforms(mask)
        return image, mask

    def read_files_path(self, folder_path):  # 截图1
        files_names_list = os.listdir(folder_path)  # os.listdir()返回folder_path所指向的文件夹下所有文件的名称
        files_paths_list = [os.path.join(folder_path, file_name) for file_name in files_names_list]  # 路径拼接
        return files_paths_list

if __name__ == '__main__':
    images_folder_path = os.path.join("/root/autodl-tmp/lxf/Mine/polyp/cell/", "cell")
    img_path = glob(os.path.join(images_folder_path, "images/*"))
    msk_path = glob(os.path.join(images_folder_path, "masks/*"))

    img_path.sort()
    msk_path.sort()
#
#     len_ids = len(img_path)
#     train_size = int((80 / 100) * len_ids)
#     valid_size = int((10 / 100) * len_ids)  ## Here 10 is the percent of images used for validation
#     test_size = int((10 / 100) * len_ids)  ## Here 10 is the percent of images used for testing
#
#     train_x, test_x = train_test_split(img_path, test_size=test_size, random_state=42)
#     train_y, test_y = train_test_split(msk_path, test_size=test_size, random_state=42)
#
#     train_x, valid_x = train_test_split(train_x, test_size=test_size, random_state=42)
#     train_y, valid_y = train_test_split(train_y, test_size=test_size, random_state=42)
#
#     file_client = mmcv.FileClient(backend='disk')
#     img_bytes = file_client.get('/root/projects/Datasets/polyp_colon/test/imgs/12.tif')
#     image = mmcv.imfrombytes(img_bytes, flag='color', backend='tifffile')
#
#     print('debugger')

# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import os
# import torch.utils.data
# import os.path
# from PIL import Image
# import cv2
# import numpy as np
# from data import data_transforms
# import mmcv
# # from albumentations import RandomRotate90,Resize
#
# """torch.utils.data模块是子类化数据，transforms 库对数据进行预处理"""
#
#
# # 定义MyTrainData类，继承Dataset方法，并重写__getitem__()和__len__()方法
# class MyTrainData(torch.utils.data.Dataset):
#
#     # 初始化函数，得到数据
#     def __init__(self, root,img_ids,img_ext,mask_ext, train=True, img_scale=0.5, transform=None):
#
#         self.train = train
#         self.root = root
#         self.images_folder_path = os.path.join(self.root, "imgs")
#         self.masks_folder_path = os.path.join(self.root, "masks")
#         self.images_path_list = self.read_files_path(self.images_folder_path)
#         self.masks_path_list = self.read_files_path(self.masks_folder_path)
#         self.scale = img_scale
#         self.transform = transform
#         self.img_ids = img_ids
#         self.img_ext = img_ext
#         self.mask_ext = mask_ext
#
#     # index 是根据batchsize划分数据后得到索引，最后将data和对应的masks一起返回
#     def __getitem__(self, index):
#         img_id = self.img_ids[index]
#         # image_path = self.images_path_list[index]
#         # mask_path = self.masks_path_list[index]
#         img = cv2.imread(os.path.join(self.images_folder_path, img_id + self.img_ext))# 512*512*3
#         # mask = cv2.imread(os.path.join(self.masks_folder_path, img_id + self.img_ext))
#         mask = []
#         mask.append(cv2.imread(os.path.join(self.masks_folder_path,img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
#         mask = np.dstack(mask)
#         # img = Image.open(os.path.join(self.images_folder_path, img_id + self.img_ext)).convert('RGB')
#         # mask = Image.open(os.path.join(self.masks_folder_path, img_id + self.img_ext))
#
#         # image = Image.open(image_path).convert('RGB')
#         # mask = Image.open(mask_path)
#         image = self.preprocess(img, self.scale)
#         mask = self.preprocess(mask, self.scale)
#
#         if self.transform is not None:
#             image, mask = self.train_transform(image, mask)
#         if self.train is False:
#             image, mask = self.val_transform(image, mask)
#
#         img = np.array(image)
#         mask = np.array(mask)
#         img = img.astype('float32') / 255
#         # img = img.transpose(1, 2, 0)
#         mask = mask.astype('float32') / 255# 3,256,256
#         # mask = mask.transpose(1, 2, 0)
#         # print(mask.shape)#1*256*256
#         # print(img.shape)#3*256*256
#
#         return img, mask, {'img_id': img_id}
#
#     # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
#     def __len__(self):
#         return len(self.img_ids)
#
#     @classmethod
#     def preprocess(cls, pil_img, scale):
#         w, h, ch = pil_img.shape
#         # pil_img = Image.fromarray(np.uint8(pil_img))
#         # newW, newH = int(scale * w), int(scale * h)
#         # assert newW > 0 and newH > 0, 'Scale is too small'
#         # pil_img = pil_img.resize((newW, newH))
#         pil_img = cv2.resize(pil_img,(int(w*scale),int(h*scale)))
#         img_nd = np.array(pil_img)
#         # if len(img_nd.shape) == 2:
#         #     img_nd = np.expand_dims(img_nd, axis=2)
#
#         return img_nd
#
#     def train_transform(self, image, mask):
#         image_transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         mask_transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.ToTensor()
#         ])
#         image = image_transforms(image)
#         mask = mask_transforms(mask)
#         if self.transform is not None:
#             data_trans = data_transforms.Compose([
#                 data_transforms.CenterCrop(256),
#                 data_transforms.RandomHorizontalFlip(0.5),
#             ])
#             sample = data_trans(image, mask)
#             image = sample['image']
#             mask = sample['mask']
#         return image, mask
#
#     def val_transform(self, image, mask):
#         image_transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.CenterCrop(256),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         mask_transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.CenterCrop(256),
#             transforms.ToTensor()
#         ])
#         image = image_transforms(image)
#         mask = mask_transforms(mask)
#         return image, mask
#
#     def read_files_path(self, folder_path):  # 截图1
#         files_names_list = os.listdir(folder_path)  # os.listdir()返回folder_path所指向的文件夹下所有文件的名称
#         files_paths_list = [os.path.join(folder_path, file_name) for file_name in files_names_list]  # 路径拼接
#         return files_paths_list
#
#
# if __name__ == '__main__':
#
#     file_client = mmcv.FileClient(backend='disk')
#     img_bytes = file_client.get('/root/projects/Datasets/polyp_colon/test/imgs/12.tif')
#     image = mmcv.imfrombytes(img_bytes, flag='color', backend='tifffile')
#
#     print('debugger')
#
