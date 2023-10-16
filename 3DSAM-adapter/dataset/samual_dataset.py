import joblib
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.ndimage import rotate


class MultiLabelDataset(Dataset):
    def __init__(self, data, labels, texts, transform=None):
        self.data = data
        self.labels = labels
        self.texts = texts  
        self.transform = transform

    def __len__(self):
        return len(self.data)   
    
    def __getitem__(self, idx):
        input_data = self.data[idx]
        label = self.labels[idx]
        text = self.texts[idx]
        if self.transform:
            input_data, label = self.transform(input_data, label)
        return {'image': input_data, 'label': label, 'text': text}  

class RandomFlipAndRotate3D(object):
    def __call__(self, img, mask):
        img_volume = img
        mask_volume = mask
        # 随机确定是否进行翻转
        flip_x = np.random.choice([True, False])
        flip_y = np.random.choice([True, False])
        flip_z = np.random.choice([True, False])

        # 使用np.flip函数进行翻转
        if flip_x:
            img_volume = np.flip(img_volume, axis=1).copy()
            mask_volume = np.flip(mask_volume, axis=1).copy()
        if flip_y:
            img_volume = np.flip(img_volume, axis=2).copy()
            mask_volume = np.flip(mask_volume, axis=2).copy()
        if flip_z:
            img_volume = np.flip(img_volume, axis=3).copy()
            mask_volume = np.flip(mask_volume, axis=3).copy()

        # 随机确定是否进行旋转
        if np.random.rand() > 0.5:
            img_volume, mask_volume = self.random_rotate_xy(img_volume, mask_volume, 180)
            
        return img_volume, mask_volume
    def random_rotate_xy(self, img_volume, mask_volume, max_angle=30):
    
        # 随机生成旋转角度
        angle = np.random.uniform(-max_angle, max_angle)
    
        num_slices = img_volume.shape[3]
        # 应用旋转矩阵到体积上
        rotated_img_volume = np.zeros_like(img_volume)
        rotated_mask_volume = np.zeros_like(mask_volume)

        for z in range(num_slices):
            rotated_img_volume[0, :, :, z] = rotate(img_volume[0, :, :, z], angle, reshape=False)
            rotated_mask_volume[0, :, :, z] = rotate(mask_volume[0, :, :, z], angle, reshape=False)
            rotated_img_volume[0, :, :, z] = np.clip(rotated_img_volume[0, :, :, z], 0, 1)
            rotated_mask_volume[0, :, :, z] = np.clip(rotated_mask_volume[0, :, :, z], 0, 1)
        rotated_mask_volume = (rotated_mask_volume > 0.5).astype(int)
        return rotated_img_volume, rotated_mask_volume


def get_train_dataset(args):
    # train 

    # file_name = '../numpy_data/train/imgs.pkl'  
    file_name = 'D:/SAM/numpy_data_slice/train/small/imgs.pkl'  
    with open(file_name, 'rb') as file:
        images = joblib.load(file)  


    # file_name = '../numpy_data/train/labels.pkl'
    file_name = 'D:/SAM/numpy_data_slice/train/small/labels.pkl'
    with open(file_name, 'rb') as file:
        labels = joblib.load(file)  


    # file_name = '../numpy_data/train/texts.pkl'  
    file_name = 'D:/SAM/numpy_data_slice/train/small/texts.pkl' 
    with open(file_name, 'rb') as file:
        texts = joblib.load(file)   
    print(len(images))
    train_slice_list = [slice(0, 70, 1), slice(80, 150, 1), slice(160, 230, 1), slice(240, 310, 1), slice(320, 390, 1), slice(400, 470, 1)] 
    valid_slice_list = [slice(70, 80, 1), slice(150, 160, 1), slice(230, 240, 1), slice(310, 320, 1), slice(390, 400, 1), slice(470, 480, 1)] 

    train_imgs = images[train_slice_list[0]] + images[train_slice_list[1]] + images[train_slice_list[2]] + images[train_slice_list[3]] +images[train_slice_list[4]] + images[train_slice_list[5]]
    valid_imgs = images[valid_slice_list[0]] + images[valid_slice_list[1]] + images[valid_slice_list[2]] + images[valid_slice_list[3]] +images[valid_slice_list[4]] + images[valid_slice_list[5]]
    train_labels = labels[train_slice_list[0]] + labels[train_slice_list[1]] + labels[train_slice_list[2]] + labels[train_slice_list[3]] +labels[train_slice_list[4]] + labels[train_slice_list[5]]
    valid_labels = labels[valid_slice_list[0]] + labels[valid_slice_list[1]] + labels[valid_slice_list[2]] + labels[valid_slice_list[3]] +labels[valid_slice_list[4]] + labels[valid_slice_list[5]]
    train_texts = texts[train_slice_list[0]] + texts[train_slice_list[1]] + texts[train_slice_list[2]] + texts[train_slice_list[3]] +texts[train_slice_list[4]] + texts[train_slice_list[5]]
    valid_texts = texts[valid_slice_list[0]] + texts[valid_slice_list[1]] + texts[valid_slice_list[2]] + texts[valid_slice_list[3]] +texts[valid_slice_list[4]] + texts[valid_slice_list[5]]


    train_dataset = MultiLabelDataset(train_imgs, train_labels, train_texts, RandomFlipAndRotate3D())
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataset = MultiLabelDataset(valid_imgs, valid_labels, valid_texts)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)   

    return train_dataloader, valid_dataloader #, eval_train_dataloader, eval_valid_dataloader


if __name__ == "__main__":
    args="foo"
    train_dataloader, valid_dataloader = get_train_dataset(args)
    for x, y in zip(train_dataloader, valid_dataloader):
        print(x["image"].shape)