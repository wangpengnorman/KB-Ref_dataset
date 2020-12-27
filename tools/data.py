from PIL import Image
from torch.utils.data import Dataset
import torch
import json
import os
import numpy as np
from torchvision import transforms
import random


def dataLoader(path):
    return Image.open(path).convert("RGB")


class data(Dataset):
    def __init__(self, jpg_path, label_path, data_transform=None, loader=dataLoader):
        super(data, self).__init__()
        jpg_name = []
        jpg_label = []
        f_label = []
        expression = []
        bbox = []
        facts = []
        w_h = []
        mask = []
        e_mask = []
        c_mask = []
        self.loader = dataLoader
        self.transform = data_transform
        with open(label_path) as file:
            Data = json.load(file)
        for x in Data:
            jpg_name.append(os.path.join(jpg_path, x['image'])+'.jpg')
            jpg_label.append(x['label'])
            f_label.append(0)
            expression.append(x['expression'])
            bbox.append(x['bbox'])
            facts.append(x['facts'])
            w_h.append(x['w_h'])
            mask.append(x['mask'])
            e_mask.append(x['e_mask'])
            c_mask.append(x['c_mask'])
        self.jpg_name = jpg_name
        self.jpg_label = jpg_label
        self.f_label = f_label
        self.expression = expression
        self.bbox = bbox
        self.facts = facts
        self.w_h = w_h
        self.mask = mask
        self.e_mask = e_mask
        self.c_mask = c_mask

    def __getitem__(self, item):
        jpg_name = self.jpg_name[item]
        jpg_label = self.jpg_label[item]
        jpg = self.loader(jpg_name)
        if self.transform is not None:
            jpg = self.transform(jpg)
        label = torch.LongTensor(1)
        label[0] = jpg_label
        f_label = self.f_label[item]
        f_label = torch.from_numpy(np.eye(500)[jpg_label*50+f_label].reshape(10, 50)).type(torch.FloatTensor)
        bboxs = self.bbox[item]
        local = []
        locations = []
        for bbox in bboxs:
            can = Image.open(jpg_name).crop([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]).convert("RGB")
            locations.append([bbox[0] / self.w_h[item][0], bbox[1] /self.w_h[item][1], (bbox[0]+bbox[2]) / self.w_h[item][0], (bbox[1]+bbox[3]) / self.w_h[item][1], bbox[2]*bbox[3] / self.w_h[item][0] /self.w_h[item][1]])
            if self.transform is not None:
                can = self.transform(can)
            #print(np.array(can).shape)
            local.append(np.array(can))
        local = torch.from_numpy(np.array(local)).view(-1, 224, 224)
        expression = self.expression[item][:self.e_mask[item]]
        #print(expression, type(expression))
        random.shuffle(expression)
        #print(np.array(expression))
        expression = np.pad(np.array(expression), (0, 50-self.e_mask[item]), 'constant', constant_values=(0, 15732))
        expression = torch.from_numpy(np.array(expression)).type(torch.LongTensor)
        #print(np.array(self.facts[item]).shape)
        facts = torch.from_numpy(np.array(self.facts[item])).type(torch.LongTensor).view(-1, 50)
        locations = torch.from_numpy(np.array(locations)).type(torch.FloatTensor)
        #print(self.mask[item])
        #print(np.array(self.mask[item]).shape)
        mask = []
        #print(len(self.mask[item][0]))
        #print(len(self.mask[item]))
        f_mask = []
        ff_mask = []
        for i in range(len(self.mask[item])):
            middle = []
            l = 0
            #print(len(self.mask[item][i]))
            #print(self.mask[item][i])
            for j in range(len(self.mask[item][0])):#len(self.mask[item][i])):
                if self.mask[item][i][j] > 0:
                    middle.append(list(np.pad(np.ones(self.mask[item][i][j]), (0, 50-self.mask[item][i][j])) / self.mask[item][i][j]))
                    #middle.append(list(np.eye(50)[self.mask[item][i][j]-1]))
                    l += 1
                else:
                    middle.append(list(np.zeros(50)))
            mask.append(middle)
            f_mask.append(list(np.pad(np.ones(l), (0, 50-l), 'constant')))
            ff_mask.append(list(np.eye(50)[l-1]))
        mask = np.array(mask)
        #print(mask.shape)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        f_mask = np.array(f_mask)
        f_mask = torch.from_numpy(f_mask).type(torch.FloatTensor)
        ff_mask = np.array(ff_mask)
        ff_mask = torch.from_numpy(ff_mask).type(torch.FloatTensor)
        #mask = torch.from_numpy(np.array(self.mask[item])).type(torch.FloatTensor)
        #e_mask = np.eye(50)[self.e_mask[item]-1]
        e_mask = np.pad(np.ones(self.e_mask[item]), (0, 50-self.e_mask[item]))# / self.e_mask[item]
        e_mask = torch.from_numpy(e_mask).type(torch.FloatTensor)
        #e_mask = torch.from_numpy(np.array(self.e_mask[item])).type(torch.FloatTensor)
        c_mask = np.pad(np.ones(self.c_mask[item]), (0, 10-self.c_mask[item]))# / self.e_mask[item]
        c_mask = torch.from_numpy(c_mask).type(torch.FloatTensor)
        return jpg, label, f_label, expression, e_mask, local, locations, facts, mask, f_mask, ff_mask, jpg_name, c_mask

    def __len__(self):
        return len(self.jpg_name)

		
if __name__ == '__main__':
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
