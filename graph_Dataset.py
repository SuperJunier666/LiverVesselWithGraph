import torch
import nibabel as nib
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch_geometric
import networkx as nx
from skimage import transform
import random
from torch_geometric.data import Data,Batch



class nei_graphDataset(Dataset):
    def __init__(self, image_dir, graph_dir, stage='train', ):
        super(nei_graphDataset, self).__init__()
        self.stage = stage
        self.image_paths = [os.path.join(image_dir + self.stage, x, y)
                            for x in os.listdir(image_dir + self.stage)
                            for y in os.listdir(os.path.join(image_dir + self.stage, x))
                            if y.endswith('_img.nii.gz')]
        self.label_paths = [os.path.join(image_dir + self.stage, x, y)
                            for x in os.listdir(image_dir + self.stage)
                            for y in os.listdir(os.path.join(image_dir + self.stage, x))
                            if y.endswith('_seg.nii.gz')]
        self.cenertline_paths = [os.path.join(image_dir + self.stage, x, y)
                                 for x in os.listdir(image_dir + self.stage)
                                 for y in os.listdir(os.path.join(image_dir + self.stage, x))
                                 if y.endswith('_cenertline.nii.gz')]
        self.graph_paths = [os.path.join(graph_dir + 'graph_' + self.stage, x, y)
                            for x in os.listdir(graph_dir + 'graph_' + self.stage)
                            for y in os.listdir(os.path.join(graph_dir + 'graph_' + self.stage, x))
                            if y.endswith('.gpickle')]

        self.image_paths.sort()
        self.label_paths.sort()
        self.graph_paths.sort()
        self.cenertline_paths.sort()
        # print(self.image_paths)
        # print(self.label_paths)
        # print(self.cenertline_paths)
        # print(self.graph_paths)

        assert len(self.image_paths) == len(self.label_paths)
        assert len(self.image_paths) == len(self.cenertline_paths)
        assert len(self.image_paths) == len(self.graph_paths)

    def __getitem__(self, index):
        image_itk = nib.load(self.image_paths[index])
        label_itk = nib.load(self.label_paths[index])
        center_itk = nib.load(self.cenertline_paths[index])
        # print(self.image_paths[index])

        image = image_itk.get_fdata()
        label = label_itk.get_fdata()
        centerline = center_itk.get_fdata()

        # z_size = image.shape[2]
        # padding_z = (16 - z_size % 16) % 16
        # image = np.pad(image, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        # label = np.pad(label, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        # centerline = np.pad(centerline, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        # y_size = image.shape[1]
        # padding_y = (16 - y_size % 16) % 16
        # image = np.pad(image, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        # label = np.pad(label, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        # centerline = np.pad(centerline, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        # x_size = image.shape[0]
        # padding_x = (16 - x_size % 16) % 16
        # image = np.pad(image, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        # label = np.pad(label, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        # centerline = np.pad(centerline, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')

        if self.stage == 'train':
            if random.randint(0, 1):
                shift_value = random.uniform(-0.1, 0.1)
                scale_value = random.uniform(0.9, 1.1)
                image = image * scale_value + shift_value
            if random.random() > 0.5:
                intensity_factor = random.uniform(0.9, 1.1)
                image *= intensity_factor
        image = (image - image.min()) / (image.max() - image.min()) * 255.
        # print(image.max(),image.min())
        image = image[np.newaxis]  # CDHW
        label = label[np.newaxis]  # CDHW
        centerline = centerline[np.newaxis]  # CDHW

        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)
        centerline = np.ascontiguousarray(centerline)

        image = torch.from_numpy(image / 255.).float()
        label = torch.from_numpy(label / 1.0).float()
        centerline = torch.from_numpy(centerline / 1.0).float()

        # print(type(image),type(label),type(centerline))

        graph = nx.read_gpickle(self.graph_paths[index])
        pos = []
        node_list = torch.tensor(list(graph.nodes))
        graph_label = torch.tensor([graph.nodes[i]['node_label'] for i in range(len(node_list))], dtype=torch.long)
        # graph_label = np.array(graph_label).astype(np.float)
        for nd in range(len(node_list)):
            pos.append([graph.nodes[nd]['x'], graph.nodes[nd]['y'], graph.nodes[nd]['z']])
        pos = torch.tensor(pos)
        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).permute(1,0)
        # edge_index = np.array(edge_index).astype(int)

        adj = nx.adjacency_matrix(graph).astype(float).todense()
        adj =  torch.tensor(adj,dtype=torch.float)

        data = Data(x=node_list,edge_index=edge_index, y=graph_label,pos=pos,Adjacency_matrix=adj)
        # assert torch.isfinite(image).all(), "image包含 NaN 或无穷大值"
        # assert torch.isfinite(label).all(), "label包含 NaN 或无穷大值"
        # assert torch.isfinite(centerline).all(), "centerline包含 NaN 或无穷大值"
        # assert torch.isfinite(data.pos).all(), "pos包含 NaN 或无穷大值"
        return image, label, centerline,data

    def __len__(self):
        return len(self.image_paths)



if __name__ == '__main__':
    if __name__ == '__main__':
        img_dir  = '/media/Data/yanxc/Liver_vessel/pre_data1/'
        graph_dir = '/media/Data/yanxc/Liver_vessel/pre_data1/'

        # img_dir = '../graph_data/data_train_val/code_test_data/img/'
        # graph_dir = '../graph_data/data_train_val/code_test_data/graph/'

        dataset = nei_graphDataset(img_dir,graph_dir, stage='train')
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False,num_workers=0,collate_fn=collate_fn)
        dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=2 ,shuffle=False,num_workers=4)
        # image, label = next(iter(dataloader))
        # print(len(dataset))
        for iteration, data in enumerate(dataloader):
            image, label, centerline,batch = data
            print("输入的统计信息: min={}, max={}, mean={}, std={}".format(image.min().item(), image.max().item(),image.mean().item(), image.std().item()))
            # print("输入的统计信息: min={}, max={}".format(label.min().item(), label.max().item(),))
            # print("输入的统计信息: min={}, max={}".format(centerline.min().item(), centerline.max().item(), ))
            print(np.unique(label),np.unique(centerline))
            # print("节点特征的形状:",batch.x.shape)
            # print("节点位置的形状:",batch.pos.shape)
            # print(torch.unique(batch.batch))
            # print(batch.y.shape)
            # print(batch.adj.shape)

            # upper_tri = np.triu(adj.squeeze(0), k=1)
            # edge_index = np.where(upper_tri != 0)
            # edge_index = np.array(edge_index)






