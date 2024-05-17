import os
import networkx as nx
import numpy
import numpy as np
from rivuletpy.rivulet import rtrace
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D
import skfmm
from scipy import ndimage
import pickle as pkl
import shutil
import time
import torch
import matplotlib.pyplot as plt


# _cenertline
def graph_construct(img_file,seg_file,size,edge_dist_thresh,save_dir):
    image_name = str(img_file.split('/')[-1].split('.')[0].split('_img')[0])
    print(image_name,"start")
    image_itk = nib.load(img_file)
    image_arr = image_itk.get_fdata().astype(np.float32)  # x, y, z
    seg_itk = nib.load(seg_file)
    seg_arr = seg_itk.get_fdata().astype(np.float32)  # x, y, z
    # print(seg_arr.shape)

    xyz_list = []
    max_val = []
    max_pos = []
    label_idx = []
    num_node = 0
    num_edge = 0

    z_size = image_arr.shape[2]
    padding_z = (16 - z_size % 16) % 16
    img = np.pad(image_arr, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
    vessel_prob = np.pad(seg_arr, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
    y_size = img.shape[1]
    padding_y = (16 - y_size % 16) % 16
    img = np.pad(img, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
    vessel_prob = np.pad(vessel_prob, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
    x_size = img.shape[0]
    padding_x = (16 - x_size % 16) % 16
    img = np.pad(img, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
    vessel_prob = np.pad(vessel_prob, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')

    # find local maxima
    im_x = img.shape[0]
    im_y = img.shape[1]
    im_z = img.shape[2]
    x_quan = range(0, im_x, int(size / 2))
    x_quan = sorted(list(set(x_quan) | set([im_x])))
    y_quan = range(0, im_y, size)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    z_quan = range(0, im_z, size)
    z_quan = sorted(list(set(z_quan) | set([im_z])))


    for x_idx in range(len(x_quan) - 1):
        for y_idx in range(len(y_quan) - 1):
            for z_idx in range(len(z_quan) - 1):
                cur_patch = vessel_prob[x_quan[x_idx]:x_quan[x_idx + 1], y_quan[y_idx]:y_quan[y_idx + 1], z_quan[z_idx]:z_quan[z_idx + 1]]
                if np.sum(cur_patch) == 0 :
                    max_val.append(0)
                    max_pos.append((x_quan[x_idx] + int(cur_patch.shape[0] / 2),
                                    y_quan[y_idx] + int(cur_patch.shape[1] / 2),
                                    z_quan[z_idx] + int(cur_patch.shape[2] / 2)))
                    label_idx.append(0)
                    continue
                else:
                    # temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                    # max_pos.append((x_quan[x_idx] + temp[0], y_quan[y_idx] + temp[1],z_quan[z_idx] + temp[2]))
                    max_val.append(np.max(cur_patch))
                    temp = np.zeros(3)
                    count = 0
                    for i_0 in range(cur_patch.shape[0]):
                        for i_1 in range(cur_patch.shape[1]):
                            for i_2 in range(cur_patch.shape[2]):
                                if cur_patch[i_0, i_1, i_2] == 1:
                                    temp += np.array([i_0, i_1, i_2])
                                    count += 1
                    temp = np.around(temp / count).astype(int)
                    max_pos.append((x_quan[x_idx] + temp[0], y_quan[y_idx] + temp[1], z_quan[z_idx] + temp[2]))
                    label_idx.append(1)
                num_node += 1

    graph = nx.Graph()
    # add nodes
    for node_idx, (node_x, node_y,node_z) in enumerate(max_pos):
        graph.add_node(node_idx , x=node_x, y=node_y, z=node_z ,node_label = label_idx[node_idx])
        xyz_list.append((node_x,node_y,node_z))
        # print(node_x,node_y,node_z)

    speed = seg_arr
    #边构建的距离阈值
    edge_method = 'geo_dist'
    node_list = list(graph.nodes)
    # print(len(node_list),node_list)
    # print(graph.nodes.data())
    # print(graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z'])
    for i, n in enumerate(node_list):
        if speed[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] == 0:
            continue
        neighbor = speed[max(0, graph.nodes[n]['x'] - 1): min(im_x, graph.nodes[n]['x'] + 2),
                         max(0, graph.nodes[n]['y'] - 1): min(im_x, graph.nodes[n]['y'] + 2),
                         max(0, graph.nodes[n]['z'] - 1): min(im_x, graph.nodes[n]['z'] + 2)],
        if np.mean(neighbor) < 0.1:
            continue

        if edge_method == 'geo_dist':
            # phi = np.ones_like(speed)
            # phi[graph.nodes[n]['x'], graph.nodes[n]['y'],graph.nodes[n]['z']] = -1
            mask = np.zeros_like(speed, dtype=bool)
            mask[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] = True
            dist = ndimage.distance_transform_edt(~mask)
            phi = np.where(mask, -dist, dist)
            tt = skfmm.travel_time(phi, speed, narrow=edge_dist_thresh)  # travel time

            for n_comp in node_list[i + 1:]:
                geo_dist = tt[graph.nodes[n_comp]['x'], graph.nodes[n_comp]['y'],graph.nodes[n_comp]['z']]  # travel time
                if geo_dist < edge_dist_thresh:
                    graph.add_edge(n, n_comp, weight=edge_dist_thresh / (edge_dist_thresh + geo_dist))
                    num_edge += 1
                    # print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')
    # print('Generate total', num_node, 'nodes, ', num_edge, 'edges.')

    #
    ## The graph to visualize
    # node_xyz_list = np.array(xyz_list)
    # edge_list = np.array(graph.edges)
    # edge_xyz_list = []
    # #
    # for idx in range(len(edge_list)):
    #     for node_idx in range(len(node_list)):
    #         if edge_list[idx][0] == node_list[node_idx] :
    #             edge_xyz_list.append([tuple(node_xyz_list[edge_list[idx][0]]),tuple(node_xyz_list[edge_list[idx][1]])])
    # # Create the 3D figure
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d", )
    # # Plot the nodes - alpha is scaled by "depth" automatically
    # ax.scatter(*node_xyz_list.T, s=5, ec="r")
    # # Plot the edges
    # for vizedge in np.array(edge_xyz_list):
    #     ax.plot(*vizedge.T, color="tab:gray")
    # def _format_axes(ax):
    #     """Visualization options for the 3D axes."""
    #     # Turn gridlines off
    #     ax.grid(False)
    #     # Suppress tick labels
    #     for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    #         dim.set_ticks([])
    #     # Set axes labels
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("z")
    # _format_axes(ax)
    # fig.tight_layout()
    # plt.show()
    # Save the graph as files
    file_name = image_name.split('_p')[0]
    save_dir = save_dir + '/'+  file_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    graph_save_path = os.path.join(save_dir, image_name + '_' + str(size) + '.gpickle')
    nx.write_gpickle(graph, graph_save_path, protocol=pkl.HIGHEST_PROTOCOL)
    return graph

if __name__ == '__main__':

    centerline = '/media/Data/yanxc/Liver_vessel/pre_data/val/'
    save_dir = '/media/Data/yanxc/Liver_vessel/pre_data/graph_val'
    seg_paths = [os.path.join(centerline, x, y)
                    for x in os.listdir(centerline)
                    for y in os.listdir(os.path.join(centerline, x))
                    if y.endswith('_seg.nii.gz')]
    img_paths = [os.path.join(centerline, x, y)
                    for x in os.listdir(centerline)
                    for y in os.listdir(os.path.join(centerline, x))
                    if y.endswith('_img.nii.gz')]


    seg_paths.sort()
    img_paths.sort()
    print(len(seg_paths),seg_paths)
    print(len(img_paths), img_paths)

    for idx in range(len(img_paths)):
        print(img_paths[idx])
        time_start = time.time()
        graph = graph_construct(img_paths[idx], seg_paths[idx], size=12, edge_dist_thresh=20,save_dir=save_dir)
        lens = len(list(graph.nodes))
        pos = torch.zeros((lens, 3))

        for i in range(lens):
            x, y, z = graph.nodes[i]['x'] + 1, graph.nodes[i]['y'] + 1, graph.nodes[i]['z'] + 1  # 加一是因为在三维矩阵中索引是从0开始(0-127)，而在图像中是从1开始(1-128)，为了保证坐标轴对应
            pos[i] = torch.tensor([x, y, z], dtype=torch.float)

        edge_index = torch.tensor(list(graph.edges), dtype=torch.long)
        x = torch.tensor(list(graph.nodes), dtype=torch.float)
        target = torch.tensor([graph.nodes[i]['node_label'] for i in range(lens)], dtype=torch.long)
        print('节点个数:', len(list(graph.nodes)), ' ', '边个数:', len(list(graph.edges)), ' ')
        print('********** Time:', time.time() - time_start, '**********')
        # try:
        #     graph,_ = graph_construct(patch_paths[idx], seg_paths[idx],size=8,edge_dist_thresh=15)
        # except ValueError:
        #     print('跳过了',patch_paths[idx])
        #     patch_name = patch_paths[idx].split('\\')[-1].split('_c')[0]
        #     shutil.move(patch_paths[idx], "E:/graph_data/del_data/" + patch_name + '_cenertline.nii.gz')
        # continue

