import time
import copy
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from loss import WCEDCELoss,dice_metric
from utils import *
import torch.nn as nn
import torch_geometric
from torch.utils import data
from scipy.ndimage import zoom
from graph_Dataset import nei_graphDataset
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")

class MixLoss(nn.Module):
    def __init__(self):
        super(MixLoss, self).__init__()

    #     self.weights = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0]), requires_grad=True)
    def forward(self, loss1, loss2,loss3):
    #     weights = F.softmax(self.weights, dim=0)
        # print('Weight:{:.3f},{:.3f}  '.format(alpha.cpu().detach().numpy(),belt.cpu().detach().numpy()) ,end='')
        a = (loss1+1e-4)/(loss1+loss2+loss3)
        b = (loss2+1e-4)/(loss1+loss2+loss3)
        c = (loss3+1e-4)/(loss1+loss2+loss3)
        loss =  a * loss1 + b * loss2 + c * loss3
        # loss = 0.5 * loss1 + 0.5* loss2
        return loss

def trans_cnn_feature(feature_map, node_positions,device):
    import numpy as np
    # print('node_positions',node_positions.shape)
    batch_size, n_feat, x,y,z  = feature_map.shape
    total_nodes = node_positions.shape[0]
    num_nodes_per_batch = total_nodes // batch_size
    # 初始化新特征张量
    new_feature = torch.zeros((total_nodes, n_feat), dtype=torch.float)
    # 计算每个节点对应的批次索引
    batch_indices = torch.arange(total_nodes) // num_nodes_per_batch
    # 遍历每个节点，提取特征
    for node_index in range(total_nodes):
        batch_index = batch_indices[node_index]
        new_feature[node_index, :] = feature_map[batch_index, :, node_positions[node_index, 0], node_positions[node_index, 1], node_positions[node_index, 2]]
    return new_feature.to(device)

def extract_neighborhood_features(feature_map, batch,device,neighborhood_size=2):
    from torch_geometric.data import Data
    batch_size, n_feat, _,_,_  = feature_map.shape
    padded_feat_map = torch.nn.functional.pad(feature_map, (neighborhood_size, neighborhood_size, neighborhood_size,
                                                            neighborhood_size, neighborhood_size, neighborhood_size),mode='constant', value=0)
    total_nodes = batch.pos.shape[0]
    num_nodes_per_batch = total_nodes // batch_size
    neighborhood_dim = neighborhood_size * 2 + 1
    # 初始化新特征张量
    new_feature = torch.zeros((total_nodes, n_feat * neighborhood_dim**3), dtype=torch.float)
    # 计算每个节点对应的批次索引
    batch_indices = torch.arange(total_nodes) // num_nodes_per_batch

    # 遍历每个节点，提取特征
    for node_index in range(total_nodes):
        batch_index = batch_indices[node_index]
        x, y, z = batch.pos[node_index] + neighborhood_size
        neighborhood = padded_feat_map[batch_index, :, x - neighborhood_size : x + neighborhood_size + 1,
                                                       y - neighborhood_size : y + neighborhood_size + 1,
                                                       z - neighborhood_size : z + neighborhood_size + 1]
        # print(neighborhood.shape)
        new_feature[node_index] = neighborhood.contiguous().view(-1)
    # gdata = Data(x=new_feature, edge_index=batch.edge_index, batch=batch.batch)
    return new_feature.to(device)

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    ############################
    # Parameters
    ############################
    k = 516
    NUM_EPOCHS = 60
    LEARNINGRATE = 1e-4
    NUM_CLASS = 2
    Batch_size = 2

    ############################
    # load the net
    ###########################
    from torch_geometric.nn import GraphUNet
    from model.SubeNet import SubeNet
    Architectures = ['']
    architecture = Architectures[-1]
    inet1 = SubeNet(in_channels=1, out_channels=2, base_filters_num=16)
    inet2 = GraphUNet(in_channels=432, hidden_channels=512, out_channels=2,depth=4)

    print('#Unet parameters:', sum(param.numel() for param in (inet1).parameters()))
    print('#GraphNet parameters:', sum(param.numel() for param in (inet2).parameters()))

    inet1 = inet1.to(device)
    inet2 = inet2.to(device)
    # inet1.load_state_dict(torch.load('/media/Data/yanxc/Liver_vessel/graph_projtct/checkpoints/514/net_514_best_epoch_21.pth'))
    logpath = 'statistics/' + architecture + str(k) + '/'
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    train_logger = logger(logpath + 'train_' + architecture + '.log')
    train_logger.info('{}-fold start training!'.format(k))
    ############################
    # load data
    ###########################
    #-----------------------K-fold  使用dataset--------------------
    img_dir = '/media/Data/yanxc/Liver_vessel/pre_data1/'
    graph_dir = '/media/Data/yanxc/Liver_vessel/pre_data1/'

    Train_set = nei_graphDataset(img_dir,graph_dir,stage='train')
    Val_set = nei_graphDataset(img_dir,graph_dir, stage='val')
    train_size = int( Train_set.__len__())
    #print(train_size)

    train_loader = torch_geometric.loader.DataLoader(dataset=Train_set, num_workers=4, batch_size=Batch_size, shuffle=True, pin_memory=True,drop_last=True)
    val_loader = torch_geometric.loader.DataLoader(dataset=Val_set, num_workers=4, batch_size=Batch_size, shuffle=True, pin_memory=True)
    mix_loss = MixLoss().to(device)
    ############################
    # loss and optimization
    ###########################
    criterion1 = WCEDCELoss(num_classes=NUM_CLASS,intra_weights=torch.tensor([1., 3.]).to(device), device=device, inter_weights=0.5)
    criterion1_cl = WCEDCELoss(num_classes=NUM_CLASS, intra_weights=torch.tensor([1., 9.]).to(device),device=device, inter_weights=0.5)
    criterion2 = torch.nn.CrossEntropyLoss()

    optimizer1 = optim.Adam([{'params': inet1.parameters()},{'params': inet2.parameters()}], lr=LEARNINGRATE,  weight_decay=0.0005,eps=1e-4)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1,T_max=NUM_EPOCHS)

    ############################
    # Train the net
    ############################
    results = {'loss': [], 'dice': [],  'val_loss': [], 'val_dice': []}
    # for saving best model
    best_model_wts = copy.deepcopy(inet1.state_dict())
    best_loss = np.inf
    best_epoch = 0
    since = time.time()
    # TODO
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(1, NUM_EPOCHS + 1):
        # adjust_lr(optimizer=optimizer, init_lr=LEARNINGRATE, epoch=epoch, decay_rate=0.1, decay_epoch=5)
        print('学习率:{} :'.format(optimizer1.state_dict()['param_groups'][0]['lr']))
        epochresults = {'loss': [], 'dice': [], 'val_loss': [], 'val_dice': []}
        inet1.train()
        inet2.train()
        for iteration, data in enumerate(train_loader):
            image, label, centerline, batch = data

            image = image.to(device)
            label = label.to(device)

            centerline = centerline.to(device)
            batch = batch.to(device)

            pred_ves, pred_cl,feature_map,ds_3,ds_2,ds_1= inet1(image)
            ## feature_map = torch.rand(Batch_size, 8, 160, 160, 96)
            # out_fm = extract_neighborhood_features(feature_map, batch,device,neighborhood_size=1)
            out_fm = trans_cnn_feature(feature_map, batch.pos,device)
            # out_fm = torch.rand(3024, 1000).to(device)
            graph_output = inet2(x=out_fm, edge_index=batch.edge_index, batch=batch.batch)

            loss1 = criterion1(pred_ves, label.squeeze(1).long())
            loss_ds1 = criterion1(ds_1, torch.from_numpy( zoom(label.squeeze(1).cpu().numpy(), zoom=[1., 1. / 2., 1. / 2., 1. / 2.], order=0,mode='nearest')).long().to(device))
            loss_ds2 = criterion1(ds_2, torch.from_numpy(zoom(label.squeeze(1).cpu().numpy(), zoom=[1., 1. / 4., 1. / 4., 1. / 4.], order=0,mode='nearest')).long().to(device))
            loss_ds3 = criterion1(ds_3, torch.from_numpy(zoom(label.squeeze(1).cpu().numpy(), zoom=[1., 1. / 8., 1. / 8., 1. / 8.], order=0,mode='nearest')).long().to(device))
            loss_deep =  0.4 * loss1 + 0.3 * loss_ds1 + 0.2 * loss_ds2 + 0.1 * loss_ds3

            loss_cl = criterion1_cl(pred_cl, centerline.squeeze(1).long())
            loss2 = criterion2(graph_output.unsqueeze(0).permute(0, 2, 1), batch.y.unsqueeze(0).long())
            loss = mix_loss(loss_deep,loss_cl,loss2)
            optimizer1.zero_grad()


            loss.backward()
            torch.nn.utils.clip_grad_norm_(inet1.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(inet2.parameters(), max_norm=1)
            optimizer1.step()

            ves_dice = dice_metric(pred_ves,label.squeeze(1).long())
            # ves_dice = dice[0, :][1]
            # ves_dice = (((dice[0,:][1]+dice[0,:][2])/2) + ((dice[1,:][1]+dice[1,:][2])/2))/2

            if iteration % 10 == 0:
                train_logger.info("Train: Epoch/Epoches {}/{}\t"
                                  "iteration/iterations {}/{}\t"
                                  "loss {:.3f}\t"
                                  "CNN loss {:.3f}\t"
                                  "GNN loss {:.3f}\t"
                                  "Dice {:.2f}\t".format(epoch, NUM_EPOCHS, iteration,
                                                         len(train_loader),
                                                         loss.item(),
                                                         loss1.item(),
                                                         loss2.item(),
                                                         ves_dice.cpu().detach().numpy()))
            epochresults['loss'].append(loss1.item())
            epochresults['dice'].append(ves_dice.cpu().detach().numpy())

        results['loss'].append(np.mean(epochresults['loss']))
        results['dice'].append(np.mean(epochresults['dice']))

        ############################
        # validate the net
        ############################
        inet1.eval()
        inet2.eval()
        with torch.no_grad():
            for val_iteration, val_data in enumerate(val_loader):
                val_image, val_label, val_centerline, val_batch = val_data

                val_image = val_image.to(device)
                val_label = val_label.to(device)
                val_centerline = val_centerline.to(device)
                val_batch = val_batch.to(device)

                val_pred_ves,val_pred_cl,_,_,_,_ = inet1(val_image)
                val_loss = criterion1(val_pred_ves, val_label.squeeze(1).long())
                ves_dice_val = dice_metric(val_pred_ves,val_label.squeeze(1).long())

                # ves_dice_val = val_dice[0, :][1]
                # ves_dice_val = (((val_dice[0, :][1] + val_dice[0, :][2]) / 2) + ((val_dice[1, :][1] + val_dice[1, :][2]) / 2)) / 2
                if val_iteration % 10 == 0:
                    train_logger.info("Val: Epoch/Epoches {}/{}\t"
                                      "iteration/iterations {}/{}\t"
                                      "val loss {:.3f}\t"
                                      "Dice {:.2f}\t".format(epoch, NUM_EPOCHS, val_iteration,
                                                             len(val_loader), val_loss.item(),
                                                             ves_dice_val.cpu().detach().numpy(), ))
                    epochresults['val_loss'].append(val_loss.item())
                    epochresults['val_dice'].append(ves_dice_val.cpu().detach().numpy())

            results['val_loss'].append(np.mean(epochresults['val_loss']))
            results['val_dice'].append(np.mean(epochresults['val_dice']))

            train_logger.info("Average: Epoch/Epoches {}/{}\t"
                        "train epoch loss {:.3f}\t"
                        "val epoch loss {:.3f}\t"
                        "train epoch dice {:.3f}\t"
                        "val epoch dice {:.3f}\t".format(epoch, NUM_EPOCHS, np.mean(epochresults['loss']),
                                                           np.mean(epochresults['val_loss']),
                                                           np.mean(epochresults['dice']),
                                                           np.mean(epochresults['val_dice']),))
        # saving the best model parameters
        val_dice = np.mean(epochresults['val_dice'])
        if np.mean(epochresults['val_loss']) <= best_loss:
            best_loss = np.mean(epochresults['val_loss'])
            best_model1_wts = copy.deepcopy(inet1.state_dict())
            # best_model2_wts = copy.deepcopy(inet2.state_dict())
            best_epoch = epoch

            filepath = 'checkpoints/' + architecture + str(k) + '/'
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            torch.save(best_model1_wts, filepath + '/net_' + str(k) + '_best_epoch_%d_%.3f.pth' % (best_epoch,val_dice))
            # torch.save(best_model2_wts, filepath + '/gat_' + str(k) + '_best_epoch_%d.pth' % best_epoch)
        if NUM_EPOCHS-epoch < 5:
            filepath = 'checkpoints/' + architecture + str(k) + '/'
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            torch.save(inet1.state_dict(), filepath + '/net_' + str(k) + '_save_epoch_%d_%.3f.pth' % epoch,val_dice)
            # torch.save(inet2.state_dict(), filepath + '/gat_' + str(k) + '_save_epoch_%d.pth' % epoch)
        scheduler1.step()
        torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    train_logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    train_logger.info('{}-fold finish training!'.format(k))

    ############################
    # save the results
    ############################
    data_frame = pd.DataFrame(
        data={'loss': results['loss'],
              'val_loss': results['val_loss'],
              'dice': results['dice'],
              'val_dice': results['val_dice'],
              },
        index=range(1, NUM_EPOCHS + 1))
    data_frame.to_csv('statistics/' + architecture + str(k) + '/results' + str(k) + '.csv', index_label='Epoch')
    #
    # ############################
    # # plot the results
    # ############################
    plt.figure()
    plt.title("Loss During Training and Validating")
    plt.plot(results['loss'], label="Train")
    plt.plot(results['val_loss'], label="Val")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('statistics/' + architecture + str(k) + '/train_epoch_losses' + str(k) + '.tif')

    plt.figure()
    plt.title("Dice During Training and Validating")
    plt.plot(results['dice'], label="Train")
    plt.plot(results['val_dice'], label="Val")
    plt.xlabel("epochs")
    plt.ylabel("Dice")
    plt.legend()
    plt.savefig('statistics/' + architecture + str(k) + '/dice' + str(k) + '.tif')
    train_logger.info('train finished')





