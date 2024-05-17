import torch
from torch import nn
from torch.nn.functional import one_hot

# def dice_metric(pred,label):
#
#     label_onehot = one_hot(label, num_classes=2).permute(0, 4, 1, 2, 3).contiguous()
#     smooth = 1e-5
#     prediction = torch.softmax(pred, dim=1)
#     batchsize = label_onehot.size(0)
#     num_classes = label_onehot.size(1)
#     prediction = prediction.view(batchsize, num_classes, -1)
#     target = label_onehot.view(batchsize, num_classes, -1)
#
#     intersection = (prediction * target)
#
#     dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
#     # print(dice[0,:])
#     return dice
def dice_metric(pred,label):
    label = label.long()
    num_classes = 2
    label_onehot = one_hot(label, num_classes=num_classes).permute(0, 4, 1, 2, 3).contiguous()
    pred = torch.softmax(pred, dim=1)
    pred = pred.view(pred.size(0),num_classes,-1)
    label_onehot = label_onehot.view(label_onehot.size(0), num_classes, -1)
    intersection = torch.sum(pred * label_onehot,dim=2)
    union = torch.sum(pred,dim=2) + torch.sum(label_onehot,dim=2)
    dice = (2. * intersection +1e-6) / (union + 1e-6)
    dice_score = torch.mean(dice,dim=1)
    return dice_score.mean()


class WCEDCELoss(nn.Module):
    def __init__(self, num_classes=2, inter_weights=0.5, intra_weights=None, device='cuda'):
        super(WCEDCELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=intra_weights)
        self.num_classes = num_classes
        self.intra_weights = intra_weights
        self.inter_weights = inter_weights
        self.device = device

    def dice_loss(self, prediction, target, weights):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        smooth = 1e-5

        prediction = torch.softmax(prediction, dim=1)
        batchsize = target.size(0)
        num_classes = target.size(1)
        prediction = prediction.view(batchsize, num_classes, -1)
        target = target.view(batchsize, num_classes, -1)

        intersection = (prediction * target)

        dice = (2. * intersection.sum(2) + smooth) / (prediction.sum(2) + target.sum(2) + smooth)
        # print('dice: ', dice)
        dice_loss = 1 - dice.sum(0) / batchsize
        weighted_dice_loss = dice_loss * weights

        # print(dice_loss, weighted_dice_loss)
        return weighted_dice_loss.mean()

    def forward(self, pred, label):
        """Calculating the loss and metrics
            Args:
                prediction = predicted image
                target = Targeted image
                metrics = Metrics printed
                bce_weight = 0.5 (default)
            Output:
                loss : dice loss of the epoch """
        cel = self.ce_loss(pred, label)
        label_onehot = one_hot(label, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).contiguous()
        dicel = self.dice_loss(pred, label_onehot, self.intra_weights)
        # print('ce: ', cel*self.inter_weights, 'dicel: ', dicel * (1 - self.inter_weights))
        loss = cel * self.inter_weights + dicel * (1 - self.inter_weights)
        return loss


if __name__ == '__main__':
    wcedceloss = WCEDCELoss(intra_weights=torch.tensor([1., 3., 3.]).cuda())
    label = torch.randint(low=0, high=2, size=[2, 64, 64, 64]).cuda()
    print(one_hot(label, 3).permute(0, 4, 1, 2, 3).contiguous().shape)
    prediction = torch.randn([2, 3, 64, 64, 64]).cuda()
    loss = wcedceloss(pred=prediction, label=label)
    print('loss: ', loss)