import torch
import torch.nn as nn
import torchmetrics as tmt

class SSIM(nn.Module):
    def __init__(self, device="cpu"):
        super(SSIM, self).__init__()
        self.ssim = tmt.image.StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(device)

    def forward(self, preds, targets):
        return self.ssim(preds, targets)

class IOU(nn.Module):
    def __init__(self):
        super(IOU, self).__init__()
        self.smooth = 1.0e-9

    def forward(self, preds, targets):
        intersection = torch.sum(torch.abs(targets * preds), dim=[1,2,3])
        union = (torch.sum(targets, dim=[1,2,3]) + torch.sum(preds, dim=[1,2,3])) - intersection
        iou = torch.mean((intersection + self.smooth) / (union + self.smooth))
        return iou

class BASNetLoss(nn.Module):
    """BASNet hybrid loss."""
    
    def __init__(self, device="cpu"):
        super(BASNetLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.ssim = SSIM(device=device)
        self.iou = IOU()
        self.smooth = 1.0e-9
        self._is_train = True
        
    def train(self):
        self._is_train = True

    def eval(self):
        self._is_train = False

    def hybrid_loss(self, y_pred, y_true):
        bce_loss = self.bce_loss(y_pred, y_true)

        ssim_value = self.ssim(y_pred, y_true)
        ssim_loss = 1 - ssim_value + self.smooth

        iou_value = self.iou(y_pred, y_true)
        iou_loss = 1 - iou_value

        # Add all three losses
        return bce_loss + ssim_loss + iou_loss

    def forward(self, sup8, sup1, sup2, sup3, sup4, sup5, sup6, sup7, target):

        loss8 = self.hybrid_loss(sup8, target)

        if self._is_train:
            loss1 = self.hybrid_loss(sup1, target)
            loss2 = self.hybrid_loss(sup2, target)
            loss3 = self.hybrid_loss(sup3, target)
            loss4 = self.hybrid_loss(sup4, target)
            loss5 = self.hybrid_loss(sup5, target)
            loss6 = self.hybrid_loss(sup6, target)
            loss7 = self.hybrid_loss(sup7, target)
            
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
            return loss, loss8

        return loss8
