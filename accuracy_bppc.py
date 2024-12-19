from lib.backbone.lib.utils.evaluate import accuracy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def cal_acc(hrnet_heatmap, apc_heatmap, gt_heatmap):
    acc_h = AverageMeter()
    acc_a = AverageMeter()
    
    _, avg_acc_h, cnt_h, pred_h = accuracy(hrnet_heatmap.cpu().numpy(), gt_heatmap.cpu().numpy())
    _, avg_acc_a, cnt_a, pred_a = accuracy(apc_heatmap.cpu().numpy(), gt_heatmap.cpu().numpy())
    acc_h.update(avg_acc_h, cnt_h)
    acc_a.update(avg_acc_a, cnt_a)

    return acc_h, acc_a


def cal_conf_acc(hrnet_heatmap, apc_heatmap, gt_heatmap, idx):
    acc_h = AverageMeter()
    acc_a = AverageMeter()

    hrnet_heatmap = hrnet_heatmap[idx[1], idx[2], :, :]
    apc_heatmap = apc_heatmap[idx[1], idx[2], :, :]
    gt_heatmap = gt_heatmap[idx[1], idx[2], :, :]
    
    _, avg_acc_h, cnt_h, pred_h = accuracy(hrnet_heatmap.unsqueeze(0).unsqueeze(0).cpu().numpy(), gt_heatmap.unsqueeze(0).unsqueeze(0).cpu().numpy())
    _, avg_acc_a, cnt_a, pred_a = accuracy(apc_heatmap.unsqueeze(0).unsqueeze(0).cpu().numpy(), gt_heatmap.unsqueeze(0).unsqueeze(0).cpu().numpy())
    acc_h.update(avg_acc_h, cnt_h)
    acc_a.update(avg_acc_a, cnt_a)

    return acc_h, acc_a

    