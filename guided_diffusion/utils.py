
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff


softmax_helper = lambda x: F.softmax(x, 1)
sigmoid_helper = lambda x: F.sigmoid(x)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def staple(a):
    # a: n,c,h,w detach tensor
    # print("a:",a.shape)#1,256,256
    mvres = mv(a)#1,256,256
    # print("mvres:",mvres.shape)
    gap = 0.4
    if gap > 0.02:
        for i, s in enumerate(a):
            # print("i:",i)#0
            # print("s:",s.shape)#256,256
            r = s * mvres
            res = r if i == 0 else torch.cat((res,r),0)
        nres = mv(res)
        gap = torch.mean(torch.abs(mvres - nres))
        mvres = nres
        a = res
    return mvres

def allone(disc,cup):
    disc = np.array(disc) / 255
    cup = np.array(cup) / 255
    res = np.clip(disc * 0.5 + cup,0,1) * 255
    res = 255 - res
    res = Image.fromarray(np.uint8(res))
    return res

def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:

    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def haussdorf(preds, target):
    # print("n_pred:", preds.shape)
    # print("mask:",target.shape)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()
    res = numpy_haussdorf(n_pred, n_target)

    return res

def dice_score(pred, targs):
    # pred = (pred > 0.5)
    smooth = 1
    return (2. * (pred*targs).sum() + smooth) / ((pred+targs).sum()+smooth)

def iou_score2(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        # output = torch.sigmoid(output).data.cpu().numpy()
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def dice_coeff2(output, target):
    smooth = 1e-5

    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

    return dice

def mv(a):
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)#求第一维的大小
    return torch.sum(a, 0, keepdim=True) / b

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image

def export(tar, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tar.size(1)
    if c == 3:
        vutils.save_image(tar, fp = img_path)
    else:
        s = th.tensor(tar)[:,-1,:,:].unsqueeze(1)
        s = th.cat((s,s,s),1)
        vutils.save_image(s, fp = img_path)

def norm(t):
    m, s, v = torch.mean(t), torch.std(t), torch.var(t)
    return (t - m) / s
