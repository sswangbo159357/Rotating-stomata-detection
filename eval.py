import os
import cv2
import math
import time
import torch
import evaluation
import numpy as np
from models import DlaNet
import matplotlib.pyplot as plt
from test import pre_process, ctdet_decode, post_process, merge_outputs

def process(images, return_time=False):
    with torch.no_grad():
        output = model(images)
        hm = output['hm'].sigmoid_()
        ang = output['ang'].relu_()
        conduct = output['conduct'].relu_()
        wh = output['wh']
        reg = output['reg']

        torch.cuda.synchronize()
        forward_time = time.time()
        dets = ctdet_decode(hm, wh, conduct, ang, reg=reg, K=100)

    if return_time:
        return output, dets, forward_time
    else:
        return output, dets

def iou_rotate_calculate(boxes1, boxes2):
    area1 = boxes1[2] * boxes1[3]
    area2 = boxes2[2] * boxes2[3]
    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])
    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        ious = int_area * 1.0 / (area1 + area2 - int_area)
    else:
        ious = 0
    return ious

def get_lab_ret(xml_path):
    ret = []
    with open(xml_path, 'r', encoding='UTF-8') as fp:
        ob = []
        flag = 0
        for p in fp:
            key = p.split('>')[0].split('<')[1]
            if key == 'cx':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'cy':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'w':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'h':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'angle':
                ob.append(p.split('>')[1].split('<')[0])
                flag = 1
            if flag == 1:
                x1 = float(ob[0])
                y1 = float(ob[1])
                w = float(ob[2])
                h = float(ob[3])
                angle = float(ob[4]) * 180 / math.pi
                angle = angle if angle < 180 else angle - 180
                bbox = [x1, y1, w, h, angle]
                ret.append(bbox)
                ob = []
                flag = 0
    return ret

def get_pre_ret(img_path, device):
    image = cv2.imread(img_path)
    images, meta = pre_process(image)
    images = images.to(device)
    output, dets, forward_time = process(images, return_time=True)

    dets = post_process(dets, meta)
    ret = merge_outputs(dets)
    res = np.empty([1, 7])
    for i, c in ret.items():
        tmp_s = ret[i][ret[i][:, 5] > 0.3]
        tmp_c = np.ones(len(tmp_s)) * (i + 1)
        tmp = np.c_[tmp_c, tmp_s]
        res = np.append(res, tmp, axis=0)
    res = np.delete(res, 0, 0)
    res = res.tolist()
    return res


def pre_recall(root_path, device, iou=0.5):
    imgs = os.listdir(root_path)
    num = 0
    tn = 0
    all_pre_num = 0
    all_lab_num = 0
    for img in imgs:
        if img.split('.')[-1] == 'jpg':
            img_path = os.path.join(root_path, img)
            xml_path = os.path.join(root_path, img.split('.')[0] + '.xml')
            pre_ret = get_pre_ret(img_path, device)
            lab_ret = get_lab_ret(xml_path)
            all_pre_num += len(pre_ret)
            all_lab_num += len(lab_ret)
            for class_name, lx, ly, rx, ry, ang, prob in pre_ret:
                pre_one = np.array([(rx + lx) / 2, (ry + ly) / 2, rx - lx, ry - ly, ang])
                print('pre_one', pre_one)
                for cx, cy, w, h, ang_l in lab_ret:
                    lab_one = np.array([cx, cy, w, h, ang_l])
                    iou = iou_rotate_calculate(pre_one, lab_one)
                    ang_err = abs(ang - ang_l) / 180
                    if iou > 0.5:
                        num += 1
                        miou += iou
                        mang += ang_err
                    else:
                        tn += 1
    return num / all_pre_num, num / all_lab_num

if __name__ == '__main__':
    model = DlaNet(34)
    device = torch.device('cuda')
    model.load_state_dict(torch.load(''))
    model.eval()
    model.cuda()

    p, r= pre_recall('imgs', device)
    F1 = (2 * p * r) / (p + r)
