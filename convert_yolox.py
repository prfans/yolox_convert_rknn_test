import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
import torchvision
from rknn.api import RKNN
from get_im_list import *

import torch
from shapely.geometry import Polygon
import shapely

ONNX_MODEL = 'yolox.onnx'
RKNN_MODEL = 'yolox.rknn'
IMG_PATH = './1.bmp'
DATASET = './dataset.txt'

QUANTIZE_ON = True

BOX_THRESH = 0.25
NMS_THRESH = 0.4
IMG_SIZE = 320
num_classes = 4


CLASSES = ("c1", "c2", "c3", "c4 ")
COLORS = ((255,255,0), (0,0,255),(0,255,0),(255,0,0))


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    # [x y w h obj_conf cls... angle...]
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        # get angle and angle confidence
        # angle_conf, angle_pred = torch.max(image_pred[:, 5 + num_classes:], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        
        # Detections ordered as (cx, cy, w, h, obj_conf, class_conf, class_pred, angle)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        # print('detections: ', detections)
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def meshgrid(*tensors):
    #return torch.meshgrid(*tensors, indexing="ij")
    return torch.meshgrid(*tensors)
    # if _TORCH_VER >= [1, 10]:
    #    return torch.meshgrid(*tensors, indexing="ij")
    #else:
    #    return torch.meshgrid(*tensors)


def decode_outputs(hw, outputs, dtype):
    grids = []
    strides = []
    _strides = [8, 16, 32]
    for (hsize, wsize), stride in zip(hw, _strides):
        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    strides = torch.cat(strides, dim=1).type(dtype)

    outputs_xy, outputs_wh, outputs_obj, outputs_cls = torch.split(outputs,[2, 2, 1, 4], dim=-1)
    outputs_xy = (outputs_xy + grids) * strides
    outputs_wh = torch.exp(outputs_wh) * strides
    outputs = torch.cat([outputs_xy, outputs_wh, outputs_obj, outputs_cls], dim=-1)

    # outputs[..., :2] = (outputs[..., :2] + grids) * strides
    # outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    return outputs


def get_outputs(xin):
    outputs = []
    strides = [8, 16, 32]

    """
    for k, (stride_this_level, x) in enumerate(
        zip(strides, xin)
    ):
        # reg_output = x[:, 0:4, :, :]
        # obj_output = x[:, 4, :, :].unsqueeze(1)
        # cls_output = x[:, 5:5+self.num_classes, :]
        # angle_output = x[:, 5+self.num_classes:, :]
        # 不用用切片，否则onnx会出错，使用torch.split函数：
        reg_output, obj_output, cls_output, angle_output = torch.split(x, [4, 1,6,180], dim=1)
        #reg_output = x[0]
        #obj_output = x[1]
        #cls_output = x[2]
        #angle_output = x[3]

        output = torch.cat(
            [reg_output, obj_output.sigmoid(), cls_output.sigmoid(), angle_output.sigmoid()], 1
        )

        outputs.append(output)

    hw = [x.shape[-2:] for x in outputs]
    # [batch, n_anchors_all, 85]
    outputs = torch.cat(
        [x.flatten(start_dim=2) for x in outputs], dim=2
    ).permute(0, 2, 1)
    """

    hw = [[40,40],[20,20],[10,10]]
    outputs = xin

    return decode_outputs(hw, outputs, dtype=xin.type())

def vis(img, boxes, scores, cls_ids, conf=0.5):

    for i in range(len(boxes)):
        box = boxes[i]
        print(box)
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue

        color = COLORS[cls_id]
        text = '{}:{:.1f}%'.format(CLASSES[cls_id], score * 100)
        txt_color = COLORS[cls_id]
        font = cv2.FONT_HERSHEY_SIMPLEX

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        # cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        #"""
        txt_bk_color = (np.array(COLORS[cls_id])*0.3).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        #"""

    return img

def visual(img, output, cls_conf=0.35):
    ratio = 1.0 #img_info["ratio"]
    #img = img_info["raw_img"]
    if output is None:
        return img
    # output = output.cpu()

    bboxes = output[:, 0:4]

    # preprocessing: resize
    bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    vis_res = vis(img, bboxes, scores, cls, cls_conf)

    return vis_res

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    if not os.path.exists(RKNN_MODEL):
        # pre-process config
        print('--> Config model')
        rknn.config(reorder_channel='0 1 2',
                    #optimization_level=3,
                    target_platform = 'rk1126',
                    #output_optimize=1,
                    )
        print('done')

        # Load ONNX model
        print('--> Loading model')
        ret = rknn.load_onnx(model=ONNX_MODEL) #,outputs=['output'])
        if ret != 0:
            print('Load yolox failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True,  dataset='./dataset.txt')
        if ret != 0:
            print('Build yolox failed!')
            exit(ret)
        print('done')

        # Export RKNN model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(RKNN_MODEL)
        if ret != 0:
            print('Export yolox_rknn failed!')
            exit(ret)
        print('done')

    # 加载rknn文件
    ret = rknn.load_rknn(RKNN_MODEL)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk1808', device_id='1808')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    if not os.path.exists("./output"):
        os.mkdir("./output")

    imgs = get_im_list("./data", ".bmp")
    for path in imgs:
        # Set inputs
        img = cv2.imread(path)
        # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))

        # Inference
        print('--> Running model')
        outputs = rknn.inference(inputs=[img])

        print(len(outputs))
        backbone_outputs = []
        for i in range(len(outputs)):
            print(outputs[i].shape, type(outputs[i]))
            backbone_outputs.append(torch.from_numpy(outputs[i]))

        outputs = get_outputs(backbone_outputs[0])
        print("outputs: ", outputs.shape)

        print("post-process ...")
        #outputs = torch.from_numpy(outputs[0])
        outputs = postprocess(
            outputs, num_classes, BOX_THRESH,
            NMS_THRESH, class_agnostic=True
        )
        if outputs[0] is not None:
            print(outputs[0].shape)

        label = visual(img, outputs[0])

        base_name = os.path.basename(path)
        save_name = os.path.join("./output", base_name)
        cv2.imwrite(save_name, label)

    rknn.release()
