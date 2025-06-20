import numpy as np
import cv2
import os
from PIL import Image
import skimage
from skimage import morphology


class IOU:
    def __init__(self, mask, gt, gt_buffer, is_buffer=False):
        self.mask = mask
        self.gt = gt
        self.gt_buffer = gt_buffer
        self.is_buffer = is_buffer
        self.TPP = 0
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

    def cal_iou(self, mylogs):
        self.TPP = np.sum(self.mask*self.gt_buffer)
        self.TP = np.sum(self.mask*self.gt)
        self.TN = np.sum((self.mask+self.gt) == 0)
        self.FP = np.sum((self.mask-self.gt) == 1)
        self.FN = np.sum((self.gt-self.mask) == 1)
        if self.is_buffer:
            self.iou = self.TPP/(self.TP+self.FP+self.FN)
        else:
            self.iou = self.TP/(self.TP+self.FP+self.FN)
        self.recall = self.TP/(self.TP+self.FN)
        self.precision = self.TP/(self.TP+self.FP)
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        self.accuracy = (self.TP + self.TN) / (self.TP+self.FN+self.TN+self.FP)
        print('Accuracy:', round(self.accuracy, 2), ',Precision:', round(self.precision, 2), ',Recall:', round(self.recall, 2),
              ',F1-Score:', round(self.f1, 2), ',IoU:', round(self.iou, 2), end='', file=mylogs)
        print('Accuracy:', round(self.accuracy, 2), ',Precision:', round(self.precision, 2), ',Recall:', round(self.recall, 2),
              ',F1-Score:', round(self.f1, 2), ',IoU:', round(self.iou, 2), end='')
        

def thin_image(mask_file):
    im = cv2.imread(mask_file, 0)
    im = im > 128
    selem = skimage.morphology.disk(2)
    im = skimage.morphology.binary_dilation(im, selem)
    im = skimage.morphology.thin(im)
    return im.astype(np.uint8) * 255


if __name__=="__main__":
    # initialize lists that store the performance measure values for all predicted images
    recall_list = []
    precision_list = []
    f1_list = []
    accuracy_list = []
    iou_list = []
    completeness_list = []

    is_gt_buffer = False # !!!!!
    print("!!!!! buffer: "+str(is_gt_buffer))

    # 修改路径指向你的数据位置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_root = os.path.join(base_dir, 'data/Shaoxing/test_satellite/')
    lab_root = os.path.join(base_dir, 'data/Shaoxing/test_label/')
    mask_root = os.path.join(base_dir, 'output/test_results/')

    log_dir = os.path.join(base_dir, 'BSNet/evaluate/logs/')
    os.makedirs(log_dir, exist_ok=True)
    log_name = "initial_seg_evaluation"
    mylogs = open(os.path.join(log_dir, log_name + '.log'), 'w')

    # 获取所有测试图片
    test_images = [f for f in os.listdir(img_root) if f.endswith('_sat.png')]
    
    for image_name in test_images:
        base_name = image_name[:-8]  # 移除 '_sat.png'
        img = os.path.join(img_root, image_name)
        lab = os.path.join(lab_root, base_name + '_osm.png')
        mask = os.path.join(mask_root, base_name + '_mask.png')
        
        if os.path.exists(img) and os.path.exists(lab) and os.path.exists(mask):
            print(f'Evaluating {image_name}', file=mylogs)
            print(f'Evaluating {image_name}')
            
            image = cv2.imread(img)
            prediction = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            truevalue = cv2.imread(lab, cv2.IMREAD_GRAYSCALE)
            
            if truevalue is None:
                print(f"Warning: Could not read label file: {lab}")
                continue
                
            dilated_kernel = np.ones((3,3))
            gt_buffer = cv2.dilate(truevalue, dilated_kernel)

            thin_gt = thin_image(lab)
            num_mask = np.sum(thin_gt[prediction > 128]) / 255
            num_gt = np.sum(thin_gt) / 255
            completeness = num_mask / (num_gt + 0.00001)
            if num_gt != 0:
                completeness_list.append(completeness)

            if np.sum(truevalue) < 10:
                continue
            image = np.ndarray.astype(image, dtype='uint8')
            prediction = np.array(prediction, np.float32) / 255.0
            truevalue = np.array(truevalue, np.float32) / 255.0
            truevalue[truevalue >= 0.5] = 1
            truevalue[truevalue <= 0.5] = 0
            gt_buffer = np.array(gt_buffer, np.float32) / 255.0
            gt_buffer[gt_buffer >= 0.5] = 1
            gt_buffer[gt_buffer <= 0.5] = 0

            # initialize result class that calculates and stores all evaluation measures
            print('test image ', base_name, file=mylogs)
            print('test image ', base_name)
            res = IOU(prediction, truevalue, gt_buffer, is_buffer=is_gt_buffer)
            res.cal_iou(mylogs)

            print(',Completeness:', round(completeness, 2), file=mylogs)
            print(',Completeness:', round(completeness, 2))

            # append to evaluation lists
            recall_list.append(res.recall)
            precision_list.append(res.precision)
            f1_list.append(res.f1)
            accuracy_list.append(res.accuracy)
            iou_list.append(res.iou)


    # print the results for the evaluation measures to the command line
    print('********************************', file=mylogs)
    print('Accuracy:',  round((sum(accuracy_list) / len(accuracy_list)), 2), file=mylogs)
    print('Precision:', round((sum(precision_list) / len(precision_list)), 2), file=mylogs)
    print('Recall:', round((sum(recall_list) / len(recall_list)), 2), file=mylogs)
    print('F1-Score:', round((sum(f1_list) / len(f1_list)), 2), file=mylogs)
    print('IoU:', round((sum(iou_list) / len(iou_list)), 2), file=mylogs)
    print('Completeness:', round((sum(completeness_list) / len(completeness_list)), 2), file=mylogs)
    print('********************************')
    print('Accuracy:',  round((sum(accuracy_list) / len(accuracy_list)), 2))
    print('Precision:', round((sum(precision_list) / len(precision_list)), 2))
    print('Recall:', round((sum(recall_list) / len(recall_list)), 2))
    print('F1-Score:', round((sum(f1_list) / len(f1_list)), 2))
    print('IoU:', round((sum(iou_list) / len(iou_list)), 2))
    print('Completeness:', round((sum(completeness_list) / len(completeness_list)), 2))

    mylogs.close()

