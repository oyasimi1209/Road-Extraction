import cv2
import numpy as np
import scipy.ndimage
import skimage.morphology
import os

def good_feature_to_track(thin_mask, mask, out_name, save_path):
    """
     Apply the detector on the segmentation map to detect the road junctions as starting points for tracing.
    :param thin_mask: one-pixel width segmentation map
    :param mask: road segmentation map
    :param out_name: filename
    :param save_path: the directory of corner detection results
    :return:
    """
    # set a padding to avoid image edge corners
    padding_x = 128+5
    padding_y = 128
    #corners = cv2.goodFeaturesToTrack(thin_mask, 100, 0.1, 500)
    # 调整角点检测参数，降低质量阈值和最小距离
    corners = cv2.goodFeaturesToTrack(thin_mask, 100, 0.1, 500)
    
    # 如果没有检测到角点，返回
    if corners is None:
        print(f"No corners found in {out_name}")
        return
        
    corners = np.int0(corners)
    img = np.zeros((mask.shape[0], mask.shape[1], 3))
    img[:, :, 0] = mask
    img[:, :, 1] = mask
    img[:, :, 2] = mask
    corner_num = 0
    with open(os.path.join(save_path, out_name[:-4]+".txt"), "w") as f:
        for i in corners:
            x, y = i.ravel()
            if x < padding_x or x > img.shape[0]-padding_x:
                continue
            if y < padding_y or y > img.shape[1]-padding_y:
                continue

            f.write("{},{}\n".format(x,y))
            cv2.circle(img, (x, y), 20, (0, 0, 255), -1)
            corner_num += 1
    print("total corners number:{}".format(corner_num))
    cv2.imwrite(os.path.join(save_path, out_name[:-4]+'_with_corners.png'), img)


def thin_image(mask_dir, filename):
    """
    Skeletonize the road segmentation map to a one-pixel width
    :param mask_dir: the directory of road segmentation map
    :param filename: the filename of road segmentation map
    :return: one-pixel width segmentation map
    """
    file_path = os.path.join(mask_dir, filename)
    im = cv2.imread(file_path, 0)  # 使用cv2.imread替代scipy.ndimage.imread
    if im is None:
        raise ValueError(f"Could not read image: {file_path}")
    im = im > 128
    selem = skimage.morphology.disk(2)
    im = skimage.morphology.binary_dilation(im, selem)
    im = skimage.morphology.thin(im)
    return im.astype(np.uint8) * 255


if __name__ == "__main__":
    # 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mask_dir = os.path.join(current_dir, "out", "corner_detect", "seg_mask")
    txt_dir = os.path.join(current_dir, "out", "corner_detect", "corners")
    
    print(f"Looking for images in: {mask_dir}")
    
    # 确保目录存在
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    
    # 获取所有分割图像文件
    image_files = [f for f in os.listdir(mask_dir) if f.endswith('_binary.png')]
    print(f"Found {len(image_files)} images")
    
    for mask_filename in image_files:
        print(f"Processing {mask_filename}")
        try:
            thin_img = thin_image(mask_dir, mask_filename)
            mask = cv2.imread(os.path.join(mask_dir, mask_filename), 0)
            good_feature_to_track(thin_img, mask, mask_filename, txt_dir)
        except Exception as e:
            print(f"Error processing {mask_filename}: {str(e)}")
            continue