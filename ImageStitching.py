import numpy as np
import cv2
import matplotlib.pyplot as plt


class Stitch(object):
    def __init__(self, img1, img2, re_size=(600, 400), is_img_array=False):
        if is_img_array:
            self.img1 = img1
            self.img2 = img2
        else:
            self.img1 = cv2.imread(img1, cv2.IMREAD_COLOR)
            self.img2 = cv2.imread(img2, cv2.IMREAD_COLOR)

        self.img1_resize = cv2.resize(self.img1, re_size)
        self.img2_resize = cv2.resize(self.img2, re_size)

        self.img1_gray = self.to_gray(self.img1_resize).astype(np.uint8)
        self.img2_gray = self.to_gray(self.img2_resize).astype(np.uint8)


    def to_gray(self, img):
        img = img.astype(np.float)
        gray = np.power(0.6 * img[:, :, 0], 2.2) + \
               np.power(1.5 * img[:, :, 1], 2.2) + \
               np.power(img[:, :, 2], 2.2)
        gray = np.power(gray / (1+1.5**2.2+0.6**2.2), 1/2.2)
        return gray

    def get_kp_des(self):
        sift = cv2.xfeatures2d.SIFT_create()
        # kp1 = orb.detect(self.img1_resize, None)
        kp1, des1 = sift.detectAndCompute(self.img1_resize, None)

        # img2 = cv2.drawKeypoints(self.img1_resize, kp1, None, color=(0, 255, 0), flags=0)
        # plt.imshow(img2)
        # plt.show()
        kp2, des2 = sift.detectAndCompute(self.img2_resize, None)

        # kp1 = np.float32([kp.pt for kp in kp1])
        # kp2 = np.float32([kp.pt for kp in kp2])
        return kp1, des1, kp2, des2

    def get_matched_points(self, kp1, des1, kp2, des2):
        matcher = cv2.BFMatcher()
        # Since des is a floating-point descriptor NORM_L2 is used
        knn_matches = matcher.knnMatch(des1, des2, k=2)
        # Filter matches using the Lowe's ratio test
        ratio_thresh = 0.7
        good_matches = []
        for m, n in knn_matches:
            # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        src_pts = np.array([kp1[m.queryIdx].pt for m in good_matches])  # 查询图像的特征描述子索引
        dst_pts = np.array([kp2[m.trainIdx].pt for m in good_matches])  # 训练(模板)图像的特征描述子索引
        return src_pts, dst_pts

    def find_H(self, pts_1, pts_2):
        # 获取图像1到图像2的投影映射矩阵, 尺寸为3 * 3
        # findHomography参考https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
        # 单应矩阵:https://www.cnblogs.com/wangguchangqing/p/8287585.html
        (H, status) = cv2.findHomography(pts_1, pts_2, cv2.RANSAC)  # 生成变换矩阵
        return H

    def h_stitch(self, img1, img2, H):
        # 透视变换
        h1, w1 = img1.shape[0:2]
        h2, w2 = img2.shape[0:2]
        shift = np.array([[1.0, 0, w1], [0, 1.0, 0], [0, 0, 1.0]])
        M = np.dot(shift, H)  # 获取左边图像到右边图像的投影映射关系
        dst1 = cv2.warpPerspective(img1, M, (w1+w2, max(h1, h2)))  # 透视变换，新图像可容纳完整的两幅图
        dst2 = np.zeros_like(dst1)
        dst2[0:h2, w1:w1 + w2] = img2
        mask1 = np.sum(dst1, axis=2) > 0
        mask2 = np.sum(dst2, axis=2) > 0

        return dst1, mask1, dst2, mask2

    def v_stitch(self, img1, img2, H):
        # 透视变换
        h1, w1 = img1.shape[0:2]
        h2, w2 = img2.shape[0:2]
        shift = np.array([[1.0, 0, 0], [0, 1.0, h1], [0, 0, 1.0]])
        M = np.dot(shift, H)  # 获取左边图像到右边图像的投影映射关系
        dst1 = cv2.warpPerspective(img1, M, (max(w1, w2), h1 + h2))  # 透视变换，新图像可容纳完整的两幅图
        dst2 = np.zeros_like(dst1)
        dst2[h1:h1+h2, 0:w1] = img2
        mask1 = np.sum(dst1, axis=2) > 0
        mask2 = np.sum(dst2, axis=2) > 0

        return dst1, mask1, dst2, mask2

    def __call__(self, pattern='horizontal'):
        kp1, des1, kp2, des2 = self.get_kp_des()
        pts_1, pts_2 = self.get_matched_points(kp1, des1, kp2, des2)
        H = self.find_H(pts_1, pts_2)
        if pattern == 'horizontal':
            dst1, mask1, dst2, mask2 = self.h_stitch(self.img1_resize, self.img2_resize, H)
            dst = self.h_optimize_seam(dst1, mask1, dst2, mask2)
        elif pattern == 'vertical':
            dst1, mask1, dst2, mask2 = self.v_stitch(self.img1_resize, self.img2_resize, H)
            dst = self.v_optimize_seam(dst1, mask1, dst2, mask2)
        else:
            raise Exception("pattern is wrong, please check it")
        return dst

    def h_optimize_seam(self, dst1, mask1, dst2, mask2):
        dst = dst1.copy()
        dst[mask2] = dst2[mask2]
        mask = mask1.astype(np.int) + mask2.astype(np.int)
        mask = mask == 2
        h, w = mask.shape
        for i in range(h):
            line_length = np.sum(mask[i])
            for j in range(w):
                if mask[i, j] == 1:
                    start_idx = j
                    for id in range(start_idx, start_idx+line_length):
                        alpha = (id - start_idx) / line_length
                        dst[i, id] = ((1-alpha) * dst1[i, id] + alpha * dst2[i, id]).astype(np.uint8)
                    break
        return dst

    def v_optimize_seam(self, dst1, mask1, dst2, mask2):
        dst = dst1.copy()
        dst[mask2] = dst2[mask2]
        mask = mask1.astype(np.int) + mask2.astype(np.int)
        mask = mask == 2
        h, w = mask.shape
        for j in range(w):
            line_length = np.sum(mask[:, j])
            for i in range(h):
                if mask[i, j] == 1:
                    start_idx = i
                    for id in range(start_idx, start_idx+line_length):
                        alpha = (id - start_idx) / line_length
                        dst[id, j] = ((1-alpha) * dst1[id, j] + alpha * dst2[id, j]).astype(np.uint8)
                    break
        return dst


if __name__ == '__main__':
    img1 = r'G:\Coding\ImagesStitching\lt.jpg'
    img2 = r'G:\Coding\ImagesStitching\rt.jpg'
    img3 = r'G:\Coding\ImagesStitching\lb.jpg'
    img4 = r'G:\Coding\ImagesStitching\rb.jpg'
    img_stitcher1 = Stitch(img1, img3, re_size=(800, 600))
    dst1 = img_stitcher1(pattern='vertical')
    img_stitcher2 = Stitch(img2, img4, re_size=(800, 600))
    dst2 = img_stitcher2(pattern='vertical')

    img_stitcher = Stitch(dst1, dst2, re_size=(800, 1200), is_img_array=True)
    dst = img_stitcher(pattern='horizontal')
    dst = cv2.resize(dst, (800, 600))
    dst_gray = img_stitcher.to_gray(dst)

    # cv2.imshow("1", dst1)
    # cv2.imshow("2", dst2)
    # cv2.waitKey()
    cv2.imwrite('result1_color.jpg', dst)
    cv2.imwrite('result1_gray.jpg', dst_gray)
