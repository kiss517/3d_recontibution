import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mayavi import mlab

# 定义棋盘格的尺寸
chessboard_size = (11, 8)

# 准备棋盘格角点的3D坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)*5

# 存储角点
obj_points = []  # 3D 世界坐标
img_points_left = []  # 左相机的2D图像坐标
img_points_right = []  # 右相机的2D图像坐标

# 读取左右相机的图片
left_images = glob.glob('l/*.jpg')
right_images = glob.glob('r/*.jpg')

# 遍历所有图像，查找棋盘格角点
for left_img, right_img in zip(left_images, right_images):
    img_left = cv2.imread(left_img)
    img_right = cv2.imread(right_img)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 查找左图和右图的棋盘格角点
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

    if ret_left and ret_right:
        obj_points.append(objp)
        img_points_left.append(corners_left)
        img_points_right.append(corners_right)

# 相机内参标定（针对每个相机单独进行）
ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    obj_points, img_points_left, gray_left.shape[::-1], None, None)
ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    obj_points, img_points_right, gray_right.shape[::-1], None, None)

# 双目标定，获取外参（旋转矩阵和位移向量）
ret, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left, img_points_right, 
    K_left, dist_left, K_right, dist_right, 
    gray_left.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    flags=cv2.CALIB_FIX_INTRINSIC
)

# 执行立体校正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    K_left, dist_left, K_right, dist_right,
    gray_left.shape[::-1], R, T, alpha=0
)

# 1. 加载左右相机的图像
left_image = cv2.imread('left.jpg', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right.jpg', cv2.IMREAD_GRAYSCALE)


height, width = left_image.shape[:2]
imageSize = (width, height)  # 替换为你的图像宽度和高度
map1x, map1y = cv2.initUndistortRectifyMap(K_left, dist_left, R1, P1, imageSize, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K_right, dist_right, R2, P2, imageSize, cv2.CV_32FC1)
rectified_left = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

min_disparity = 700
num_disparities = 128
block_size = 5  # Block size for the stereo matcher

# Create the StereoSGBM object
stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                               numDisparities=num_disparities,
                               blockSize=block_size,
                               uniquenessRatio=20)

disparity_map = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0


# 绘制视差图
plt.figure()
plt.imshow(disparity_map, cmap='jet')
plt.colorbar()
plt.title("Disparity Map")
plt.show()


# 读取视差图和重投影矩阵 Q
points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

mask = (disparity_map > 700)  # Z坐标 > 0
points_3D_valid = points_3D[mask]



# 提取 X, Y, Z 坐标
X = points_3D_valid[:, 0]
Y = points_3D_valid[:, 1]
Z = points_3D_valid[:, 2]

# 使用 Mayavi 绘制点云
mlab.points3d(X, Y, Z, mode="point")

mlab.show()