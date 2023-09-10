import cv2
import numpy as np


def detect_green_circles(image_path):          #高射炮
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义绿色范围的下限和上限
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])

    # 创建掩码，只保留绿色范围内的像素
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # 执行形态学操作以去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 查找图像中的圆圈
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        center_points = []
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            center_points.append(center)

        return center_points
    else:
        return []

'''
# 调用函数并获取中心点
image_path = 'your_image.jpg'  # 替换为您的图像路径
centers = detect_green_circles(image_path)

# 打印中心点坐标
for center in centers:
    print("Center Point:", center)
'''
#############################################
#导弹防空
import cv2
import numpy as np

def find_green_square_center(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建一个绿色的HSV范围
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # 根据绿色范围创建掩码
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # 执行形态学操作以清除噪音
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 存储矩形中心点坐标的列表
    center_points = []

    # 遍历轮廓并绘制矩形
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # 如果轮廓近似为四边形
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - h) < 10:  # 确保宽高差异不超过10像素
                center_x = x + w // 2
                center_y = y + h // 2
                center_points.append((center_x, center_y))

    return center_points
'''
# 使用示例
image_path = 'your_image.jpg'
centers = find_green_square_center(image_path)
print("矩形中心点坐标：", centers)
'''

