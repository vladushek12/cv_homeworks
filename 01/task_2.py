import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """
    road_number = 0


    ROAD_COLOR = np.array([213, 213, 213])
    # N = 10
    # step = image.shape[1] // N
    # print(step)

    left_bound, right_bound = None, None

    roads = []

    prev_color = None

    for i in range(image.shape[1]):
        
        new_color = image[image.shape[0]//2, i].copy().astype(dtype = np.int16)
        
        dist = np.linalg.norm(new_color - ROAD_COLOR)

        if dist < 40:
            if not left_bound:
                left_bound = i
            else:
                right_bound = i
        else:
            if left_bound and right_bound:
                roads.append([image.shape[0]//2, (right_bound - left_bound) // 2 + left_bound])
                left_bound, right_bound = None, None
       

    for road_number, road in enumerate(roads):
        for i in range(image.shape[0]//2):
            if (image[road[0]-i, road[1]].astype(dtype = np.int16) != ROAD_COLOR).all():
                break

        if i == image.shape[0] // 2 - 1:
            return road_number + 1

    return road_number