import queue
import cv2
import numpy as np


def find_way_from_maze(image: np.ndarray = np.array([])) -> np.ndarray:
    WHITE = tuple([255, 255, 255])
    BLACK = tuple([0, 0, 0])

    def get_num_from_color(pixel: tuple) -> int:
        num = [(bin(num)[2:]).rjust(8, "0") for num in pixel]
        return int(num[0]+num[1]+num[2], 2)


    def get_colors_from_num(num: int) -> np.ndarray:
        str_num = (bin(num)[2:]).rjust(24, "0")
        pixel = [str_num[0:8], str_num[8:16], str_num[16:24]]

        return np.array([int(pixel_color, 2) for pixel_color in pixel])

    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    
    res_image = image.copy()

    input = [image.shape[0]-1, image.shape[1]//2+5]
    output = [0,               image.shape[1]//2-7]

    queue = [[*input, 1]]

    while queue:
        x, y, value = queue.pop(0)

        pixel = tuple(res_image[x][y])
        
        if pixel == BLACK:
            continue
        elif pixel == WHITE:
            res_image[x][y] = get_colors_from_num(value)
        elif get_num_from_color(pixel) <= value:
            continue

        if x == output[0] and y == output[1]:
            break

        # print(x, y, res_image[x][y])

        if x > 0:
            queue.append([x-1, y, value+1])
        if y > 0:
            queue.append([x, y-1, value+1])
        if x<image.shape[0]-1:
            queue.append([x+1, y, value+1])
        if y<image.shape[1]-1:
            queue.append([x, y+1, value+1])        

    # backtrack
    queue = [output]

    x_arr, y_arr = [], []

    while queue:
        x, y = queue.pop()
        x_arr.append(x)
        y_arr.append(y)

        pixel = tuple(res_image[x][y])
        value = get_num_from_color(pixel)

        if [x, y] == input:
            break

        if x > 0:
            new_pixel = tuple(res_image[x-1][y])
            if new_pixel != WHITE and new_pixel != BLACK:
                new_value = get_num_from_color(new_pixel)
                if new_value < value:
                    new_x, new_y = x-1, y
        if y > 0:
            new_pixel = tuple(res_image[x][y-1])
            if new_pixel != WHITE and new_pixel != BLACK:
                new_value = get_num_from_color(new_pixel)
                if new_value < value:
                    new_x, new_y  = x, y-1
        if x<image.shape[0]-1:
            new_pixel = tuple(res_image[x+1][y])
            if new_pixel != WHITE and new_pixel != BLACK:
                new_value = get_num_from_color(new_pixel)
                if new_value < value:
                    new_x, new_y  = x+1, y

        if y<image.shape[1]-1:
            new_pixel = tuple(res_image[x][y+1])
            if new_pixel != WHITE and new_pixel != BLACK:
                new_value = get_num_from_color(new_pixel)
                if new_value < value:
                    new_x, new_y  = x, y+1

        queue.append([new_x, new_y])

    print(len(x_arr), len(y_arr))

    return np.array([x_arr, y_arr])
