import cv2
import numpy as np


import cv2
import numpy as np

def new_borders(M, h: int, w: int):
    """
    Вычисляет новые границы изображения после применения аффинного преобразования.

    Args:
        M (np.ndarray): Матрица аффинного преобразования (2x3).
        h (int): Высота исходного изображения.
        w (int): Ширина исходного изображения.

    Returns:
        tuple: Кортеж, содержащий:
            - (float, float): Координаты нижнего левого угла (min_x, min_y) нового изображения.
            - (int, int): Новый размер изображения (new_w, new_h).
    """
    # Крайние точки исходного изображения
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
    transformed_corners = M @ corners.T

    # Получаем новые границы
    min_x = np.min(transformed_corners[0, :])
    max_x = np.max(transformed_corners[0, :])
    min_y = np.min(transformed_corners[1, :])
    max_y = np.max(transformed_corners[1, :])

    new_w = int(np.round(max_x - min_x))
    new_h = int(np.round(max_y - min_y))

    return (min_x, min_y), (new_w, new_h)


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    h, w, _ = image.shape

    M = cv2.getRotationMatrix2D(point, angle, scale=1.0)

    (low_coords, new_size) = new_borders(M, h, w)
    new_w, new_h = new_size

    M[0, 2] -= low_coords[0]
    M[1, 2] -= low_coords[1]

    dst_rotate = cv2.warpAffine(image, M, new_size)
    return dst_rotate


def apply_warpAffine(image, points1, points2):
    """
    Применяет перспективное преобразование к изображению.

    Args:
        image (np.ndarray): Исходное изображение, к которому будет применяться преобразование.
        points1 (list of tuple): Список координат точек в исходном изображении (4 точки).
        points2 (list of tuple): Список координат точек в целевом изображении (4 точки).

    Returns:
        np.ndarray: Преобразованное изображение.
    """
    # Получаем матрицу перспективного преобразования
    M = cv2.getPerspectiveTransform(np.float32(points1), np.float32(points2))

    # Определяем размер нового изображения
    width = int(max(points2[0][0], points2[1][0], points2[2][0], points2[3][0]))
    height = int(max(points2[0][1], points2[1][1], points2[2][1], points2[3][1]))

    # Применяем перспективное преобразование
    transformed_image = cv2.warpPerspective(image, M, (width, height))

    return transformed_image