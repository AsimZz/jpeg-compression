import numpy as np

# Quantization Matrix No.1
jpeg_quantiz_matrix_1 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]])

# Quantization Matrix No.2
jpeg_quantiz_matrix_2 = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]])

# Quantization Matrix No.3
jpeg_quantiz_matrix_3 = np.array([
    [9, 9, 12, 24, 50, 50, 50, 50],
    [9, 11, 13, 33, 50, 50, 50, 50],
    [12, 13, 28, 50, 50, 50, 50, 50],
    [24, 33, 50, 50, 50, 50, 50, 50],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [50, 50, 50, 50, 50, 50, 50, 50]])

# Quantization Matrix No.4
jpeg_quantiz_matrix_4 = np.array([
    [16, 17, 18, 19, 20, 21, 22, 23],
    [17, 18, 19, 20, 21, 22, 23, 24],
    [18, 19, 20, 21, 22, 23, 24, 25],
    [19, 20, 21, 22, 23, 24, 25, 26],
    [20, 21, 22, 23, 24, 25, 26, 27],
    [21, 22, 23, 24, 25, 26, 27, 28],
    [22, 23, 24, 25, 26, 27, 28, 29],
    [23, 24, 25, 26, 27, 28, 29, 30],
])


def zig_zag(array, n=None):
    """
    Return a new array where only the first n subelements in zig-zag order are kept.
    The remaining elements are set to 0.
    :param array: 2D array_like
    :param n: Keep up to n subelements. Default: all subelements
    :return: The new reduced array.
    """

    shape = np.array(array).shape

    assert len(shape) >= 2, "Array must be a 2D array_like"
    if n == None:
        n = shape[0] * shape[1]
    assert 0 <= n <= shape[0] * \
        shape[1], 'n must be the number of subelements to return'

    res = np.zeros_like(array)

    (j, i) = (0, 0)
    direction = 'r'  # {'r': right, 'd': down, 'ur': up-right, 'dl': down-left}

    for _ in range(1, n + 1):
        res[j][i] = array[j][i]
        if direction == 'r':
            i += 1
            if j == shape[0] - 1:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'dl':
            i -= 1
            j += 1
            if j == shape[0] - 1:
                direction = 'r'
            elif i == 0:
                direction = 'd'
        elif direction == 'd':
            j += 1
            if i == 0:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'ur':
            i += 1
            j -= 1
            if i == shape[1] - 1:
                direction = 'd'
            elif j == 0:
                direction = 'r'

    return res
