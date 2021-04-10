import numpy as np
import itertools
from cv2 import cv2 as cv
from PIL import Image as img
from scipy.fftpack import dct, idct


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


# converting RGB image array to YUV

def conv_rgb_to_ycbcr(img_arr):
    x_form = np.array(
        [[.299, .587, .114], [-.1687, -.3313, 0.5], [0.5, -.4187, -.0813]])
    ycbcr = img_arr.dot(x_form.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)

# applying chroma subsampling to our YCbCr image


def chroma_subsampling(img_arr):
    res_arr = img_arr.copy()
    res_arr[1::2, :, (1, 2)] = res_arr[::2, :, (1, 2)]
    # Vertically, every second element equals to element above itself.
    res_arr[:, 1::2, (1, 2)] = res_arr[:, ::2, (1, 2)]
    # Horizontally, every second element equals to the element on its left side.
    return res_arr


# a method to Discrete Cosine Transform using dct method from scipy.fftpack library
def dct2d(yuv_img):
    return dct(dct(yuv_img.T, norm='ortho').T, norm='ortho')


def chunks(img_arr, size):
    """ Yield successive n-sized chunks from l """
    for i in range(0, len(img_arr), size):
        yield img_arr[i:i + size]


def jpeg_compression_logic(img_arr):
    # prevent against multiple-channel images
    if len(img_arr.shape) != 2:
        raise ValueError('Input image must be a single channel 2D array')

    # shape of image
    height = img_arr.shape[0]
    width = img_arr.shape[1]
    if (height % 8 != 0) or (width % 8 != 0):
        raise ValueError(
            "Image dimensions (%s, %s) must be multiple of 8" % (height, width))

    # split into 8 x 8 pixels blocks
    img_blocks = [img_arr[j:j + 8, i:i + 8]
                  for (j, i) in itertools.product(range(0, height, 8),
                                                  range(0, width, 8))]  # DCT transform every 8x8 block

    dct_blocks = [dct2d(img_block) for img_block in img_blocks]

    # quantize all the DCT coefficients using the quantization matrix and the scaling factor
    reduced_dct_coeffs = [np.round(dct_block / (jpeg_quantiz_matrix_4))
                          for dct_block in dct_blocks]
    # and get the original coefficients back
    reduced_dct_coeffs = [reduced_dct_coeff * (jpeg_quantiz_matrix_4)
                          for reduced_dct_coeff in reduced_dct_coeffs]

    # applying the IDCT of every block
    rec_img_blocks = [idct2d(coeff_block)
                      for coeff_block in reduced_dct_coeffs]

    # reshape the reconstructed image blocks
    rec_img = []
    for chunk_row_blocks in chunks(rec_img_blocks, width // 8):
        for row_block_num in range(8):
            for block in chunk_row_blocks:
                rec_img.extend(block[row_block_num])
    rec_img = np.array(rec_img).reshape(height, width)

    return rec_img


def idct2d(yuv_img):
    return idct(idct(yuv_img.T, norm='ortho').T, norm='ortho')


def conv_ycbcr_to_rgb(img_arr):
    x_form = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(x_form.T)
    return np.uint8(rgb)


# HERE I applied the JPEG Compression Algorithm

# 1 we imported the image first from the image folder
image = img.open('images/selena_gomez_photo.jpg')
# 2 then I convert the image to YCbCr color model
img_yuv = conv_rgb_to_ycbcr(np.array(image))
# 3 now we implement the chroma subsampling technique
im = chroma_subsampling(img_yuv)
rec_img = np.empty_like(im)

'''
# 4 then here we injected our image array into the 'jpeg_compression_logic' method which divide the image into 8x8 blocks
    ,after that we apply the dct function from scipy libaray(I didn't find an appropriate code for the dct using matrix multiplication so I used scipy libaray)
    ,then I did the quantization the dct matrix coefficents using different quantization matrices No.2 was the best one,finally I reconstruct the image from its blocks
'''
for channel_num in range(3):
    mono_image = jpeg_compression_logic(im[:, :, channel_num],
                                        )
    print(mono_image.shape)
    rec_img[:, :, channel_num] = mono_image

# 5 finally I just convert the image back to RGB and display it
rec_img_rgb = conv_ycbcr_to_rgb(rec_img)
rec_img_rgb[rec_img_rgb < 0] = 0
rec_img_rgb[rec_img_rgb > 255] = 255
rec_img_rgb = np.uint8(rec_img_rgb)
final_image = img.fromarray(rec_img_rgb)
final_image.save('selena_gomez_result_image.jpg')
final_image.show()
