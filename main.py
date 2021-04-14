import numpy as np
import itertools
from PIL import Image as img
import cv2 as cv
from scipy.fftpack import dct, idct
import utils


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


def conv_ycbcr_to_rgb(img_arr):
    x_form = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(x_form.T)
    return np.uint8(rgb)


# a method to implement Discrete Cosine Transform using dct method from scipy.fftpack library
def dct2d(yuv_img):
    return dct(dct(yuv_img.T, norm='ortho').T, norm='ortho')

# a method to implement Inverse Discrete Cosine Transform using dct method from scipy.fftpack library


def idct2d(yuv_img):
    return idct(idct(yuv_img.T, norm='ortho').T, norm='ortho')


def chunks(img_arr, size):
    """ Yield successive n-sized chunks from l """
    for i in range(0, len(img_arr), size):
        yield img_arr[i:i + size]


def jpeg_compression_logic(img_arr, num_coeffs=None):
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
                                                  range(0, width, 8))]

    # DCT transform every 8x8 block
    dct_blocks = [dct2d(img_block) for img_block in img_blocks]

    if num_coeffs is not None:
        # keep only the first K DCT coefficients of every block
        reduced_dct_coeffs = [utils.zig_zag(
            dct_block, num_coeffs) for dct_block in dct_blocks]
    else:
        # quantize all the DCT coefficients using the quantization matrix and the scaling factor
        reduced_dct_coeffs = [np.round(dct_block / (utils.jpeg_quantiz_matrix_2))
                              for dct_block in dct_blocks]

        # and get the original coefficients back
        reduced_dct_coeffs = [reduced_dct_coeff * (utils.jpeg_quantiz_matrix_2)
                              for reduced_dct_coeff in reduced_dct_coeffs]

    # IDCT of every block
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


# HERE I applied the JPEG Compression Algorithm
# 1 we imported the image first from the image folder
image = img.open('images/hozier.jpg')
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
rec_img_rgb = cv.cvtColor(rec_img, code=cv.COLOR_YCrCb2BGR)
#rec_img_rgb = conv_ycbcr_to_rgb(rec_img)
rec_img_rgb[rec_img_rgb < 0] = 0
rec_img_rgb[rec_img_rgb > 255] = 255
rec_img_rgb = np.uint8(rec_img_rgb)
final_image = img.fromarray(rec_img_rgb)
final_image.save('selena_gomez_result_image.jpg')
final_image.show()
