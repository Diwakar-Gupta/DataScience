import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image


def load_image(image_file):
	img = Image.open(image_file)
	return img

def scale_image(image):
    image = (image-image.min())
    image = image/image.max()
    return image

def seperator():
    return st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:white;" /> """, unsafe_allow_html=True)

def activate(image_filter, activation='Relu'):
    if activation == 'Relu':
        return tf.nn.relu(image_filter)
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(image_filter)

def numpy_to_image(image):
    image = image[0].numpy()
    image = scale_image(image)
    return image

def pool(image_detect, pool_type):
    return tf.nn.max_pool2d(image_detect, ksize=(5,5), strides=(1,1), padding = 'VALID')

def get_kernel(name='sobal', channel=3):
    if name=='sobal':
        ar = np.array([
          [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]],
          
          [[-1,-2,-1],
          [0,0,0],
          [1,2,1],],

          [[0, 1, 2],
          [-1, 0, 1],
          [-2, -1, 0],]
        ])
        if channel == 1:
            return ar[0].reshape(3,3,1,1)
        else:
            return ar.reshape(3,3,3,1)

getter, image_view = st.columns(2)

image_file = getter.file_uploader('Select Image')

def run(image, channel):

    select_kernel, select_activation, select_pool = st.columns(3)
    view_kernel, view_activation, view_pool = st.columns(3)

    select_kernel, view_kernel = st.columns(2)
    select_activation, view_activation = st.columns(2)
    select_pool, view_pool = st.columns(2)
    
    seperator()
    getter, image_view = st.columns(2)
    kernel = select_kernel.selectbox('Kernel', ['sobal'])

    kernel = get_kernel(kernel, channel)
    image_filter = tf.nn.conv2d(
        input=image,
        filters=kernel,
        strides=1,
        padding='SAME',
    )
    image = numpy_to_image(image_filter)
    view_kernel.image(image, width=300)
    
    seperator()
    activation = select_activation.selectbox('Activation', ['Relu', 'sigmoid'])
    # if activation == 'ReLu':
    image_detect = activate(image_filter, activation)
    image = numpy_to_image(image_detect)
    view_activation.image(image, width=300)

    seperator()
    pool_type = select_pool.selectbox('Pool Type', ['Max'])
    img_conv = pool(image_detect, pool_type)
    image = numpy_to_image(img_conv)
    view_pool.image(image, width=300)


if image_file is not None:
    image = load_image(image_file)
    image_view.image(image,width=250)
    
    image = np.asarray(image, dtype=np.float32)
    if image.ndim < 3:
        image = image.reshape([*image.shape, 1])
    _, _, channel = image.shape
    run(image.reshape(1, *image.shape), channel)

