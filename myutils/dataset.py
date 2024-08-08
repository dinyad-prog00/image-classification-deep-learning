import h5py
import fidle
import os,ast
import matplotlib.pyplot as plt
from skimage import color, exposure, transform
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import rank
import numpy as np
from tensorflow import keras



def plot_statistics(train_xxx,test_xxx,name,xlim=[0,100]):
    plt.figure(figsize=(16,6))
    plt.hist([train_xxx,test_xxx], bins=100)
    plt.gca().set(title='{} - Train=[{:5.2f}, {:5.2f}]'.format(name,min(train_xxx),max(train_xxx)), 
              ylabel='Population', xlim=xlim)
    plt.legend(['Train','Test'])
    plt.show()

def labels_to_class_name(labels_object, labels):
    return [labels_object[i] for i in labels]


def images_enhancement(images, width=25, height=25, mode='RGB'):
    '''
    Resize and convert images - doesn't change originals.
    input images must be RGBA or RGB.
    Note : all outputs are fixed size numpy array of float64
    args:
        images :         images list
        width,height :   new images size (25,25)
        mode :           RGB | RGB-HE | L | L-HE | L-LHE | L-CLAHE
    return:
        numpy array of enhanced images
    '''
    modes = { 'RGB':3, 'RGB-HE':3, 'L':1, 'L-HE':1, 'L-LHE':1, 'L-CLAHE':1}
    lz=modes[mode]
    
    out=[]
    for img in images:
        
        # ---- if RGBA, convert to RGB
        if img.shape[2]==4:
            img=color.rgba2rgb(img)
            
        # ---- Resize
        img = transform.resize(img, (width,height))

        # ---- RGB / Histogram Equalization
        if mode=='RGB-HE':
            hsv = color.rgb2hsv(img.reshape(width,height,3))
            hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
            img = color.hsv2rgb(hsv)
        
        # ---- Grayscale
        if mode=='L':
            img=color.rgb2gray(img)
            
        # ---- Grayscale / Histogram Equalization
        if mode=='L-HE':
            img=color.rgb2gray(img)
            img=exposure.equalize_hist(img)
            
        # ---- Grayscale / Local Histogram Equalization
        if mode=='L-LHE':        
            img=color.rgb2gray(img)
            img = img_as_ubyte(img)
            img=rank.equalize(img, disk(10))/255.
        
        # ---- Grayscale / Contrast Limited Adaptive Histogram Equalization (CLAHE)
        if mode=='L-CLAHE':
            img=color.rgb2gray(img)
            img=exposure.equalize_adapthist(img)
            
        # ---- Add image in list of list
        out.append(img)
        fidle.utils.update_progress('Enhancement: ',len(out),len(images))
        

    # ---- Reshape images
    #     (-1, width,height,1) for L
    #     (-1, width,height,3) for RGB
    #
    out = np.array(out,dtype='float64')
    out = out.reshape(-1,width,height,lz)
    return out

def save_h5_dataset(x_train, y_train, x_test, y_test, labels ,filename):
        
    # ---- Create h5 file
    with h5py.File(filename, "w") as f:
        f.create_dataset("x_train", data=x_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("x_test",  data=x_test)
        f.create_dataset("y_test",  data=y_test)
        f.create_dataset("labels",data=str(labels))
        
        
    # ---- done
    size=os.path.getsize(filename)/(1024*1024)
    print('Dataset : {:24s}  shape : {:22s} size : {:6.1f} Mo   (saved)'.format(filename, str(x_train.shape),size))


def read_dataset(enhanced_dir, dataset_name):
    '''Reads h5 dataset
    Args:
        enhanced_dir     : enhanced data folder
        dataset_name : dataset name, without .h5
    Returns:    x_train,y_train, x_test,y_test, labels '''
    # ---- Read dataset
    filename = f'{enhanced_dir}/{dataset_name}.h5'
    with  h5py.File(filename,'r') as f:
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_test  = f['x_test'][:]
        y_test  = f['y_test'][:]
        labels  = ast.literal_eval(f['labels'][()].decode('utf-8'))
        
    x_train,y_train=fidle.utils.shuffle_np_dataset(x_train,y_train)
   
    return x_train,y_train, x_test,y_test,labels

def dataset_size(enhanced_dir, dataset_name):
    filename = f'{enhanced_dir}/{dataset_name}.h5'
    return os.path.getsize(filename)/(1024*1024)
    

# Data generators for data augmentation
# Generate batches of tensor image data with real-time data augmentation.
def get_data_generator_v1():
    return keras.preprocessing.image.ImageDataGenerator(
                                featurewise_center=False,
                             featurewise_std_normalization=False,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10
                             )

def get_data_generator_v2():
    return keras.preprocessing.image.ImageDataGenerator(
                                featurewise_center=False,
                             featurewise_std_normalization=False,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.4,
                             shear_range=0.1,
                             rotation_range=5
                             )