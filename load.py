import os
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image


def getMinibatch(batch_idx, batch_size,
                  # ## PATH need to be fixed
                  mscoco="../../datasets/mscoco/inpainting/", split="train2014"):
    '''
    Show an example of how to read the dataset
    '''

    data_path = os.path.join(mscoco, split)
    
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    batch_input = []
    batch_target = []

    for i, img_path in enumerate(batch_imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        # ## Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
            
            batch_input.append(np.array(input).transpose(2,0,1).reshape(3,64,64))
            batch_target.append(np.array(target).transpose(2,0,1).reshape(3,32,32))

    return np.array(batch_input,dtype='float32')/255, np.array(batch_target,dtype='float32')/255

def loadDataset(mscoco="../../datasets/mscoco/inpainting/"):
    train_size = datasetSize(mscoco, split="train2014")
    valid_size = datasetSize(mscoco, split="val2014")

    X_train, y_train = getMinibatch(0,train_size, split='train2014')
    X_val, y_val = getMinibatch(0,valid_size, split='val2014')

    return X_train, y_train, X_val, y_val 

def datasetSize(mscoco="../../datasets/mscoco/inpainting/", split="train2014"):
    data_path = os.path.join(mscoco, split)
    return len(glob.glob(data_path + "/*.jpg"))

# inputs and target are (batch, channel, height, width)
def rebuildImage(outerImages, innerImages):
    outerImages = outerImages*255
    innerImages = innerImages*255
    outerImages = outerImages.transpose(0,2,3,1).reshape(len(outerImages),64,64,3)
    innerImages = innerImages.transpose(0,2,3,1).reshape(len(innerImages),32,32,3)
    
    batch_full_img = []

    for i, img in enumerate(outerImages):
        center = (int(np.floor(outerImages.shape[1] / 2.)), int(np.floor(outerImages.shape[2] / 2.)))

        full_img = np.copy(img)
        full_img[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = innerImages[i, :,:,:]

        batch_full_img.append(full_img)

    return np.array(batch_full_img)

def saveImages(images, path='../results'):
    for i, img in enumerate(images):
        save_dir = path + '/result_'+str(i)+'.jpg'

        img = Image.fromarray(img.astype('uint8'))
        img.save(save_dir)








