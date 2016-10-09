import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg
from PIL import Image
import theano
import theano.tensor as T


#'''
#Implement the functions that were not implemented and complete the
#parts of main according to the instructions in comments.
#'''

def reconstructed_image(D,c,num_coeffs,X_mean,im_num):
#    '''
#    This function reconstructs an image given the number of
#    coefficients for each image specified by num_coeffs
#    '''
#    
#    '''
#        Parameters
#    ---------------
#    c: np.ndarray
#        a n x m matrix  representing the coefficients of all the image blocks.
#        n represents the maximum dimension of the PCA space.
#        m is (number of images x n_blocks**2)
#
#    D: np.ndarray
#        an N x n matrix representing the basis vectors of the PCA space
#        N is the dimension of the original space (number of pixels in a block)
#
#    im_num: Integer
#        index of the image to visualize
#
#    X_mean: np.ndarray
#        a matrix representing the mean block.
#
#    num_coeffs: Integer
#        an integer that specifies the number of top components to be
#        considered while reconstructing
#    '''
    
    c_im = c[:num_coeffs,im_num]
    D_im = D[:,:num_coeffs]
    recon_data=np.dot(D_im,c_im)
    image_data=recon_data.reshape(256,256)+X_mean
    X_recon_img=Image.fromarray(image_data)
    return X_recon_img

def plot_reconstructions(D,c,num_coeff_array,X_mean,im_num):
#    '''
#    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
#    vectors from c
#
#    Parameters
#    ------------------------
#        num_coeff_array: Iterable
#           an iterable with 9 elements representing the number_of coefficients
#            to use for reconstruction for each of the 9 plots
#        
#        c: np.ndarray
#            a l x m matrix  representing the coefficients of all blocks in a particular image
#            l represents the dimension of the PCA space used for reconstruction
#            m represents the number of blocks in an image
#
#        D: np.ndarray
#            an N x l matrix representing l basis vectors of the PCA space
#            N is the dimension of the original space (number of pixels in a block)
#
#        X_mean: basis vectors represent the divergence from the mean so this
#            matrix should be added to all reconstructed blocks
#
#        im_num: Integer
#            index of the image to visualize
#    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,im_num))
            
    f.savefig('output/hw1b_{0}.png'.format(im_num))
    plt.close(f)
    
    
    
def plot_top_16(D, sz, imname):
#    '''
#    Plots the top 16 components from the basis matrix D.
#    Each basis vector represents an image block of shape (sz, sz)
#
#    Parameters
#    -------------
#    D: np.ndarray
#        N x n matrix representing the basis vectors of the PCA space
#        N is the dimension of the original space (number of pixels in a block)
#        n represents the maximum dimension of the PCA space (assumed to be atleast 16)
#
#    sz: Integer
#        The height and width of each block
#
#    imname: string
#        name of file where image will be saved.
#    '''
    f,axxr=plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            D_comp=D[:,i*4+j].reshape((sz,sz))
            D_comp=255*(D_comp-np.min(D_comp))/(np.max(D_comp)-np.min(D_comp))
            D_img=Image.fromarray(D_comp)
            plt.axes(axxr[i,j])
            plt.imshow(D_img)
    
    f.savefig(imname)
    plt.close(f)

    
def main():

#   Read here all images(grayscale) from Fei_256 folder and collapse 
#   each image to get an numpy array Ims with size (no_images, height*width).
#   Make sure the images are read after sorting the filenames
 
    filepath = 'Fei_256/'
    names = sorted(os.listdir(filepath))
    f=mpl.pyplot.imread(filepath+names[0]).reshape((1,-1))
    Ims=np.array(f,dtype=np.uint8)
    for i in range(1,len(names)):
        f=mpl.pyplot.imread(filepath+names[i]).reshape((1,-1)) 
        Ims=np.append(Ims,f,axis=0)
    
    Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

#    Use theano to perform gradient descent to get top 16 PCA components of X
#    Put them into a matrix D with decreasing order of eigenvalues
#
#    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
#    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
#    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2

    epsilon=1.e-05
    learn_rate=0.005
    count=0
    stop_iteration=2000
    D=np.zeros((65536,16),dtype=np.float32)
    L=np.zeros((16,16),dtype=np.float32)
# Define theano symbolic variables
    d1=T.fvector('arg1')
    XX=T.fmatrix('arg2')
    XD=T.fmatrix('arg3')
    XL=T.fmatrix('arg4')

    mat_vec1=T.dot(XX,d1)
    cost=-T.dot(mat_vec1.T,mat_vec1)
    temp1=T.dot(d1.T,XD)
    temp2=T.dot(temp1,XL)
    temp3=T.dot(temp2,temp1.T)

    grad=theano.gradient.grad(cost,d1)
    gradp1=theano.gradient.grad(-cost,d1)
    gradp2=theano.gradient.grad(temp3,d1)
# Define theano symbolic functions
    calc_grad=theano.function([d1,XX],grad)
    calc_gradp1=theano.function([d1,XX],gradp1)
    calc_gradp2=theano.function([XD,XL,d1],gradp2)
    calc_L=theano.function([d1,XX],-cost)
# Initialization
    old_d=np.random.randn(65536).astype(np.float32)
    old_d=old_d/linalg.norm(old_d)
    dervtv=calc_grad(old_d,X)
    dervtv=dervtv/linalg.norm(dervtv)
# Calculating the first component
    while count<stop_iteration:
        new_d=old_d-dervtv*learn_rate
        new_d=new_d/linalg.norm(new_d)
        if linalg.norm(new_d-old_d)<epsilon:
            break
        dervtv=calc_grad(new_d,X)
        dervtv=dervtv/linalg.norm(dervtv)
        old_d=new_d
        count=count+1
    L[0,0]=calc_L(new_d,X)
    D[:,0]=new_d
# Calculating the rest 15 components
    for i in range(1,16):
        old_d=np.random.randn(65536).astype(np.float32)
        old_d=old_d/linalg.norm(old_d)
        count=0
        known_d=D[:,:i]
        known_l=L[:i,:i]

        dervtv=-calc_gradp1(old_d,X)+calc_gradp2(known_d,known_l,old_d)
        dervtv=dervtv/linalg.norm(dervtv)

        while count<stop_iteration:
            new_d=old_d-dervtv*learn_rate
            new_d=new_d/linalg.norm(new_d)
            if linalg.norm(new_d-old_d)<epsilon:
                break
            dervtv=-calc_gradp1(old_d,X)+calc_gradp2(known_d,known_l,new_d)
            dervtv=dervtv/linalg.norm(dervtv)
            old_d=new_d
            count=count+1
        L[i,i]=calc_L(new_d,X)
        D[:,i]=new_d
    c=np.dot(D.T,X.T)
#====================================================================#        
    for i in range(0, 200, 10):
        plot_reconstructions(D=D, c=c, num_coeff_array=[1, 2, 4, 6, 8, 10, 12, 14, 16], X_mean=X_mn.reshape((256, 256)), im_num=i)

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')


if __name__ == '__main__':
    main()
    
    