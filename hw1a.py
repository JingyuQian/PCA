import os
from os import walk
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg

from PIL import Image

import theano
import theano.tensor as T

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''


def reconstructed_image(D, c, num_coeffs, X_mean, n_blocks, im_num):
#    '''
#    This function reconstructs an image X_recon_img given the number of
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
#        
#
#    n_blocks: Integer
#        number of blocks comprising the image in each direction.
#        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
#    '''
    sz=256/n_blocks
    c_im = c[:num_coeffs, n_blocks * n_blocks * im_num:n_blocks * n_blocks * (im_num + 1)]
    D_im = D[:, :num_coeffs]
    recon_data=np.zeros((256,256),dtype=np.float32)
    for i in range(256/sz):
        for j in range(256/sz):
            block=np.dot(D_im,c_im[:,i*(256/sz)+j]).reshape(sz,sz)+X_mean
            recon_data[i*sz:(i+1)*sz,j*sz:(j+1)*sz]=block
    X_recon_img=Image.fromarray(recon_data)
    return X_recon_img


def plot_reconstructions(D, c, num_coeff_array, X_mean, n_blocks, im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i, j])
            plt.imshow(reconstructed_image(D, c, num_coeff_array[i * 3 + j], X_mean, n_blocks, im_num))

    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    count=0
    f,axxr=plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            D_comp=D[:,i].reshape((sz,sz))
            D_comp=255*(D_comp-np.min(D_comp))/(np.max(D_comp)-np.min(D_comp))
            D_img=Image.fromarray(D_comp)
            plt.axes(axxr[i,j])
            plt.imshow(D_img)
            count=count+1
    f.savefig(imname)
    plt.close(f)
    
#    raise NotImplementedError


def main():

    filepath = 'Fei_256/'
    names = sorted(os.listdir(filepath))
    f=mpl.pyplot.imread(filepath+names[0]).reshape((1,256,256))
    Ims=np.array(f,dtype=np.uint8)
    for i in range(1,len(names)):
        f=mpl.pyplot.imread(filepath+names[i]).reshape((1,256,256)) 
        Ims=np.append(Ims,f,axis=0)

#########################################
    szs = [8, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

#########################################
    no_images=Ims.shape[0]
    for sz, nc in zip(szs, num_coeffs):
#      '''
#      Divide here each image into non-overlapping blocks of shape (sz, sz).
#      Flatten each block and arrange all the blocks in a
#       (no_images*n_blocks_in_image) x (sz*sz) matrix called X
#        '''
        n_blocks_in_image=256*256/(sz*sz)
        X=np.zeros((no_images*n_blocks_in_image,sz*sz),dtype=np.uint8)
        index=0
        for i in range(0,no_images):
            for j in range(0,Ims.shape[1]/sz):
                for k in range(0,Ims.shape[2]/sz):
                    temp=Ims[i,sz*j:sz*(j+1),sz*k:sz*(k+1)].reshape((1,sz*sz))
                    X[index,:]=temp
                    index=index+1
#########################################

        X_mean = np.mean(X, 0)
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)

#        '''
#        Perform eigendecomposition on X^T X and arrange the eigenvectors
#        in decreasing order of eigenvalues into a matrix D
#        '''
        egval,egvec=linalg.eigh(np.dot(X.T,X))
        D=np.zeros((sz*sz,sz*sz),dtype=np.float32)
        for i in range(sz*sz):
            D[:,i]=egvec[:,sz*sz-1-i]
        c = np.dot(D.T, X.T)

#        for i in range(0, 200, 10):
#            plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean.reshape((sz, sz)), n_blocks=int(256 / sz),
#                                 im_num=i)

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()
