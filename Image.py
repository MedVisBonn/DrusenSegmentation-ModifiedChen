
'''
Created on 2 May 2015

@author: shekoufeh
'''
import FED
import numpy as np
import scipy as sc
from skimage import io
from scipy import misc
from scipy.ndimage import filters
from math import sqrt, floor, ceil, exp

class Image(object):

    def __init__(self):
        self.image=[]
        self.w = 0
        self.h = 0
        self.xb = 0
        self.yb = 0

    def add_zero_to_left(self, num_str):
        while(len(num_str) < 3):
            num_str = "0" + num_str
        return num_str

    def read_3d_image_from_2d_slices_sery(self, file_path, file_name_prefix,\
            start_num, end_num, file_format, resize = None):
        interp='bicubic'
        r   = end_num - start_num + 1
        img = []
        for i in range(r):
            img_id    = str(start_num + i)
            img_id    = self.add_zero_to_left(img_id)
            file_name = file_path + file_name_prefix + img_id + "." +\
                        file_format
            img_slice = self.read_2d_image(file_name)
            if( resize != None ):
                img_slice = misc.imresize(img_slice, (resize[1], resize[2]),\
                    interp)
            img.append(img_slice)
        img = np.asarray(img)
        if( resize != None ):
            d, h, w = img.shape
            new_img = []
            for i in range(h):
                img_slice = img[:,i,:]
                img_slice = misc.imresize(img_slice, (resize[0], resize[2]),
                                          interp)
                new_img.append(img_slice)
            img = np.asarray(new_img)
        return img

    def save_3d_image_as_2d_slices(self, image, file_path, file_name_prefix,\
            file_format):
        slice_num = image.shape[0]
        img_min = image.min()
        img_max = image.max()

        if( len(image.shape) <= 3 ):
            for i in range(slice_num):
                img_id    = str(i)
                img_id    = self.add_zero_to_left(img_id)
                file_name = file_path + file_name_prefix + img_id + "." +\
                            file_format
                misc.toimage(image[i,:,:], cmax = img_max, cmin = img_min).\
                            save(file_name)
        else:
            for i in range(slice_num):
                img_id    = str(i)
                img_id    = self.add_zero_to_left(img_id)
                file_name = file_path + file_name_prefix + img_id + "." +\
                            file_format
                misc.toimage(image[i,:,:,:], cmax = img_max, cmin = img_min).\
                            save(file_name)
        return
    def save_2d_image(self, image, file_name):
        img_min = image.min()
        img_max = image.max()
        misc.toimage(image, cmax = img_max, cmin = img_min).\
                    save(file_name)

    def normalize(self, image):
        img_min = np.min(image)
        image   = image - img_min
        img_max = np.max(image)
        image   = image/float(img_max)
        return image

    def threshold_image(self, image, threshold):
        image[ image<threshold ] = 0.0

    def read_2d_image(self, file_name):
        image = io.imread(file_name)
        dim = image.shape
        if(len(dim)>2):
            image = np.dot(image[...,:3], [0.299, 0.587, 0.144])
        return image

    def read_image(self, file_name, num_channels=1):
        self.image = io.imread(file_name)
        dim = self.image.shape
#        print(dim)
        if( num_channels == 1 ):
            if(len(dim)>2):
                self.image = np.dot(self.image[...,:3], [0.299, 0.587, 0.144])
            self.h, self.w = self.image.shape
        
    def save_image(self, file_name, data):
        io.imsave(file_name, data)
   
    
        
        
    def mirror_boundary(self, image, xb, yb, zb = 0):
        dim = image.shape
        if len(dim) == 2 :
            h, w = dim
            #mirror in x direction
            x_dir_reversed = np.fliplr(image)
            res_image = np.concatenate((x_dir_reversed[:, w - xb:w], image,\
                        x_dir_reversed[:, 0:xb]), axis = 1)

            #mirror in y direction
            y_dir_reversed = np.flipud(res_image)
            res_image = np.concatenate((y_dir_reversed[h - yb:h,:], res_image,\
                        y_dir_reversed[0:yb,:]), axis = 0)

        elif len(dim) == 3 :
            d, h, w = dim

            #mirror in x direction
            # fliplr only flips in the row direction and here we have
            # multi-dim image, we swap x and y position to do the flip
            # in x-dir and then swap the axes back
            res_image = np.swapaxes(image, 2, 1)
            x_dir_l = np.fliplr(res_image[:,0:xb,:]  )
            x_dir_r = np.fliplr(res_image[:,w-xb:w,:])
            res_image = np.concatenate((x_dir_l, res_image, x_dir_r), axis = 1)
            res_image = np.swapaxes(res_image, 2, 1) # swap the axes back

            #mirror in y direction
            y_dir_l = np.fliplr(res_image[:,0:yb,:]  )
            y_dir_r = np.fliplr(res_image[:,h-yb:h,:])
            res_image = np.concatenate((y_dir_l, res_image, y_dir_r), axis = 1)

            #mirror in z direction
            z_dir_l = np.flipud(res_image[0:zb,:,:]  )
            z_dir_r = np.flipud(res_image[d-zb:d,:,:])
            res_image = np.concatenate((z_dir_l, res_image, z_dir_r), axis = 0)


        else:
            raise Exception("In function mirror_boundary(): Invalid image"\
                            " dimension. Images must be either 2D or 3D.")

        return res_image

    def mirror_boundary_pivot(self, image, xb, yb, zb = 0):
        dim = image.shape
        if len(dim) == 2 :
            h, w = dim
            #mirror in x direction
            x_dir_reversed = np.fliplr(image)
            res_image = np.concatenate((x_dir_reversed[:, w - xb:w-1],\
                        image, x_dir_reversed[:, 1:xb]), axis = 1)

            #mirror in y direction
            y_dir_reversed = np.flipud(res_image)
            res_image = np.concatenate((y_dir_reversed[h - yb:h-1,:],\
                        res_image, y_dir_reversed[1:yb,:]), axis = 0)

        elif len(dim) == 3 :
            d, h, w = dim
            #mirror in x direction
            # fliplr only flips in the row direction and here we have multi-dim
            # image, we swap x and y position to do the flip in x-dir and then
            # swap the axes back
            res_image = np.swapaxes(image, 2, 1)
            x_dir_l = np.fliplr(res_image[:,1:xb,:]  )
            x_dir_r = np.fliplr(res_image[:,w-xb:w-1,:])
            res_image = np.concatenate((x_dir_l, res_image, x_dir_r), axis = 1)
            res_image = np.swapaxes(res_image, 2, 1) # swap the axes back

            #mirror in y direction
            y_dir_l = np.fliplr(res_image[:,1:yb,:]  )
            y_dir_r = np.fliplr(res_image[:,h-yb:h-1,:])
            res_image = np.concatenate((y_dir_l, res_image, y_dir_r), axis = 1)

            #mirror in z direction
            z_dir_l = np.flipud(res_image[1:zb,:,:]  )
            z_dir_r = np.flipud(res_image[d-zb:d-1,:,:])
            res_image = np.concatenate((z_dir_l, res_image, z_dir_r), axis = 0)
        else:
            raise Exception("Error in function mirror_boundary_pivot():"\
                            " Invalid image dimension. Image must be 2D or"\
                            " 3D.")

        return res_image

    def mirror_boundary_dirichlet(self, image, xb, yb, bvalue):
        height, width = image.shape
        res = np.empty((xb*2 + width)*(yb*2 + height)).reshape(yb+height+yb,\
              xb+width+xb)
        res.fill(bvalue)
        res[yb:height+yb,xb:width+xb] = image
        return res

    def cut_boundry(self, image, xb, yb, zb = 0):
        dim  = image.shape
        if len(dim) == 2 :
            h, w = dim
            res = image[yb:h-yb, xb:w-xb]
        elif len(dim) == 3 :
            d, h, w = dim
            res = image[zb:d-zb, yb:h-yb, xb:w-xb]
        else:
            raise Exception("Error in function cut_boundary() : Invalid image"\
                            " size. Image must be 2D or 3D.")
        return res
   
    def contrast_enhancement(self, image, new_std, new_mean):
        old_mean = np.mean(image)
        old_std  = np.std(image)
        en_img = new_mean + ((new_std * (image - old_mean)) / old_std)
        return en_img

    def image_convolve(self, image, kernel, mode='mirror'):
        if( mode == 'constant'):
            return filters.convolve(image, kernel, mode = mode, cval = 0.0 )
        return filters.convolve(image, kernel, mode = mode )

    #axis = 0 means in y_direction
    #axis = 1 means in x_direction
    def image_convolve1d(self, image, kernel, axis = 1):
        return filters.convolve1d(image, kernel, axis, mode = 'mirror')

    def image_convolve_gaussian(self, image, sigma, mode = 'mirror'):
        return filters.gaussian_filter(image, sigma, order = 0, mode = mode)

    def image_second_derivative_old(self, image, dd = 'xx'):

        if dd == 'xx':
            image_der_dd = self.mirror_boundry(image, 3, 3)
            image_der_dd = np.diff(image_der_dd, n = 2, axis = 1)
            image_der_dd = self.cut_boundry(image_der_dd, 2, 3)

        elif dd == 'yy':
            image_der_dd = self.mirror_boundry(image, 3, 3)
            image_der_dd = np.diff(image_der_dd, n = 2, axis = 0)
            image_der_dd = self.cut_boundry(image_der_dd, 3, 2)

        elif (dd == 'xy' or dd == 'yx'):
            conv_kernel_xy = [
                        [ 1, 0,-1],
                        [ 0, 0, 0],
                        [-1, 0, 1]]
            conv_kernel_xy = np.multiply(conv_kernel_xy, 0.25)
            image_der_dd = self.image_convolve(image, conv_kernel_xy)
        else:
            print('Invalid derivation direction. In image_der_dd function')


        return image_der_dd
    def mirror_boundry_pivot(self, image, xb, yb):
        h, w = image.shape
        #mirror in x direction
        x_dir_reversed = np.fliplr(image)
        res_image = np.concatenate((x_dir_reversed[:, w - xb:w-1], image, x_dir_reversed[:, 1:xb]), axis = 1)

        #mirror in y direction
        y_dir_reversed = np.flipud(res_image)
        res_image = np.concatenate((y_dir_reversed[h - yb:h-1,:], res_image, y_dir_reversed[1:yb,:]), axis = 0)

        return res_image
    def image_der_d_old(self, image):

        image_der_d = self.mirror_boundry_pivot(image, 4, 4)

        kernel  =  np.asarray([1., 0., -1.])

        image_der_x = 0.5 * self.image_convolve1d(image_der_d, kernel, axis = 1)
        image_der_y = 0.5 * self.image_convolve1d(image_der_d, kernel, axis = 0)

        image_der_x = self.cut_boundry(image_der_x, 3, 3)
        image_der_y = self.cut_boundry(image_der_y, 3, 3)

        gradient = np.dstack((image_der_x, image_der_y))
        return gradient
    def mirror_boundry(self, image, xb, yb):
        h, w = image.shape
        #mirror in x direction
        x_dir_reversed = np.fliplr(image)
        res_image = np.concatenate((x_dir_reversed[:, w - xb:w], image, x_dir_reversed[:, 0:xb]), axis = 1)

        #mirror in y direction
        y_dir_reversed = np.flipud(res_image)
        res_image = np.concatenate((y_dir_reversed[h - yb:h,:], res_image, y_dir_reversed[0:yb,:]), axis = 0)

        return res_image
    #axis = 0 means in y_direction
    #axis = 1 means in x_direction
    #based on didas 2009 paper
    def image_der_dd_old(self, image, dd = 'xx'):
        if dd == 'xx':
            image_der_dd = self.mirror_boundry(image, 3, 3)
            image_der_dd = np.diff(image_der_dd, n = 2, axis = 1)
            image_der_dd = self.cut_boundry(image_der_dd, 2, 3)

        elif dd == 'yy':
            image_der_dd = self.mirror_boundry(image, 3, 3)
            image_der_dd = np.diff(image_der_dd, n = 2, axis = 0)
            image_der_dd = self.cut_boundry(image_der_dd, 3, 2)

        elif dd == 'xy':
            conv_kernel_xy = [
                        [0, -1, 1],
                        [-1, 2,-1],
                        [ 1,-1, 0]]
            conv_kernel_xy = np.multiply(conv_kernel_xy, 0.5)
            image_der_dd = self.image_convolve(image, conv_kernel_xy)

        elif dd == 'yx':
            conv_kernel_yx = [
                        [-1, 1,  0],
                        [ 1,-2,  1],
                        [ 0, 1, -1]]
            conv_kernel_yx = np.multiply(conv_kernel_yx, 0.5)
            image_der_dd = self.image_convolve(image, conv_kernel_yx)

        else:
            print('Invalid derivation direction. In image_der_dd function')

        return image_der_dd

    def image_der_d(self, image, hx = 1.0, hy = 1.0, hz = 1.0, mode='central'):

        dim = image.shape
        
        if len(dim) == 2 :
            h, w = dim
            image_der_d = self.mirror_boundary_pivot(image, 4, 4)
            if( mode == 'central'):
                kernel_x  =  np.asarray([ 1., 0., -1.])
                kernel_y  =  np.asarray([-1., 0.,  1.])
            elif(mode == 'forward'):
                kernel_x  =  np.asarray([-1., 1.])
                kernel_y  =  np.asarray([-1., 1.])
            elif(mode == 'backward'):
                kernel_x  =  np.asarray([1., -1.])
                kernel_y  =  np.asarray([1., -1.])
            else:
                raise Exception("Error in function image_der_d() : Invalid"\
                " defferentiation mode. mode={central, forward, backward}.")
            image_der_x = 0.5 * (1.0/hx) *\
                         self.image_convolve1d(image_der_d, kernel_x, axis = 1)
            image_der_y = 0.5 * (1.0/hy) *\
                         self.image_convolve1d(image_der_d, kernel_y, axis = 0)

            image_der_x = self.cut_boundry(image_der_x, 3, 3)
            image_der_y = self.cut_boundry(image_der_y, 3, 3)
              
            gradient = np.dstack((image_der_x, image_der_y))

        elif len(dim) == 3 :
            kernel_x  =  np.asarray([ 1., 0., -1.])
            kernel_y  =  np.asarray([-1., 0.,  1.])
            kernel_z  =  np.asarray([ 1., 0., -1.])

            image_der_d = self.mirror_boundary_pivot(image, 4, 4, 4)
            image_der_x = 0.5 * (1.0/hx) * self.image_convolve1d(image_der_d,\
                            kernel_x, axis = 2)
            image_der_y = 0.5 * (1.0/hy) * self.image_convolve1d(image_der_d,\
                            kernel_y, axis = 1)
            image_der_z = 0.5 * (1.0/hz) * self.image_convolve1d(image_der_d,\
                            kernel_z, axis = 0)

            image_der_x = self.cut_boundry(image_der_x, xb = 3, yb = 3, zb = 3)
            image_der_y = self.cut_boundry(image_der_y, xb = 3, yb = 3, zb = 3)
            image_der_z = self.cut_boundry(image_der_z, xb = 3, yb = 3, zb = 3)

            gradient = np.stack((image_der_x, image_der_y, image_der_z),\
                        axis = -1)

        else:
            raise Exception("Error in function image_der_d() : Invalid"\
                " image dimensionality. Image must be 2D or 3D.")
        return gradient


    def image_der_d2(self, image, hx = 1.0, hy = 1.0, hz = 1.0):
        dim = image.shape

        if len(dim) == 2 :
            image_der_d = self.mirror_boundary_pivot(image, 4, 4)

            kernel_x  =  np.asarray([ -1., 0., 1.])
            kernel_y  =  np.asarray([ -1., 0., 1.])

            image_der_x = 0.5 * (1.0/hx) * self.image_convolve1d(image_der_d,\
                            kernel_x, axis = 1)
            image_der_y = 0.5 * (1.0/hy) * self.image_convolve1d(image_der_d,\
                            kernel_y, axis = 0)

            image_der_x = self.cut_boundry(image_der_x, 3, 3)
            image_der_y = self.cut_boundry(image_der_y, 3, 3)

            g = np.dstack((image_der_x, image_der_y))

        elif len(dim) == 3 :
            kernel_x  =  np.asarray([-1., 0.,  1.])
            kernel_y  =  np.asarray([-1., 0.,  1.])
            kernel_z  =  np.asarray([-1., 0.,  1.])

            image_der_d = self.mirror_boundary_pivot(image, 4, 4, 4)
            image_der_x = 0.5 * (1.0/hx) * self.image_convolve1d(image_der_d,\
                            kernel_x, axis = 2)
            image_der_y = 0.5 * (1.0/hy) * self.image_convolve1d(image_der_d,\
                            kernel_y, axis = 1)
            image_der_z = 0.5 * (1.0/hz) * self.image_convolve1d(image_der_d,\
                            kernel_z, axis = 0)

            image_der_x = self.cut_boundry(image_der_x, xb = 3, yb = 3, zb = 3)
            image_der_y = self.cut_boundry(image_der_y, xb = 3, yb = 3, zb = 3)
            image_der_z = self.cut_boundry(image_der_z, xb = 3, yb = 3, zb = 3)

            g = np.stack((image_der_x, image_der_y, image_der_z), axis = -1)

        else:
            raise Exception("Error in function image_der_d() : Invalid image"\
                            " dimensionality. Image must be 2D or 3D.")
        return g

    #axis = 0 means in z_direction
    #axis = 1 means in y_direction
    #axis = 2 means in x-direction
    #based on didas 2009 paper for 2D case
    def image_der_dd(self, image, dd = 'xx', hx = 1.0, hy = 1.0, hz = 1.0,\
                    kernel_ab_type = 0, ignore_boundary = False, size = 1):

        coeff = 0.5

        if kernel_ab_type == 0 :
            conv_kernel_xy = [
                            [0., -1., 1.],
                            [-1., 2.,-1.],
                            [ 1.,-1., 0.]]

            conv_kernel_yx = [
                            [-1., 1., 0.],
                            [ 1.,-2., 1.],
                            [ 0., 1.,-1.]]

            conv_kernel_xy_3d = [
                            [[ 0., 0., 0.],[ 0., 0., 0.],[ 0., 0., 0.]],
                            [[ 0.,-1., 1.],[-1., 2.,-1.],[ 1.,-1., 0.]],
                            [[ 0., 0., 0.],[ 0., 0., 0.],[ 0., 0., 0.]]]
            conv_kernel_yx_3d = [
                            [[ 0., 0., 0.],[ 0., 0., 0.],[ 0., 0., 0.]],
                            [[-1., 1., 0.],[ 1.,-2., 1.],[ 0., 1.,-1.]],
                            [[ 0., 0., 0.],[ 0., 0., 0.],[ 0., 0., 0.]]]

            conv_kernel_xz_3d = [
                            [[ 0., 0., 0.],[ 0.,-1., 1.],[ 0., 0., 0.]],
                            [[ 0., 0., 0.],[-1., 2.,-1.],[ 0., 0., 0.]],
                            [[ 0., 0., 0.],[ 1.,-1., 0.],[ 0., 0., 0.]]]
            conv_kernel_zx_3d = [
                            [[ 0., 0., 0.],[-1., 1., 0.],[ 0., 0., 0.]],
                            [[ 0., 0., 0.],[ 1.,-2., 1.],[ 0., 0., 0.]],
                            [[ 0., 0., 0.],[ 0., 1.,-1.],[ 0., 0., 0.]]]

            conv_kernel_yz_3d = [
                            [[ 0., 0., 0.],[ 0.,-1., 0.],[ 0., 1., 0.]],
                            [[ 0.,-1., 0.],[ 0., 2., 0.],[ 0.,-1., 0.]],
                            [[ 0., 1., 0.],[ 0.,-1., 0.],[ 0., 0., 0.]]]
            conv_kernel_zy_3d = [
                            [[ 0.,-1., 0.],[ 0., 1., 0.],[ 0., 0., 0.]],
                            [[ 0., 1., 0.],[ 0.,-2., 0.],[ 0., 1., 0.]],
                            [[ 0., 0., 0.],[ 0., 1., 0.],[ 0.,-1., 0.]]]
        elif kernel_ab_type == 1 or kernel_ab_type == 2 :
            coeff = 0.25

            conv_kernel_xy = np.asarray([
                            [ 1., 0.,-1.],
                            [ 0., 0., 0.],
                            [-1., 0., 1.]])
            if kernel_ab_type == 2 :
                conv_kernel_xy = np.asarray([
                            [-1., 0., 1.],
                            [ 0., 0., 0.],
                            [ 1., 0.,-1.]])
            conv_kernel_yx = conv_kernel_xy

            conv_kernel_xy_3d = np.asarray([
                            [[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]],
                            [[ 1, 0,-1],[ 0, 0, 0],[-1, 0, 1]],
                            [[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]]])
            conv_kernel_yx_3d = conv_kernel_xy_3d

            conv_kernel_yz_3d = np.asarray([
                            [[ 0, 0, 0],[ 1, 0,-1],[ 0, 0, 0]],
                            [[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]],
                            [[ 0, 0, 0],[-1, 0, 1],[ 0, 0, 0]]])
            conv_kernel_zy_3d = conv_kernel_yz_3d

            conv_kernel_xz_3d = np.asarray([
                            [[ 0, 1, 0],[ 0, 0, 0],[ 0,-1, 0]],
                            [[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]],
                            [[ 0,-1, 0],[ 0, 0, 0],[ 0, 1, 0]]])
            conv_kernel_zx_3d = conv_kernel_xz_3d


        else:
            raise Exception("Error in function image_der_dd(image, dd):"\
                            " Invalid value for kernel_ab_type variable."\
                            " The value must be 0 or 1 or 2.")

        dim = image.shape

        if( len(dim) == 2 ):
            h, w = dim
            if dd == 'xx':
                image_der_dd = self.mirror_boundary(image, 3, 3)
                image_der_dd = np.diff(image_der_dd, n = 2, axis = 1) *\
                                (1.0/(hx**2))
                image_der_dd = self.cut_boundry(image_der_dd, 2, 3)
                if( ignore_boundary ):
                    # Set the boundary gradinets in xy direction to zero 
                    image_der_dd[:,0] = image_der_dd[:,w-1] = 0.0
                   
                    if( size == 2 ):
                        image_der_dd[:,1] = image_der_dd[:,w-2] = 0.0
                      

            elif dd == 'yy':
                image_der_dd = self.mirror_boundary(image, 3, 3)
                image_der_dd = np.diff(image_der_dd, n = 2, axis = 0) *\
                                (1.0/(hy**2))
                image_der_dd = self.cut_boundry(image_der_dd, 3, 2)
                if( ignore_boundary ):
                    # Set the boundary gradinets in xy direction to zero 
                    
                    image_der_dd[0,:] = image_der_dd[h-1,:] = 0.0
                    if( size == 2 ):
                       
                        image_der_dd[1,:] = image_der_dd[h-2,:] = 0.0
            elif dd == 'xy':
                coeff = coeff * (1.0/(hx*hy))
                conv_kernel_xy = np.multiply(conv_kernel_xy, coeff)
                # Automatically mirrors the boundaries as well
                image_der_dd = self.image_convolve(image, conv_kernel_xy)
                if( ignore_boundary ):
                    # Set the boundary gradinets in xy direction to zero 
                    image_der_dd[:,0] = image_der_dd[:,w-1] = 0.0
                    image_der_dd[0,:] = image_der_dd[h-1,:] = 0.0
                    if( size == 2 ):
                        image_der_dd[:,1] = image_der_dd[:,w-2] = 0.0
                        image_der_dd[1,:] = image_der_dd[h-2,:] = 0.0
            elif dd == 'yx':
                coeff = coeff * (1.0/(hx*hy))
                conv_kernel_yx = np.multiply(conv_kernel_yx, coeff)
                image_der_dd = self.image_convolve(image, conv_kernel_yx)
                if( ignore_boundary ):
                    # Set the boundary gradinets in xy direction to zero 
                    image_der_dd[:,0] = image_der_dd[:,w-1] = 0.0
                    image_der_dd[0,:] = image_der_dd[h-1,:] = 0.0
                    if( size == 2 ):
                        image_der_dd[:,1] = image_der_dd[:,w-2] = 0.0
                        image_der_dd[1,:] = image_der_dd[h-2,:] = 0.0
            
            else:
                raise Exception("In function image_der_dd(image, dd):"\
                                " Invalid direction dd. Possible values"\
                                " for dd are from set {'xx','yy','xy','yx'")

        elif( len(dim) == 3 ):

            if dd == 'xx':
                image_der_dd =self.mirror_boundary(image, xb = 3, yb = 0,\
                                zb = 0)
                image_der_dd = np.diff(image_der_dd, n = 2, axis = 2) *\
                                (1.0/(hx**2))
                image_der_dd = self.cut_boundry(image_der_dd, xb = 2, yb = 0,\
                                zb = 0)

            elif dd == 'yy':
                image_der_dd =self.mirror_boundary(image, xb = 0, yb = 3,\
                                zb = 0)
                image_der_dd = np.diff(image_der_dd, n = 2, axis = 1) *\
                                (1.0/(hy**2))
                image_der_dd = self.cut_boundry(image_der_dd, xb = 0, yb = 2,\
                                zb = 0)
            elif dd == 'zz':
                image_der_dd =self.mirror_boundary(image, xb = 0, yb = 0,\
                                zb = 3)
                image_der_dd = np.diff(image_der_dd, n = 2, axis = 0) *\
                                (1.0/(hz**2))
                image_der_dd = self.cut_boundry(image_der_dd, xb = 0, yb = 0,\
                                zb = 2)


            elif dd == 'xy':
                coeff = coeff * (1.0/(hx*hy))
                conv_kernel_xy_3d = np.multiply(conv_kernel_xy_3d, coeff)
                # Automatically mirrors the boundaries as well
                image_der_dd = self.image_convolve(image, conv_kernel_xy_3d)
            elif dd == 'yx':
                coeff = coeff * (1.0/(hx*hy))
                conv_kernel_yx_3d = np.multiply(conv_kernel_yx_3d, coeff)
                image_der_dd = self.image_convolve(image, conv_kernel_yx_3d)

            elif dd == 'xz':
                coeff = coeff * (1.0/(hx*hz))
                conv_kernel_xz_3d = np.multiply(conv_kernel_xz_3d, coeff)
                image_der_dd = self.image_convolve(image, conv_kernel_xz_3d)
            elif dd == 'zx':
                coeff = coeff * (1.0/(hx*hz))
                conv_kernel_zx_3d = np.multiply(conv_kernel_zx_3d, coeff)
                image_der_dd = self.image_convolve(image, conv_kernel_zx_3d)

            elif dd == 'yz':
                coeff = coeff * (1.0/(hz*hy))
                conv_kernel_yz_3d = np.multiply(conv_kernel_yz_3d, coeff)
                image_der_dd = self.image_convolve(image, conv_kernel_yz_3d)
            elif dd == 'zy':
                coeff = coeff * (1.0/(hz*hy))
                conv_kernel_zy_3d = np.multiply(conv_kernel_zy_3d, coeff)
                image_der_dd = self.image_convolve(image, conv_kernel_zy_3d)

            else:
                raise Exception("In function image_der_dd(image, dd):"\
                                " Invalid direction dd. Possible values"\
                                " for dd are from set {'xx','yy','zz','xy',"\
                                "'yx','zy','yz','xz','zx'")
        else:
            raise Exception("In function image_der_dd(image, dd):"\
                            " Invalid image dimension. Image dimension"\
                            " must be 2D or 3D.")
        '''
        if( ignore_boundary ):
            plot = MyPlot()
            plot.show_images([image_der_dd],1,1,[dd])
        '''
        return image_der_dd


    #axis = 0 means in y_direction
    #axis = 1 means in x_direction
    #based on Hajiaboli 2010 paper
    def image_second_derivative(self, image, dd = 'xx', hx = 1.0, hy = 1.0,\
                                hz = 1.0, type_xy = 0):

        if dd == 'xx':
            image_der_dd = self.mirror_boundary(image, 3, 3)
            image_der_dd = np.diff(image_der_dd, n = 2, axis = 1) *\
                            (1.0 / (hx**2.0))
            image_der_dd = self.cut_boundry(image_der_dd, 2, 3)

        elif dd == 'yy':
            image_der_dd = self.mirror_boundary(image, 3, 3)
            image_der_dd = np.diff(image_der_dd, n = 2, axis = 0) *\
                            (1.0 / (hy**2.0))
            image_der_dd = self.cut_boundry(image_der_dd, 3, 2)

        elif (dd == 'xy' or dd == 'yx'):
            coeff  = 0.25 * (1.0/(hx*hy))
            if( type_xy == 0 ):
                conv_kernel_xy = [
                            [1, 0, -1],
                            [ 0, 0, 0],
                            [ -1, 0,1]]
            else:
                conv_kernel_xy = [
                            [-1, 0, 1],
                            [ 0, 0, 0],
                            [ 1, 0,-1]]
            conv_kernel_xy = np.multiply(conv_kernel_xy, coeff)
            image_der_dd = self.image_convolve(image, conv_kernel_xy)
        else:
            print('Invalid derivation direction. In image_der_dd function')


        return image_der_dd

    def image_at_scale(self, image, t):
        return self.image_convolve_gaussian(image, sigma = t)

    def negative_image(self, u):
        maxv = np.max(u)
        u = maxv - u
        return u

    # Maximum intensity projection along axis
    def mip(self, img, axis):
        mip = np.max(img, axis=axis)
        return mip
        
###############################################################################
#                             Bilateral filter                                #
#           Source code adapted from the code in Skimage library              #
###############################################################################
    def gaussian_weight(self, sigma, value):
        return exp(-0.5 * (value / sigma)**2)


    def compute_color_lut(self, bins,  sigma,  max_value):

        color_lut = np.empty(bins, dtype=np.double)
        
        for b in range(bins):
            color_lut[b] = self.gaussian_weight(sigma, b * max_value / bins)

        return color_lut


    def compute_range_lut(self, win_size_x, win_size_y, sigma, hx=1.0, hy=1.0):

        range_lut = np.empty(win_size_x*win_size_y, dtype=np.double)
        window_ext_x = (win_size_x - 1) / 2
        window_ext_y = (win_size_y - 1) / 2
        for kr in range(win_size_y):
            for kc in range(win_size_x):
                dist = sqrt(((kr - window_ext_y)*hy)**2 + ((kc - window_ext_x)*hx)**2)
                range_lut[kr * win_size_x + kc] = self.gaussian_weight(sigma, dist)

        return range_lut    
    
    def get_pixel3d(self, cimage, rows, cols, dims, rr, cc, d, cmode, cval):
        x = rr
        y = cc
        z = d
        if( cmode == 'reflect' ):
            if( rr < 0 ):
                x = np.abs(rr)
            elif( x >= rows ):
                x = rows - (rr - rows + 2)
                
            if( cc < 0 ):
                y = np.abs(cc)
            elif( y >= cols ):
                y = cols - (cc - cols + 2)
            
            if( d < 0 ):
                z = np.abs(d)
            elif( z >= dims ):
                z = dims - (d - dims + 2)
        else:
            print("In function get_pixel3d(...) undefined cmode")
        return cimage[x,y,z]
       
    def denoise_bilateral(self, image, win_size=None, sigma_range=None,\
                      sigma_spatial=1.0,  bins=10000,\
                      mode='reflect', multichannel=True, cval=0, hx=1.0, hy=1.0):
                          
     #   win_size = max(5, 2*int(ceil(3*sigma_spatial))+1)
     #   hx = 1.0
     #   hy = 1.0
        print(hx, hy)
        win_size_x = max(5, 2*int(ceil(3*sigma_spatial/hx))+1)
        win_size_y = max(5, 2*int(ceil(3*sigma_spatial/hy))+1)
        
        
        # To have the same window ratio as Chen et al. 7x19 (ratio=2,714285714)
        win_size_x = int(ceil(win_size_y * (19./7.)))
        win_size_x = 15
        win_size_y = 5
        image = np.atleast_3d(image.astype(float))
        print("Bilateral window size is:", win_size_x, " x ", win_size_y)
        # if image.max() is 0, then dist_scale can have an unverified value
        # and color_lut[<int>(dist * dist_scale)] may cause a segmentation fault
        # so we verify we have a positive image and that the max is not 0.0.
        if image.min() < 0.0:
            raise ValueError("Image must contain only positive values")
    
        
        rows = image.shape[0]
        cols = image.shape[1]
        dims = image.shape[2]
       # window_ext = (win_size - 1) / 2
        window_ext_x = (win_size_x - 1) / 2
        window_ext_y = (win_size_y - 1) / 2
        max_color_lut_bin = bins - 1

        max_value = 0.0

       # double[:, :, ::1] cimage
       # double[:, :, ::1] out

       # double[:] color_lut
       # double[:] range_lut

       # Py_ssize_t r, c, d, wr, wc, kr, kc, rr, cc, pixel_addr, color_lut_bin
       # double value, weight, dist, total_weight, csigma_range, color_weight, \
       #        range_weight
       # double dist_scale
       # double[:] values
       # double[:] centres
       # double[:] total_values
    
        if sigma_range is None:
            csigma_range = image.std()
        else:
            csigma_range = sigma_range
    
        max_value = image.max()
    
        if max_value == 0.0:
            raise ValueError("The maximum value found in the image was 0.")
    
        if mode not in ('constant', 'wrap', 'symmetric', 'reflect', 'edge'):
            raise ValueError("Invalid mode specified.  Please use `constant`, "
                             "`edge`, `wrap`, `symmetric` or `reflect`.")
        cmode = mode
    
        cimage = np.ascontiguousarray(image)
    
        out = np.zeros((rows, cols, dims), dtype=np.double)
        color_lut = self.compute_color_lut(bins, csigma_range, max_value)
        range_lut = self.compute_range_lut(win_size_x, win_size_y, sigma_spatial, hx, hy)
        dist_scale = bins / dims / max_value
        values = np.empty(dims, dtype=np.double)
        centres = np.empty(dims, dtype=np.double)
        total_values = np.empty(dims, dtype=np.double)
    
        for r in range(rows):
            for c in range(cols):
                total_weight = 0
                for d in range(dims):
                    total_values[d] = 0
                    centres[d] = cimage[r, c, d]
                for wr in range(-window_ext_y, window_ext_y + 1):
                    rr = wr + r
                    kr = wr + window_ext_y
                    for wc in range(-window_ext_x, window_ext_x + 1):
                        cc = wc + c
                        kc = wc + window_ext_x
    
                        # save pixel values for all dims and compute euclidian
                        # distance between centre stack and current position
                        dist = 0
                        for d in range(dims):
                            
                            value = self.get_pixel3d(cimage, rows, cols, dims,
                                                rr, cc, d, cmode, cval)
                            values[d] = value
                            dist += (centres[d] - value)**2
                        dist = sqrt(dist)
    
                        range_weight = range_lut[kr * win_size_x + kc]
    
                        color_lut_bin = min((dist * dist_scale), max_color_lut_bin)
                        color_weight = color_lut[int(color_lut_bin)]
    
                        weight = range_weight * color_weight
                        for d in range(dims):
                            total_values[d] += values[d] * weight
                        total_weight += weight
                for d in range(dims):
                    out[r, c, d] = total_values[d] / total_weight

        return np.squeeze(np.asarray(out))

###############################################################################
#----------------------- Higher order structure tensors ----------------------#
###############################################################################
    def compute_st_tensor_field_H(self, g_field, h_field, rho = 0.0 ):
        h, w, d = g_field.shape



        st = np.empty((h * w * d**4)).reshape(h,w,4,4)
        if( d == 2 ):
            for i in range(h):
                for j in range(w):
                    J = h_field[i, j]
                    J = np.outer(J, h_field[i, j])

                    st[i, j] = J

        # Average over neighborhood.
        if( rho > 0.0 ):
            st[:,:,0,0] = self.image_convolve_gaussian(st[:,:,0,0], rho)
            st[:,:,0,1] = self.image_convolve_gaussian(st[:,:,0,1], rho)
            st[:,:,0,2] = self.image_convolve_gaussian(st[:,:,0,2], rho)
            st[:,:,0,3] = self.image_convolve_gaussian(st[:,:,0,3], rho)
            st[:,:,1,0] = self.image_convolve_gaussian(st[:,:,1,0], rho)
            st[:,:,1,1] = self.image_convolve_gaussian(st[:,:,1,1], rho)
            st[:,:,1,2] = self.image_convolve_gaussian(st[:,:,1,2], rho)
            st[:,:,1,3] = self.image_convolve_gaussian(st[:,:,1,3], rho)
            st[:,:,2,0] = self.image_convolve_gaussian(st[:,:,2,0], rho)
            st[:,:,2,1] = self.image_convolve_gaussian(st[:,:,2,1], rho)
            st[:,:,2,2] = self.image_convolve_gaussian(st[:,:,2,2], rho)
            st[:,:,2,3] = self.image_convolve_gaussian(st[:,:,2,3], rho)
            st[:,:,3,0] = self.image_convolve_gaussian(st[:,:,3,0], rho)
            st[:,:,3,1] = self.image_convolve_gaussian(st[:,:,3,1], rho)
            st[:,:,3,2] = self.image_convolve_gaussian(st[:,:,3,2], rho)
            st[:,:,3,3] = self.image_convolve_gaussian(st[:,:,3,3], rho)

        return st


    def compute_st_tensor_field(self, g_field, order = 2, rho = 0.0 ):
        h, w, d = g_field.shape

        st_shape = [h, w]
        J_shape  = []
        for mm in range(order):
            st_shape.append(d)
            J_shape.append(d)


        st = np.empty((h * w * d**order)).reshape(st_shape)
        if( d == 2 ):
            for i in range(h):
                for j in range(w):
                    J = g_field[i, j]
#                    if( J[0]!=0.0 or J[1]!=0 ):
                    for l in range(order-1):
                        J = np.outer(J, g_field[i, j])
                    J = J.reshape(J_shape)
                    st[i, j] = J

        if( order == 2 and rho > 0.0 ):
            st[:,:,0,0] = self.image_convolve_gaussian(st[:,:,0,0], rho)
            st[:,:,0,1] = self.image_convolve_gaussian(st[:,:,0,1], rho)
            st[:,:,1,0] = self.image_convolve_gaussian(st[:,:,1,0], rho)
            st[:,:,1,1] = self.image_convolve_gaussian(st[:,:,1,1], rho)

        if( order == 4 and rho > 0.0 ):
            st[:,:,0,0,0,0] = self.image_convolve_gaussian(st[:,:,0,0,0,0], rho)
            st[:,:,0,0,0,1] = self.image_convolve_gaussian(st[:,:,0,0,0,1], rho)
            st[:,:,0,0,1,0] = self.image_convolve_gaussian(st[:,:,0,0,1,0], rho)
            st[:,:,0,0,1,1] = self.image_convolve_gaussian(st[:,:,0,0,1,1], rho)
            st[:,:,0,1,0,0] = self.image_convolve_gaussian(st[:,:,0,1,0,0], rho)
            st[:,:,0,1,0,1] = self.image_convolve_gaussian(st[:,:,0,1,0,1], rho)
            st[:,:,0,1,1,0] = self.image_convolve_gaussian(st[:,:,0,1,1,0], rho)
            st[:,:,0,1,1,1] = self.image_convolve_gaussian(st[:,:,0,1,1,1], rho)
            st[:,:,1,0,0,0] = self.image_convolve_gaussian(st[:,:,1,0,0,0], rho)
            st[:,:,1,0,0,1] = self.image_convolve_gaussian(st[:,:,1,0,0,1], rho)
            st[:,:,1,0,1,0] = self.image_convolve_gaussian(st[:,:,1,0,1,0], rho)
            st[:,:,1,0,1,1] = self.image_convolve_gaussian(st[:,:,1,0,1,1], rho)
            st[:,:,1,1,0,0] = self.image_convolve_gaussian(st[:,:,1,1,0,0], rho)
            st[:,:,1,1,0,1] = self.image_convolve_gaussian(st[:,:,1,1,0,1], rho)
            st[:,:,1,1,1,0] = self.image_convolve_gaussian(st[:,:,1,1,1,0], rho)
            st[:,:,1,1,1,1] = self.image_convolve_gaussian(st[:,:,1,1,1,1], rho)

        return st

    def find_contrast_in_direction(self, st_tensor, u ):
        order = len(st_tensor.shape)
        c = st_tensor
        for l in range(order):
            c = np.inner(c, u)
        return c

    def find_maxima_directions(self, st_tensor, res = 2):
        # res: degree resolution
        d = 0
        contrast_list = []
        u_list = []


        coeff = 2.0*np.pi/360.0
        while d < 360:
            u = np.asarray([np.cos(d*coeff), np.sin(d*coeff)])
            if( d == 90 or d == 270 ):
                u[0] = 0.0
            u = (1.0/np.linalg.norm(u)) * u
            c = self.find_contrast_in_direction(st_tensor, u)
            contrast_list.append(c)
            u_list.append(u)
            d += res
        contrast_list = np.asarray(contrast_list)
        u_list = np.asarray(u_list)

        x = np.asarray(sc.signal.argrelmax(contrast_list)[0])
        if( len(x) > 0 ):

            return contrast_list[x], u_list[x]
        else:
            return [],[]
            
    def find_contrast_in_direction_h(self, st_tensor, u ):
        c = st_tensor.reshape((2,2,2,2))
        for l in range(4):
            c = np.inner(c, u)
        return c

    def find_maxima_directions_h(self, st_tensor, res = 2):
        # res: degree resolution
        d = 0
        contrast_list = []
        u_list = []


        coeff = 2.0*np.pi/360.0
        while d < 360:
            u = np.asarray([ np.cos(d*coeff), np.sin(d*coeff)])
            if( d == 90 or d == 270 ):
                u[0] = 0.0
            u = (1.0/np.linalg.norm(u)) * u
           # uu = np.outer(u,u)
            uu = u
            c = self.find_contrast_in_direction_h(st_tensor, uu.flatten())
            contrast_list.append(c)
            u_list.append(u)
            d += res
        contrast_list = np.asarray(contrast_list)
        u_list = np.asarray(u_list)

        x = np.asarray(sc.signal.argrelmax(contrast_list)[0])
        if( len(x) > 0 ):

            return contrast_list[x], u_list[x]
        else:
            return [],[]
            
   

 
    def multiscale_g_h(self, f, scale_image, SCALESPACE, hx = 1.0, hy = 1.0):

        h, w   = f.shape
        scales = np.sort(np.unique(scale_image))

        G = np.zeros((h, w, 2))
        H = np.zeros((h, w, 2, 2))
        for sigma in scales:
            u = np.copy(f)
            if( sigma > 0.0 ):
                if(SCALESPACE):
                    u = self.image_convolve_gaussian(u, sigma)
            else:
                u = self.image_convolve_gaussian(u, 10000.0)
            # Compute gradient of u
            g = self.image_der_d2(u, hx, hy)
            dnum = np.sqrt(g[:,:,0]**2 + g[:,:,1]**2)
            dnum[np.where(dnum==0.0)] = 1.e-20
            g[:,:,0] = g[:,:,0]
            g[:,:,1] = g[:,:,1]

            # Compute Hessian of u
            u_xx = self.image_der_dd(u, 'xx')
            u_yy = self.image_der_dd(u, 'yy')
            u_xy = self.image_der_dd(u, 'xy', kernel_ab_type = 1)

            # Normalize Hessian
            hess = np.empty((h,w,2,2))
            dnum = np.sqrt(u_xx**2 + u_xy**2+ u_xy**2 + u_yy**2)
            dnum[np.where(dnum==0.0)] = 1.e-20

            hess[:,:,0,0] =  u_xx
            hess[:,:,0,1] =  u_xy
            hess[:,:,1,0] =  u_xy
            hess[:,:,1,1] =  u_yy


            # Extract

            mask_base_g = np.repeat(scale_image.flatten(),2).\
                reshape((h, w, 2))
            mask_base_h = np.repeat(scale_image.flatten(),4).\
                reshape((h, w, 2, 2))

            mask = mask_base_g == sigma
            G = G + (mask * g)

            mask = mask_base_h == sigma
            H = H + (mask * hess)


        return G, H

   

###############################################################################
#-----------------------Second order isotropic diffusion----------------------#
###############################################################################
    def perona_malik_diffusity(self, s_2, lambda_2):
        g = 1.0/(1.0 + (s_2/float(lambda_2)))
        return g
    def second_order_isotropic_diffusion(self, u, sigma, var_lambda, tau):
        u_sigma = u
        if(sigma != 0):
            u_sigma = self.image_convolve_gaussian(u_sigma, sigma)

        der_u_sigma = self.image_der_d(u_sigma)

        norm_der_u_sigma = np.linalg.norm(der_u_sigma, axis = 2)
        norm_der_u_sigma_2 = np.power(norm_der_u_sigma, 2)

        var_lambda_2 = pow(var_lambda, 2)

        g = self.perona_malik_diffusity(norm_der_u_sigma_2, var_lambda_2)

        height, width = u.shape
        u_new = np.empty(height*width).reshape(height, width)

        g = self.mirror_boundary_dirichlet(g, 1, 1, 0)
        u = self.mirror_boundary(u, 1, 1)
        for i in range(1, height + 1):
            for j in range(1, width +1):

                g_pos_half_y = np.sqrt(g[i+1,j]*g[i,j])
                g_neg_half_y = np.sqrt(g[i-1,j]*g[i,j])

                g_pos_half_x = np.sqrt(g[i,j+1]*g[i,j])
                g_neg_half_x = np.sqrt(g[i,j-1]*g[i,j])

                u_new[i-1, j-1] = g_pos_half_y*(u[i+1,j]-u[i,j]) -\
                                  g_neg_half_y*(u[i,j]-u[i-1,j]) +\
                                  g_pos_half_x*(u[i,j+1]-u[i,j]) -\
                                  g_neg_half_x*(u[i,j]-u[i,j-1])

        u = self.cut_boundry(u, 1, 1)

        return u + tau * u_new
        


    # Sort based on eigenvalues. Order eigenvalues with decreasing value.
    # For 2d
    def sort_based_on_evalues_increasingly(self, evl, evc):
        if( len(evl.shape) == 2 ):
            list_size, k = evl.shape
            for i in range( list_size ):
                if(evl[i,0] > evl[i,1]):
                    tmp    = evl[i,0]
                    evl[i,0] = evl[i,1]
                    evl[i,1] = tmp

                    tmp      = np.copy(evc[i,:,0])
                    evc[i,:,0] = np.copy(evc[i,:,1])
                    evc[i,:,1] = np.copy(tmp)
        else:
            if(evl[0] > evl[1]):
                tmp    = evl[0]
                evl[0] = evl[1]
                evl[1] = tmp

                tmp      = np.copy(evc[:,0])
                evc[:,0] = np.copy(evc[:,1])
                evc[:,1] = np.copy(tmp)
        return
        
    def image_with_circle_of_ones(self, radius):
        # Source Stack Overflow

        # Set Image size as first odd number larger than radius
        s = floor(radius) * 2 + 1
        img = np.zeros((s,s))

        # specify circle parameters: centre ij and radius. Centre is.
        # in the middle of image
        ci = cj = int(s/2)
        cr = radius

        # Create index arrays to img
        I,J=np.meshgrid(np.arange(img.shape[0]),np.arange(img.shape[1]))

        # calculate distance of all points to centre
        dist=np.sqrt((I-ci)**2+(J-cj)**2)

        # Assign value of 1 to those points where dist<cr:
        img[np.where(dist<cr)] = 1.

        return img

    def phi(self, height, width, radius):

        r = radius
        Phi = np.empty(4*width*height).reshape(height,width,2,2)
        for x in range(height):
            for y in range(width):
                ui = -0.5 + (0.5/(height/2.0))*x
                uj = -0.5 + (0.5/(width /2.0))*y

                uiui_norm =  sqrt(ui**2+ui**2)
                ujuj_norm =  sqrt(uj**2+uj**2)
                uiuj_norm =  sqrt(ui**2+uj**2)

                Phi[x,y,1,1] = 4.*np.pi*ui*ui*\
                            exp(-2.0*(np.pi*1.0*uiui_norm)**2)*\
                            ((1.0/(uiui_norm**2))*\
                            (np.cos(2*np.pi*r*uiui_norm)-\
                            (np.sin(2.0*np.pi*uiui_norm)/\
                            (2.0*np.pi*r*uiui_norm))))
                Phi[x,y,0,0] = 4.*np.pi*uj*uj*\
                            exp(-2.0*(np.pi*1.0*ujuj_norm)**2)*\
                            ((1.0/(ujuj_norm**2))*\
                            (np.cos(2*np.pi*r*ujuj_norm)-\
                            (np.sin(2.0*np.pi*ujuj_norm)/\
                            (2.0*np.pi*r*ujuj_norm))))
                Phi[x,y,0,1] = 4.*np.pi*ui*uj*\
                            exp(-2.0*(np.pi*1.0*uiuj_norm)**2)*\
                            ((1.0/(uiuj_norm**2))*\
                            (np.cos(2*np.pi*r*uiuj_norm)-\
                            (np.sin(2.0*np.pi*uiuj_norm)/\
                            (2.0*np.pi*r*uiuj_norm))))
                Phi[x,y,1,0] = Phi[x,y,0,1]

        return Phi
        

    # e_vals must be matrix of eigen values
    def matrix_from_evals_evecs(self, e_vals_matrix, e_vecs):
        # Eigen vectors as columns of matrix v
        v = e_vecs
        eval_evec_matrix =  np.dot(e_vals_matrix , v.T)
        m = np.dot(v, eval_evec_matrix)

        return m

    # ---------------------- Weickert's method --------------------------------
    def compute_diffusion_tensor_list(self, eval_evec_matrix, var_lambda):

        lambda_2 = pow(var_lambda, 2)

        #Seperate e-values and e-vectors
        eval_evec_matrix = np.reshape(eval_evec_matrix, (4, 5))

        # Reorder them so that evalues are nonincreasing. evectors are sorted
        # accordingly
        e_vals = eval_evec_matrix[:,0]
        e_vecs = eval_evec_matrix[:,1:5]

        mixed_for_sort = np.vstack((e_vals, e_vecs))
        mixed_for_sort = mixed_for_sort.transpose()
        mixed_for_sort.view('f8,f8,f8,f8').sort(order=['f0'], axis=0)
        mixed_for_sort = mixed_for_sort.transpose()

        #Here are the sorted evalues and evectors
        e_vals = mixed_for_sort[0,:]
        e_vals = e_vals[::-1]
        e_vecs = np.fliplr(mixed_for_sort[1:5,:])

        lambda1 = e_vals[0]
        lambda2 = e_vals[1]

        kappa = pow((lambda1 - lambda2),2)

        diffusion_tensor_evalues    = np.empty(4*4).reshape((4,4))
        diffusion_tensor_evalues.fill(0.0)

        diffusion_tensor_evalues[0,0] = 0.001
        diffusion_tensor_evalues[1,1] = 0.001 + 0.999 *\
                    sc.exp(-lambda_2 / (kappa+0.00000000001))
        diffusion_tensor_evalues[2,2] = 0.001 + 0.999 *\
                    sc.exp(-lambda_2 / (kappa+0.00000000001))
        diffusion_tensor_evalues[3,3] = 0.001 + 0.999 *\
                    sc.exp(-lambda_2 / (kappa+0.00000000001))

        #compute local diffusion tensor
        diffusion_tensor =\
            self.matrix_from_evals_evecs(diffusion_tensor_evalues, e_vecs)

        return np.reshape(diffusion_tensor,(16))

###############################################################################
#-------------------Anisotropic filter Fourth order---------------------------#
###############################################################################
    def fourth_order_anisotropic_diffusion(self, u, sigma, rho,\
                var_lambda, tau):

        rows, cols = u.shape

        u_sigma = self.image_convolve_gaussian(u, sigma)

        u_der_xx = self.image_der_dd(u, 'xx')
        u_der_yy = self.image_der_dd(u, 'yy')
        u_der_xy = self.image_der_dd(u, 'xy')
        u_der_yx = self.image_der_dd(u, 'yx')

        u_sigma_der_xx = self.image_der_dd(u_sigma, 'xx')
        u_sigma_der_yy = self.image_der_dd(u_sigma, 'yy')
        u_sigma_der_xy = self.image_der_dd(u_sigma, 'xy')
        u_sigma_der_yx = self.image_der_dd(u_sigma, 'yx')
        #Generalised Structure Tensor
        structure_tensor = np.empty((10, rows, cols))
        structure_tensor[0] = np.power(u_sigma_der_xx, 2)
        structure_tensor[1] = np.multiply(u_sigma_der_xx, u_sigma_der_xy)
        structure_tensor[2] = np.multiply(u_sigma_der_xx, u_sigma_der_yx)
        structure_tensor[3] = np.multiply(u_sigma_der_xx, u_sigma_der_yy)
        structure_tensor[4] = np.power(u_sigma_der_xy, 2)
        structure_tensor[5] = np.multiply(u_sigma_der_xy, u_sigma_der_yx)
        structure_tensor[6] = np.multiply(u_sigma_der_xy, u_sigma_der_yy)
        structure_tensor[7] = np.power(u_sigma_der_yx, 2)
        structure_tensor[8] = np.multiply(u_sigma_der_yx, u_sigma_der_yy)
        structure_tensor[9] = np.power(u_sigma_der_yy, 2)

        #Integration scale
        if(rho > 0):
            structure_tensor[0] =\
                self.image_convolve_gaussian(structure_tensor[0], rho)
            structure_tensor[1] =\
                self.image_convolve_gaussian(structure_tensor[1], rho)
            structure_tensor[2] =\
                self.image_convolve_gaussian(structure_tensor[2], rho)
            structure_tensor[3] =\
                self.image_convolve_gaussian(structure_tensor[3], rho)
            structure_tensor[4] =\
                self.image_convolve_gaussian(structure_tensor[4], rho)
            structure_tensor[5] =\
                self.image_convolve_gaussian(structure_tensor[5], rho)
            structure_tensor[6] =\
                self.image_convolve_gaussian(structure_tensor[6], rho)
            structure_tensor[7] =\
                self.image_convolve_gaussian(structure_tensor[7], rho)
            structure_tensor[8] =\
                self.image_convolve_gaussian(structure_tensor[8], rho)
            structure_tensor[9] =\
                self.image_convolve_gaussian(structure_tensor[9], rho)

        local_structure_tensor =\
            np.dstack((structure_tensor[0], structure_tensor[1],\
                       structure_tensor[2], structure_tensor[3],\
                       structure_tensor[1], structure_tensor[4],\
                       structure_tensor[5], structure_tensor[6],\
                       structure_tensor[2], structure_tensor[5],\
                       structure_tensor[7], structure_tensor[8],\
                       structure_tensor[3], structure_tensor[6],\
                       structure_tensor[8], structure_tensor[9]))

        local_structure_tensor = np.reshape(local_structure_tensor,\
                                    (rows * cols, 4, 4))

        #Computing eigen-values and eigen-vectors for each local structure
        # tensor of each point of image u
        #Note: eigen-values are not sorted
        e_value_list, e_vector_list = np.linalg.eigh(local_structure_tensor)

        # Combine eval and evector together to send them into
        # compute_diffusion_tensor_list
        e_val_vec_list = np.dstack((e_value_list, e_vector_list))
        e_val_vec_list = np.reshape(e_val_vec_list, (rows * cols,20))

        diffusion_tensor_list  = np.apply_along_axis(self.\
            compute_diffusion_tensor_list, 1, e_val_vec_list, var_lambda)
        # diffusion_tensor_list includes diffusion tensor for each
        # pixel of the image
        diffusion_tensor_list = diffusion_tensor_list.\
            reshape((rows , cols, 4, 4))
        u_derivatives =\
            np.dstack((u_der_xx, u_der_xy, u_der_yx, u_der_yy)).astype(float)

        for i in range(0, rows):
            for j in range(0, cols):
                m = diffusion_tensor_list[i,j]
                v = u_derivatives[i,j]
                u_derivatives[i, j] = np.dot(m, v)

        u_der_xx =  u_derivatives[:,:,0]
        u_der_xy =  u_derivatives[:,:,1]
        u_der_yx =  u_derivatives[:,:,2]
        u_der_yy =  u_derivatives[:,:,3]

        incr_xx = self.image_der_dd(u_der_xx, 'xx')
        incr_yy = self.image_der_dd(u_der_yy, 'yy')
        incr_xy = self.image_der_dd(u_der_xy, 'xy')
        incr_yx = self.image_der_dd(u_der_yx, 'yx')

        u_new = u - tau * (incr_xx + incr_yy + incr_xy + incr_yx)

        return u_new

    # According to Hajiaboli paper
    def second_directional_derivatives_eta_xi(self, u_x, u_y, u_xx, u_yy,\
                u_xy):
        u_x_2 = np.power(u_x, 2)
        u_y_2 = np.power(u_y, 2)

        a = np.multiply(u_xx, u_x_2)
        b = np.multiply(u_x, u_y)
        b = np.multiply(b, u_xy) * 2.0
        c = np.multiply(u_yy, u_y_2)


        denum = u_x_2 + u_y_2

        # To handle division by 0! If denum is 0 then a, b and c will also
        #  be 0 and u_etaeta = 0/0 must be set zero
        np.place(denum, denum == 0, [np.inf])

        u_etaeta = (a + b + c) / denum

        a = np.multiply(u_xx, u_y_2)
        c = np.multiply(u_yy, u_x_2)

        u_xixi = (a - b + c) / denum

        return u_etaeta, u_xixi

###############################################################################
#----------------Anisotropic Fourth order - Hajiaboli method------------------#
###############################################################################
    # In Hajiaboli var_lambda is called k
    def hajiaboli_fourth_order_anisotropic_diffusion_filter(self, f,\
                   sigma, var_lambda, tau):
        u = np.copy(f)
        u = self.image_convolve_gaussian(u, sigma)
        gradient_u = self.image_der_d(u)
        g_l2_norm  = np.linalg.norm(gradient_u, axis = 2)

        h, w = u.shape
        h = np.empty((h,w,2,2))
        # Since u_xy == u_yx, computing either is sufficient
        
        u_der_xx = self.image_second_derivative(u, 'xx')
        u_der_yy = self.image_second_derivative(u, 'yy')
        u_der_xy = self.image_second_derivative(u, 'xy', type_xy = 1)
        
        # Computing second order directional derivative in gradient(eta) and
        # level set(xi) directions
        u_der_etaeta, u_der_xixi =\
            self.second_directional_derivatives_eta_xi(gradient_u[:,:,0],\
                    gradient_u[:,:,1], u_der_xx, u_der_yy, u_der_xy)

        # Compute diffusivity using Perona & Malik diffusivity function
        c = self.perona_malik_diffusity(np.power(g_l2_norm, 2),\
                    np.power(var_lambda, 2))

        c_2 = np.power(c, 2)

        tm = (c_2 * u_der_etaeta) + (c * u_der_xixi)

        incr_xx = self.image_der_dd(tm, 'xx')
        incr_yy = self.image_der_dd(tm, 'yy')

        return f - tau * (incr_xx + incr_yy)


    def index_to_string(self, i, j, k = None):
        if( k == None):
            return str(i) + "," + str(j)
        return str(i) + "," + str(j) + "," + str(k)

    def string_to_index(self, string):
        p = string.split(',')
        if( len(p) == 2 ):
            i, j = p
            return [int(i), int(j)]
        i, j, k = p
        return [int(i), int(j), int(k)]

    def find_neighbor_indices_on_cross_section_plane(self, point, nrml,\
                scale_img):

        dim = len( nrml )

        if( dim == 2 ):
            v = np.copy(nrml)

            v[0] = nrml[1]
            v[1] = nrml[0]


            # Normalize normal
            v = (1.0/np.linalg.norm(v)) * v

            all_neighbors = [[ -1,-1],[-1, 0],[-1, 1],\
                             [  0,-1]        ,[ 0, 1],\
                             [  1,-1],[ 1, 0],[ 1, 1]]
            projections = np.asarray([])

            all_neighbors = np.asarray(all_neighbors)
            for n in all_neighbors:
                n = np.asarray(n)
                n = (1.0/np.linalg.norm(n)) * n

                projections = np.append(projections, abs(np.dot(n, v)))

            # Extract 2 minimum projected directio
            sort_args = np.argsort(projections)

            two_neighbors = all_neighbors[sort_args[0:2]]

            return two_neighbors


        elif( dim == 3 ):
            normal = np.copy(nrml)
            normal[0], normal[1], normal[2] = normal[2], normal[0], normal[1]

            # Normalize normal
            normal = (1.0/np.linalg.norm(normal)) * normal
            all_neighbors = [[-1,-1,-1],[-1,-1, 0],[-1,-1, 1],\
                             [-1, 0,-1],[-1, 0, 0],[-1, 0, 1],\
                             [-1, 1,-1],[-1, 1, 0],[-1, 1, 1],\
                             [ 0,-1,-1],[ 0,-1, 0],[ 0,-1, 1],\
                             [ 0, 0,-1]           ,[ 0, 0, 1],\
                             [ 0, 1,-1],[ 0, 1, 0],[ 0, 1, 1],\
                             [ 1,-1,-1],[ 1,-1, 0],[ 1,-1, 1],\
                             [ 1, 0,-1],[ 1, 0, 0],[ 1, 0, 1],\
                             [ 1, 1,-1],[ 1, 1, 0],[ 1, 1, 1]]
            projections = np.asarray([])
            all_neighbors = np.asarray(all_neighbors)
            for n in all_neighbors:
                n = np.asarray(n)
                n = (1.0/np.linalg.norm(n)) * n
                projections = np.append(projections, abs(np.dot(n, normal)))

            # Extract 8 minimum projected directions
            sort_args = np.argsort(projections)
            eight_neighbors = all_neighbors[sort_args[0:8]]
            return eight_neighbors


    def post_filter_scales(self, scale_image, scale_list, on_vessel_indx,\
            scale_max_indx, eigenvectors):
        dim = len(scale_max_indx.shape)
        if( dim == 2 ):

            h, w = scale_max_indx.shape
            processed_pixels = set()
            for v in on_vessel_indx:
                indx_subs = np.unravel_index(v,( h, w))
                v = self.index_to_string(indx_subs[0], indx_subs[1])
                if( v in processed_pixels ):
                    continue

                processed_pixels.add(v)
                plane_cross_section_list = []
                plane_cross_section_set  = set()
                plane_scale_list    = []
                plane_cross_section_list.append(v)
                plane_cross_section_set.add(v)
                ii, jj = self.string_to_index(v)
                plane_scale_list.append(scale_image[ii, jj])
                normal_set = []

                for c in plane_cross_section_list:
                    s_i, s_j = self.string_to_index(c)
                    s = scale_max_indx[s_i, s_j]
                    # Normal of the cross section plane
                    normal = eigenvectors[s, s_i, s_j, :, 0]

                    # Find the eight neighborhood of voxel c on the
                    #cross section plane with n = normal
                    neighbors_c = self.\
                        find_neighbor_indices_on_cross_section_plane\
                        (self.string_to_index(c), normal, scale_image)

                    normal_set.append(normal)

                    for n in neighbors_c:

                        c_i, c_j = self.string_to_index(c)

                        n = n + np.asarray([ c_i, c_j])

                        n[0] = np.clip(n[0], 0, h-1)
                        n[1] = np.clip(n[1], 1, w-1)

                        n_linear = np.ravel_multi_index([n[0], n[1]],\
                                   (h,w))
                        n_str = self.index_to_string(n[0], n[1])

                        # Check if the neighbor is on vessels of has
                        #not been seen before
                        if( n_linear in on_vessel_indx and \
                                not( n_str in plane_cross_section_set) and \
                                not ( n_str in processed_pixels )):
                            plane_cross_section_set.add(n_str)
                            plane_cross_section_list.append(n_str)

                            plane_scale_list.append(scale_image[n[0],n[1]])


                scale_avg = sc.stats.mstats.mode(plane_scale_list, axis = None)


                # Find closest value from scale_list to the average scale
                closest_idx = np.argmin(np.abs(scale_list - scale_avg))
                scale_avg = scale_list[closest_idx]

                # Replace the scale of the pixels on the cross section with
                # the average value
                for c in plane_cross_section_list:
                    ii, jj = self.string_to_index(c)
                    scale_image[ii, jj] = scale_avg
                    # Remove the seen pixels from the pixels to be processed
                    processed_pixels.add(c)
                    
        if( dim == 3 ):
            d, h, w = scale_max_indx.shape
            processed_voxels = set()
            for v in on_vessel_indx:
                indx_subs = np.unravel_index(v,(d, h, w))
                v = self.index_to_string(indx_subs[0], indx_subs[1],\
                    indx_subs[2])
                if( v in processed_voxels ):
                    continue

                processed_voxels.add(v)
                plane_cross_section_list = []
                plane_cross_section_set  = set()
                plane_scale_list    = []
                plane_cross_section_list.append(v)
                plane_cross_section_set.add(v)
                kk, ii, jj = self.string_to_index(v)
                plane_scale_list.append(scale_image[kk, ii, jj])
                normal_set = []
                for c in plane_cross_section_list:
                    s_k, s_i, s_j = self.string_to_index(c)
                    s = scale_max_indx[s_k, s_i, s_j]
                    # Normal of the cross section plane
                    normal = eigenvectors[s, s_k, s_i, s_j, :, 0]

                    # Find the eight neighborhood of voxel c on the
                    #cross section plane with n = normal
                    neighbors_c = self.\
                        find_neighbor_indices_on_cross_section_plane\
                        (self.string_to_index(c), normal, scale_image)

                    normal_set.append(normal)
                    for n in neighbors_c:
                        c_k, c_i, c_j = self.string_to_index(c)

                        n = n + np.asarray([c_k, c_i, c_j])
                        n[0] = np.clip(n[0], 0, d-1)
                        n[1] = np.clip(n[1], 1, h-1)
                        n[2] = np.clip(n[2], 2, w-1)


                        n_linear = np.ravel_multi_index([n[0], n[1], n[2]],\
                                   (d,h,w))
                        n_str = self.index_to_string(n[0], n[1], n[2])

                        # Check if the neighbor is on vessels of has
                        #not been seen before
                        if( n_linear in on_vessel_indx and \
                                not( n_str in plane_cross_section_set) and \
                                not ( n_str in processed_voxels )):
                            plane_cross_section_set.add(n_str)
                            plane_cross_section_list.append(n_str)

                            plane_scale_list.append(scale_image[n[0],n[1],\
                                                            n[2]])

                scale_avg = np.average(plane_scale_list)

                # Find closest value from scale_list to the average scale
                closest_idx = np.argmin(np.abs(scale_list - scale_avg))
                scale_avg = scale_list[closest_idx]
                
                # Replace the scale of the pixels on the cross section with
                # the average value
                for c in plane_cross_section_list:
                    kk, ii, jj = self.string_to_index(c)
                    scale_image[kk, ii, jj] = scale_avg
                    
                    # Remove the seen pixels from the pixels to be processed
                    processed_voxels.add(c)
        return scale_image

###############################################################################
#---------- Slow version of Vesselness Filter by Frangi et al. ---------------#
###############################################################################
    def slow_vesselness_measure(self, f, sigma_list, general_sigma, rho,\
        c_var, alpha_var, beta_var, theta_var, crease_type = "r",\
        detect_planes=False):
        """
        -------Vesselness Filter by Frangi et al.-------

        f : 2D or 3D input image
        sigma_list : a set of possible scales in the image. E.g.
                        {0.5, 1.0, 1.5, 2.0}
        c_var : Indicates filter's sensivity to the background measure S i.e.
                        the Frobeneous norm of the Hessians.
        alpha_var : Indicates filter's sensivity to R_A parameter, which
                        calculates the deviation from blob-like structures
        beta_var : Indicatres filter's sensitivity to R_B parameter, which is
                        the distinguisher between plate-like and line-like
                        structures
        theta_var : A float between 0 and 1. Used for segmenting the vessels
                        from backgournd according to the computed vesselness
                        measurements
        crease_type : "r" for ridge detection, "v" for vessel detection, "rv"
                        for both.

        """
        dim = f.shape

        # Vesselness measure for 2D Images
        if( len(dim) == 2 ):

            # height, width
            h, w = dim

        elif( len(dim) == 3 ):

            # depth, height, width
            d, h, w = dim

            u = np.copy(f)

            num_scales = len(sigma_list)

            general_H = np.empty(h*w*d*9).reshape(d, h, w, 3, 3)
            u_sig = np.copy(u)

            # Smoothing of u
            if( general_sigma > 0. ):
                u_sig = self.image_convolve_gaussian(u, general_sigma)

            # Compute gradient of u
            # Normalizing factor for Hessian at scale sigma
            der_u = self.image_der_d(u_sig)
            denom=1.0/( np.sqrt( 1.0 + der_u[:,:,:,0]**2 + der_u[:,:,:,1]**2 +\
                        der_u[:,:,:,2]**2 ))

            #---------Compute Diffusivity------------------
            # Second derivatives of u ( smoothed version of f )
            u_der_xx = self.image_der_dd(u_sig, 'xx')
            u_der_yy = self.image_der_dd(u_sig, 'yy')
            u_der_zz = self.image_der_dd(u_sig, 'zz')

            u_der_xy = self.image_der_dd(u_sig, 'xy', kernel_ab_type = 1)
            u_der_xz = self.image_der_dd(u_sig, 'xz', kernel_ab_type = 1)
            u_der_yz = self.image_der_dd(u_sig, 'yz', kernel_ab_type = 1)

            # Copy the values into the 3x3 Hessian matrices
            general_H[:,:,:,0,0] = np.copy(u_der_xx) * denom
            general_H[:,:,:,0,1] = np.copy(u_der_xy) * denom
            general_H[:,:,:,0,2] = np.copy(u_der_xz) * denom
            general_H[:,:,:,1,0] = np.copy(u_der_xy) * denom
            general_H[:,:,:,1,1] = np.copy(u_der_yy) * denom
            general_H[:,:,:,1,2] = np.copy(u_der_yz) * denom
            general_H[:,:,:,2,0] = np.copy(u_der_xz) * denom
            general_H[:,:,:,2,1] = np.copy(u_der_yz) * denom
            general_H[:,:,:,2,2] = np.copy(u_der_zz) * denom

            # Apply smoothing on structure tensor. smooth each channel
            # separately.
            if(rho > 0.):

                general_H[:,:,:,0,0] = self.image_convolve_gaussian(general_H[:,:,:,0,0], rho)
                general_H[:,:,:,0,1] = self.image_convolve_gaussian(general_H[:,:,:,0,1], rho)
                general_H[:,:,:,0,2] = self.image_convolve_gaussian(general_H[:,:,:,0,2], rho)
                general_H[:,:,:,1,0] = self.image_convolve_gaussian(general_H[:,:,:,1,0], rho)
                general_H[:,:,:,1,1] = self.image_convolve_gaussian(general_H[:,:,:,1,1], rho)
                general_H[:,:,:,1,2] = self.image_convolve_gaussian(general_H[:,:,:,1,2], rho)
                general_H[:,:,:,2,0] = self.image_convolve_gaussian(general_H[:,:,:,2,0], rho)
                general_H[:,:,:,2,1] = self.image_convolve_gaussian(general_H[:,:,:,2,1], rho)
                general_H[:,:,:,2,2] = self.image_convolve_gaussian(general_H[:,:,:,2,2], rho)

            # Allocate memory for Vesselness measure for different scales and
            # Hessian
            V = np.empty(h*w*d*num_scales).reshape(num_scales, d, h, w)
            G = np.empty(num_scales*h*w*d*3).reshape(num_scales, d, h, w, 3)
            H = np.empty(num_scales*h*w*d*9).reshape(num_scales, d, h, w, 3, 3)

#             # Only for visualization
#             HR = np.empty(h*w*num_scales).reshape(num_scales, h, w)
#             HL = np.empty(h*w*num_scales).reshape(num_scales, h, w)

            # Compute Hessian for all the scales
            k = 0
            for sigma in sigma_list:
                sig_sqr = sigma**2
                # Smoothing of u
                u_sig = self.image_convolve_gaussian(u, sigma)
                # Compute gradient of u
                der_u = self.image_der_d(u_sig)

                # Normalizing factor for Hessian at scale sigma
                denom=1.0/( np.sqrt( 1.0 + der_u[:,:,:,0]**2 +\
                            der_u[:,:,:,1]**2 + der_u[:,:,:,2]**2 ))

                # Second derivatives of u ( smoothed version of f )
                u_der_xx = self.image_der_dd(u_sig, 'xx')
                u_der_yy = self.image_der_dd(u_sig, 'yy')
                u_der_zz = self.image_der_dd(u_sig, 'zz')

                u_der_xy = self.image_der_dd(u_sig, 'xy', kernel_ab_type = 1)
                u_der_xz = self.image_der_dd(u_sig, 'xz', kernel_ab_type = 1)
                u_der_yz = self.image_der_dd(u_sig, 'yz', kernel_ab_type = 1)

                # Copy the values into the 3x3 Hessian matrices
                H[k,:,:,:,0,0] = np.copy(u_der_xx) * denom
                H[k,:,:,:,0,1] = np.copy(u_der_xy) * denom
                H[k,:,:,:,0,2] = np.copy(u_der_xz) * denom
                H[k,:,:,:,1,0] = np.copy(u_der_xy) * denom
                H[k,:,:,:,1,1] = np.copy(u_der_yy) * denom
                H[k,:,:,:,1,2] = np.copy(u_der_yz) * denom
                H[k,:,:,:,2,0] = np.copy(u_der_xz) * denom
                H[k,:,:,:,2,1] = np.copy(u_der_yz) * denom
                H[k,:,:,:,2,2] = np.copy(u_der_zz) * denom

                G[k,:,:,:,:] = np.copy(der_u)

                # Apply smoothing on structure tensor. smooth each channel
                # separately.
                if(rho > 0.):
                    H[k,:,:,:,0,0] = self.image_convolve_gaussian(H[k,:,:,:,0,0], rho)
                    H[k,:,:,:,0,1] = self.image_convolve_gaussian(H[k,:,:,:,0,1], rho)
                    H[k,:,:,:,0,2] = self.image_convolve_gaussian(H[k,:,:,:,0,2], rho)
                    H[k,:,:,:,1,0] = self.image_convolve_gaussian(H[k,:,:,:,1,0], rho)
                    H[k,:,:,:,1,1] = self.image_convolve_gaussian(H[k,:,:,:,1,1], rho)
                    H[k,:,:,:,1,2] = self.image_convolve_gaussian(H[k,:,:,:,1,2], rho)
                    H[k,:,:,:,2,0] = self.image_convolve_gaussian(H[k,:,:,:,2,0], rho)
                    H[k,:,:,:,2,1] = self.image_convolve_gaussian(H[k,:,:,:,2,1], rho)
                    H[k,:,:,:,2,2] = self.image_convolve_gaussian(H[k,:,:,:,2,2], rho)

                    G[k,:,:,:,0] =\
                        self.image_convolve_gaussian(G[k,:,:,:,0], rho)
                    G[k,:,:,:,1] =\
                        self.image_convolve_gaussian(G[k,:,:,:,1], rho)

                hess = H[k,:,:,:,:,:]*sig_sqr
                evls, evcs = np.linalg.eigh(hess)

                # Reshape the array for finding the max norm of hessians.
                # To convert (i,j,k) index to linear index
                # (k*row*col + i*col + j) can be used.
                one_d_evls = evls.reshape(w*h*d,3)

                if( c_var == [] ):
                    # Find the maximum norm of Hessians to compute c
                    max_norm =\
                        max(np.linalg.norm(one_d_evls, ord = 2, axis = 1))
                    c = max_norm * 0.5
                else:
                    c = c_var[k]

                eps = 1.e-20

                sorted_evls = np.copy(evls)
                sorted_evcs = np.copy(evcs)


                arg_sort_evls = np.argsort(np.abs(evls))
                sorted_evls = sorted_evls[arg_sort_evls]


                sorted_evcs = sorted_evcs[arg_sort_evls]

                for i in range(h):
                    for j in range(w):
                        for z in range(d):

                            evl_sorted, evc_sorted =\
                                self.\
                                sort_based_on_evalues_modulus_increasingly(\
                                evls[z, i, j], evcs[z, i, j])

                            RB = abs(evl_sorted[0]) /\
                                (sqrt(abs((evl_sorted[1]+eps) *\
                                (evl_sorted[2]+eps))))

                            RA = abs(evl_sorted[1]) / abs(evl_sorted[2]+eps)
                            S  = np.sqrt((evl_sorted[0]**2 +\
                                          evl_sorted[1]**2 +\
                                          evl_sorted[2]**2))
                            if(detect_planes):
                                v_measure =\
                                    (exp(-1.0*(RB**2/2.0*beta_var**2))) *\
                                    (1.0-exp(-1.0*(S**2/(2.0*c**2))))
                            else:
                                v_measure =\
                                    (1.0-exp(-1.0*(RA**2/(2.0*alpha_var**2))))\
                                    * (exp(-1.0*(RB**2/2.0*beta_var**2))) *\
                                    (1.0-exp(-1.0*(S**2/(2.0*c**2))))

                            if( crease_type == "r" ):
                                if (evl_sorted[1]>0 or evl_sorted[2]>0):
                                    V[k, z, i, j] = 0.0
                                else:
                                    V[k, z, i, j] = v_measure
                            elif( crease_type == "v" ):
                                if (evl_sorted[1]<0 or evl_sorted[2]<0):
                                    V[k, z, i, j] = 0.0
                                else:
                                    V[k, z, i, j] = v_measure
                            elif ( crease_type == "rv" ):
                                V[k, z, i, j] = v_measure
                            else:
                                raise Exception("Error in function\
                                vesselness_measure(): Invalid crease_type.\
                                Crease_type must be from the set\
                                {'r','v','rv'}")
                k += 1

            on_vessel_voxels = dict()
            scale_arr = np.empty(h*w*d).reshape(d,h,w)
            scale_arr.fill(0)

            for i in range(h):
                for j in range(w):
                    for z in range(d):

                        local_V = V[:,z, i, j]
                        max_index = self.find_max(local_V)
                        max_V = local_V[max_index]

                        # If pixel is part of vessel
                        if(max_V >= theta_var):
                            # Find the maximum V for pixel i, j
                            s_max = sigma_list[max_index]
                            scale_arr[z, i, j] = s_max
                            local_h = H[int(s_max), z, i , j]
                            evl, evc = np.linalg.eigh(local_h)
                            evl_sorted, evc_sorted = self.\
                                sort_based_on_evalues_modulus_increasingly(\
                                    evl, evc)
                            on_vessel_voxels[self.index_to_string(z, i, j)] =\
                                    evc_sorted[:,0]



            scale_arr = self.slow_post_filter_scales_3d(scale_arr,\
                            on_vessel_voxels)

        else:
            raise Exception("In vesselness_measure function: Invalid"\
                            " dimensionality of file. Input file must"\
                            " be either 2D or 3D.")

        return scale_arr, on_vessel_voxels

    def slow_post_filter_scales_3d(self, scale_image, on_vessel_voxels):
            processed_voxels = set()
            for v in on_vessel_voxels:

                if( v in processed_voxels ):
                    continue

                processed_voxels.add(v)
                plane_cross_section_list = []
                plane_cross_section_set  = set()
                plane_scale_list    = []
                plane_cross_section_list.append(v)
                plane_cross_section_set.add(v)
                kk, ii, jj = self.string_to_index(v)
                plane_scale_list.append(scale_image[kk, ii, jj])
                normal_set = []
                for c in plane_cross_section_list:
                    # Normal of the cross section plane
                    normal = on_vessel_voxels[c]
                    # Find the eight neighborhood of voxel c on the cross
                    # section plane with n = normal
                    neighbors_c = self.\
                        find_neighbor_indices_on_cross_section_plane\
                        (self.string_to_index(c), normal)
                    normal_set.append(normal)
                    for n in neighbors_c:
                        c_k, c_i, c_j = self.string_to_index(c)
                        n = n + [c_k, c_i, c_j]
                        n_str = self.index_to_string(n[0], n[1], n[2])

                        # Check if the neighbor is on vessels of has not been
                        # seen before
                        if( n_str in on_vessel_voxels and\
                                not( n_str in plane_cross_section_set) and
                                not ( n_str in processed_voxels )):
                            plane_cross_section_set.add(n_str)
                            plane_cross_section_list.append(n_str)
                            plane_scale_list.append(scale_image[n[0],n[1],\
                                            n[2]])

                scale_avg = np.average(plane_scale_list)
                # Replace the scale of the pixels on the cross section with
                # the average value
                for c in plane_cross_section_list:
                    kk, ii, jj = self.string_to_index(c)
                    scale_image[kk, ii, jj] = scale_avg
                    # Remove the seen pixels from the pixels to be processed
                    processed_voxels.add(c)
                return scale_image
                
    def compute_Gradient_3d(self, u, hx = 1.0, hy = 1.0, hz = 1.0):
            
            d, h, w = u.shape
            
            # Compute gradient of u
            # Normalizing factor for Hessian at scale sigma
            der_u = self.image_der_d(u, hx, hy, hz)
          #  pp.show_images([der_u[12,:,:,0],der_u[12,:,:,1],der_u[12,:,:,2]],1,3)
            return der_u

    def compute_Hessian_3d(self, u, hx = 1.0, hy = 1.0, hz = 1.0):
        # depth, height, width
        d, h, w = u.shape

       
        H = np.empty(h*w*d*9).reshape(d, h, w, 3, 3)
    

        # Second derivatives of u ( smoothed version of f )
        u_der_xx = self.image_der_dd(u, 'xx', hx, hy, hz)
        u_der_yy = self.image_der_dd(u, 'yy', hx, hy, hz)
        u_der_zz = self.image_der_dd(u, 'zz', hx, hy, hz)

        u_der_xy = self.image_der_dd(u, 'xy', hx, hy, hz,\
                    kernel_ab_type = 1)
        u_der_xz = self.image_der_dd(u, 'xz', hx, hy, hz,\
                    kernel_ab_type = 1)
        u_der_yz = self.image_der_dd(u, 'yz', hx, hy, hz,\
                    kernel_ab_type = 1)
        denom = 1.0
        # Copy the values into the 3x3 Hessian matrices
        H[:,:,:,0,0] = np.copy(u_der_xx) * denom
        H[:,:,:,0,1] = np.copy(u_der_xy) * denom
        H[:,:,:,0,2] = np.copy(u_der_xz) * denom
        H[:,:,:,1,0] = np.copy(u_der_xy) * denom
        H[:,:,:,1,1] = np.copy(u_der_yy) * denom
        H[:,:,:,1,2] = np.copy(u_der_yz) * denom
        H[:,:,:,2,0] = np.copy(u_der_xz) * denom
        H[:,:,:,2,1] = np.copy(u_der_yz) * denom
        H[:,:,:,2,2] = np.copy(u_der_zz) * denom
        return H
        
    def compute_Hessian_3d_at_position_ijk(self, u, hx = 1.0, hy = 1.0, hz = 1.0):
        # depth, height, width
        d, h, w = u.shape
       
        H = np.empty(h*w*d*9).reshape(d, h, w, 3, 3)

        # Second derivatives of u ( smoothed version of f )
        u_der_xx = self.image_der_dd(u, 'xx', hx, hy, hz)
        u_der_yy = self.image_der_dd(u, 'yy', hx, hy, hz)
        u_der_zz = self.image_der_dd(u, 'zz', hx, hy, hz)

        u_der_xy = self.image_der_dd(u, 'xy', hx, hy, hz,\
                    kernel_ab_type = 1)
        u_der_xz = self.image_der_dd(u, 'xz', hx, hy, hz,\
                    kernel_ab_type = 1)
        u_der_yz = self.image_der_dd(u, 'yz', hx, hy, hz,\
                    kernel_ab_type = 1)
        denom = 1.0
        # Copy the values into the 3x3 Hessian matrices
        H[:,:,:,0,0] = np.copy(u_der_xx) * denom
        H[:,:,:,0,1] = np.copy(u_der_xy) * denom
        H[:,:,:,0,2] = np.copy(u_der_xz) * denom
        H[:,:,:,1,0] = np.copy(u_der_xy) * denom
        H[:,:,:,1,1] = np.copy(u_der_yy) * denom
        H[:,:,:,1,2] = np.copy(u_der_yz) * denom
        H[:,:,:,2,0] = np.copy(u_der_xz) * denom
        H[:,:,:,2,1] = np.copy(u_der_yz) * denom
        H[:,:,:,2,2] = np.copy(u_der_zz) * denom
        return H
        
###############################################################################
#-------------------- Vesselness Filter by Frangi et al. ---------------------#
###############################################################################
    def vesselness_measure(self, f, sigma_list, general_sigma, rho, c_var,\
        alpha_var, beta_var, theta_var, crease_type = "r",\
        postprocess = True, detect_planes=False, hx = 1.0, hy = 1.0, hz = 1.0,\
        RETURN_HESSIAN = False, ignore_boundary=False, BOUNDARY_CONDITION='None'):
        """
        -------Vesselness Filter by Frangi et al.-------

        f : 2D or 3D input image
        sigma_list : a set of possible scales in the image.
        E.g. {0.5, 1.0, 1.5, 2.0}
        c_var : Indicates filter's sensivity to the background measure S i.e.
            the Frobeneous norm of the Hessians.
        alpha_var : Indicates filter's sensivity to R_A parameter, which
            calculates the deviation from blob-like structures
        beta_var : Indicatres filter's sensitivity to R_B parameter, which is
            the distinguisher between plate-like and line-like structures
        theta_var : A float between 0 and 1. Used for segmenting the vessels
            from backgournd according to the computed vesselness measurements
        crease_type : "r" for ridge detection, "v" for vessel detection,
            "rv" for both.

        """
        dim = f.shape

        # To return
        scale = []
        e_val = []
        e_vec = []

#        s_eps = 1.e-3

        # Vesselness measure for 2D Images
        if( len(dim) == 2 ):

            # height, width
            h, w = dim


#            pp = MyPlot()
            u = np.copy(f)

            num_scales = len(sigma_list)
            
            bx = 2
            by = 2            
            if( general_sigma > 0. ):
                u_sig = self.image_convolve_gaussian(u, general_sigma)
                u = u_sig
            
            # Allocate memory for Vesselness measure for different scales and
            # Hessian2
            V = np.empty(h*w*num_scales).reshape(num_scales, h, w)
            G = np.empty(num_scales*h*w*2).reshape(num_scales,  h, w, 2)
            H = np.empty(num_scales*h*w*4).reshape(num_scales, h, w, 2, 2)
            Evls = np.empty(num_scales*h*w*2).reshape(num_scales,  h, w, 2)
            Evcs = np.empty(num_scales*h*w*4).reshape(num_scales,  h, w,\
                2, 2)
            H_mts = np.empty(num_scales*h*w*4).reshape(num_scales,h, w, 2, 2)
           
#             # Only for visualization
#             HR = np.empty(h*w*num_scales).reshape(num_scales, h, w)
#             HL = np.empty(h*w*num_scales).reshape(num_scales, h, w)


            # Compute Hessian for all the scales
            k = 0
            for sigma in sigma_list:


                sig_sqr = (sigma**2)

                # Smoothing of u
                u_sig = self.image_convolve_gaussian(u, sigma)
                #u_sig = np.lib.pad(u_sig, (bx, by), mode='reflect')
                
                # Compute gradient of u
                der_u = self.image_der_d(u_sig, hx, hy)
                
                if( ignore_boundary ):
                    der_u = self.image_der_d(u_sig, hx, hy)
                # Normalizing factor for Hessian at scale sigma a la Canero
                denom=1.0/( np.sqrt( 1.0 + der_u[:,:,0]**2 + der_u[:,:,1]**2 ))
#                denom=sig_sqr/2.0

                # Second derivatives of u ( smoothed version of f )
                u_der_xx = self.image_der_dd(u_sig, 'xx', hx, hy)
                u_der_yy = self.image_der_dd(u_sig, 'yy', hx, hy)
                u_der_xy = self.image_der_dd(u_sig, 'xy', hx, hy,\
                            kernel_ab_type = 1)
                
               
                # Copy the values into the 3x3 Hessian matrices
                H[k,:,:,0,0] = np.copy(u_der_xx) * denom
                H[k,:,:,0,1] = np.copy(u_der_xy) * denom
                H[k,:,:,1,0] = np.copy(u_der_xy) * denom
                H[k,:,:,1,1] = np.copy(u_der_yy) * denom
                ''' 
                if(rho > 0.):
                    H[k,:,:,0,0] =\
                    self.image_convolve_gaussian(H[k,:,:,0,0], rho)
                    H[k,:,:,0,1] =\
                    self.image_convolve_gaussian(H[k,:,:,0,1], rho)
                    H[k,:,:,1,0] =\
                    self.image_convolve_gaussian(H[k,:,:,1,0], rho)
                    H[k,:,:,1,1] =\
                    self.image_convolve_gaussian(H[k,:,:,1,1], rho)
                '''
                G[k,:,:,:] = np.copy(der_u)
                
                hess = np.empty(h*w*4).reshape(h, w, 2, 2)
                hess[:,:,0,0] = np.copy(u_der_xx) * sig_sqr
                hess[:,:,1,1] = np.copy(u_der_yy) * sig_sqr
                hess[:,:,0,1] = hess[:,:,1,0] = np.copy(u_der_xy) * sig_sqr

                H_mts[k,:,:,:,:] = np.copy(hess)
                evls, evcs = np.linalg.eigh(hess)


                # Reshape the array for finding the max norm of hessians.
                # To convert (i,j,k) index to linear index
                # (k*row*col + i*col + j) can be used.
                one_d_evls = evls.reshape(w*h,2)

                if( c_var == [] ):
                    # Find the maximum norm of Hessians to compute c
                    max_norm = max(np.linalg.norm(one_d_evls, ord = 2,\
                        axis = 1))
                    c = max_norm * 0.5
                else:
                    c = c_var[k]

                eps = 1.e-20

                # -- Sort evals & evecs based on evalue modulus increasingly --
                Evls[k,:,:,:]   = np.copy(evls)
                Evcs[k,:,:,:,:] = np.copy(evcs)

                sorting_idx = np.argsort(np.abs(evls))

                # Linearize the array of indices
                idx         = sorting_idx.flatten() +\
                    np.repeat(np.arange(w*h) * 2, 2)
                lin_evls    = Evls[k,:,:,:].flatten()[idx]
                Evls[k,:,:,:] = lin_evls.reshape((  h, w, 2 ))

                # Sort the eigenvectors according to eigenvalues
                evcs_tmp    = np.transpose(Evcs[k,:,:,:,:],\
                    axes = (0,1,3,2))
                lin_evcs    = evcs_tmp.reshape( ( w*h*2, 2 ) )[idx,:]
                Evcs[k,:,:,:,:] = np.transpose(\
                    lin_evcs.reshape((h,w,2,2)), axes = (0,1,3,2))
                # ----- End of eval/evec sorting ----



                RB = (Evls[k,:,:,0]+eps) / (Evls[k,:,:,1]+eps)
                S  = np.sqrt(Evls[k,:,:,0]**2 + Evls[k,:,:,1]**2)


                v_measure = (np.exp(-1.0*(RB**2/(2.0*beta_var**2))))\
                          * (1.0-np.exp(-1.0*(S**2/(2.0*c**2))))

                if( crease_type == "r" ):
                    c_1 = Evls[k,:,:,1] <= 0.0
                    # If(evl1 > 0 or evl2 > 0): V = 0.0 else V = v_measure
                    # Here "*" is boolean operator
                    V[k,:,:] = c_1 * v_measure

                elif( crease_type == "v" ):
                    c_1 = Evls[k,:,:,1] >= 0.0
                    # If(evl1 < 0 or evl2 < 0): V = 0.0 else V = v_measure
                    # Here "*" is boolean operator
                    V[k,:,:] = c_1 * v_measure

                elif ( crease_type == "rv" ):
                    V[k,:,:] = v_measure
                else:
                    raise Exception("Error in function vesselness_measure():\
                    Invalid crease_type. Crease_type must be from the \
                    set {'r','v','rv'}")
                H_mts[k,:,:,:,:] = np.copy(H[k,:,:,:,:])
                
                # ----- End of eval/evec sorting ----

                k += 1

            scale_arr = np.empty(h*w).reshape(h,w)
            scale_arr.fill(0)

            max_idx = np.argmax(V, axis = 0)
            max_val = np.max(V, axis = 0)

            scale_l = np.asarray(sigma_list)
            mask_on = max_val >= theta_var
            max_scl = scale_l[max_idx] * mask_on
            max_idx = max_idx * mask_on

            # A set of point indices that lie on the vessels
            on_vess_idx = np.arange(h*w)
            on_vess_idx = set(on_vess_idx * mask_on.flatten())
            if (not mask_on[0,0]):
                on_vess_idx.discard(0)

            scale = max_scl
            if( postprocess ):
                self.post_filter_scales_2( scale_l,max_scl, H_mts)
                
            # Update max_idx according to post_filter_scales
            for k in range(len(sigma_list)):
                max_idx[np.where(scale==scale_l[k])] = k

            
            # Extract only eigenvalues and eigenvectors of the max_scale for
            # each voxel in a dxhxw grid from Evls and Evcs
            mask_base_evl = np.repeat(max_idx.flatten(),2).\
                reshape(( h, w, 2))
            mask_base_evc = np.repeat(max_idx.flatten(),4).\
                reshape(( h, w, 2, 2))

            h_mts = np.empty((h, w, 2, 2))
            h_mts.fill(0.0)
            for k in range(len(sigma_list)):
                
                mask = mask_base_evc == k
                h_mts = h_mts + (mask * H_mts[k,:,:,:,:])

            

            # smooth *after* scale selection to remove discontinuities
            # that can be introduced by scale selection

            if( BOUNDARY_CONDITION == 'natural'):
                
                tmp_h = np.empty((h+(2*by), w+(2*bx), 2, 2))
                tmp_h.fill(0)
                tmp_h[by+1:h+by-1,bx+1:w+bx-1,:,:] = h_mts[1:h-1,1:w-1,:,:]
                h_mts = np.copy(tmp_h)
                h = h+(2*by)
                w = w+(2*bx)
                
            e_val = np.empty((h, w, 2))
            e_vec = np.empty((h, w, 2, 2))
            e_val.fill(0.0)
            e_vec.fill(0.0)  
            if(rho > 0.):
                h_mts[:,:,0,0] =\
                    self.image_convolve_gaussian(h_mts[:,:,0,0], rho)
                h_mts[:,:,0,1] =\
                    self.image_convolve_gaussian(h_mts[:,:,0,1], rho)
                h_mts[:,:,1,0] =\
                    self.image_convolve_gaussian(h_mts[:,:,1,0], rho)
                h_mts[:,:,1,1] =\
                    self.image_convolve_gaussian(h_mts[:,:,1,1], rho)
            
            # compute evals/evecs of regularized result
            evls, evcs = np.linalg.eigh(h_mts[:,:,:,:])
            
            # -- Sort evals & evecs based on evalue modulus increasingly --
            e_val[:,:,:]   = np.copy(evls)
            e_vec[:,:,:,:] = np.copy(evcs)
            
            sorting_idx = np.argsort(np.abs(e_val))
            
            # Linearize the array of indices
            idx         = sorting_idx.flatten() +\
                np.repeat(np.arange(w*h) * 2, 2)
            lin_evls    = e_val[:,:,:].flatten()[idx]
            e_val[:,:,:] = lin_evls.reshape((  h, w, 2 ))
            
            # Sort the eigenvectors according to eigenvalues
            evcs_tmp    = np.transpose(e_vec[:,:,:,:],\
                                           axes = (0,1,3,2))
            lin_evcs    = evcs_tmp.reshape( ( w*h*2, 2 ) )[idx,:]
            e_vec[:,:,:,:] = np.transpose(\
                lin_evcs.reshape((h,w,2,2)), axes = (0,1,3,2))
            
           # plot.show_image(np.abs(e_val[:,:,0]))
          #  plot.draw_eigen_vector_field_evecs(f, e_vec, on_vess_idx, arrow_scale = 1.0)

           # plot.show_image(np.abs(e_val[:,:,1]))


            if( postprocess ):
                if( RETURN_HESSIAN ):
                    return scale, e_val, e_vec, h_mts, on_vess_idx
                return scale, e_val, e_vec, on_vess_idx
            else:
                if( RETURN_HESSIAN ):
                    return scale, e_val, e_vec, h_mts, mask_on
                return scale, e_val, e_vec, mask_on
        elif( len(dim) == 3 ):

            # depth, height, width
            d, h, w = dim

            u = np.copy(f)

            num_scales = len(sigma_list)

            general_H = np.empty(h*w*d*9).reshape(d, h, w, 3, 3)
            u_sig = np.copy(u)

            # Smoothing of u
            if( general_sigma > 0. ):
                u_sig = self.image_convolve_gaussian(u, general_sigma)

            # Compute gradient of u
            # Normalizing factor for Hessian at scale sigma
            der_u = self.image_der_d(u_sig, hx, hy, hz)
            denom =1.0/( np.sqrt( 1.0 + der_u[:,:,:,0]**2 +\
            der_u[:,:,:,1]**2 + der_u[:,:,:,2]**2 ))

            #---------Compute Diffusivity------------------
            # Second derivatives of u ( smoothed version of f )
            u_der_xx = self.image_der_dd(u_sig, 'xx', hx, hy, hz)
            u_der_yy = self.image_der_dd(u_sig, 'yy', hx, hy, hz)
            u_der_zz = self.image_der_dd(u_sig, 'zz', hx, hy, hz)

            u_der_xy = self.image_der_dd(u_sig, 'xy', hx, hy, hz,\
                        kernel_ab_type = 1)
            u_der_xz = self.image_der_dd(u_sig, 'xz', hx, hy, hz,\
                        kernel_ab_type = 1)
            u_der_yz = self.image_der_dd(u_sig, 'yz', hx, hy, hz,\
                        kernel_ab_type = 1)

            # Copy the values into the 3x3 Hessian matrices
            general_H[:,:,:,0,0] = np.copy(u_der_xx) * denom
            general_H[:,:,:,0,1] = np.copy(u_der_xy) * denom
            general_H[:,:,:,0,2] = np.copy(u_der_xz) * denom
            general_H[:,:,:,1,0] = np.copy(u_der_xy) * denom
            general_H[:,:,:,1,1] = np.copy(u_der_yy) * denom
            general_H[:,:,:,1,2] = np.copy(u_der_yz) * denom
            general_H[:,:,:,2,0] = np.copy(u_der_xz) * denom
            general_H[:,:,:,2,1] = np.copy(u_der_yz) * denom
            general_H[:,:,:,2,2] = np.copy(u_der_zz) * denom

            # Apply smoothing on structure tensor. smooth each channel
            # separately.
            if(rho > 0.):

                general_H[:,:,:,0,0] =\
                self.image_convolve_gaussian(general_H[:,:,:,0,0], rho)
                general_H[:,:,:,0,1] =\
                self.image_convolve_gaussian(general_H[:,:,:,0,1], rho)
                general_H[:,:,:,0,2] =\
                self.image_convolve_gaussian(general_H[:,:,:,0,2], rho)
                general_H[:,:,:,1,0] =\
                self.image_convolve_gaussian(general_H[:,:,:,1,0], rho)
                general_H[:,:,:,1,1] =\
                self.image_convolve_gaussian(general_H[:,:,:,1,1], rho)
                general_H[:,:,:,1,2] =\
                self.image_convolve_gaussian(general_H[:,:,:,1,2], rho)
                general_H[:,:,:,2,0] =\
                self.image_convolve_gaussian(general_H[:,:,:,2,0], rho)
                general_H[:,:,:,2,1] =\
                self.image_convolve_gaussian(general_H[:,:,:,2,1], rho)
                general_H[:,:,:,2,2] =\
                self.image_convolve_gaussian(general_H[:,:,:,2,2], rho)

            # Allocate memory for Vesselness measure for different scales and
            # Hessian
            V = np.empty(h*w*d*num_scales).reshape(num_scales, d, h, w)
            G = np.empty(num_scales*h*w*d*3).reshape(num_scales, d, h, w, 3)
            H = np.empty(num_scales*h*w*d*9).reshape(num_scales, d, h, w, 3, 3)
            Evls = np.empty(num_scales*h*w*d*3).reshape(num_scales, d, h, w, 3)
            Evcs = np.empty(num_scales*h*w*d*9).reshape(num_scales, d, h, w,\
                3, 3)
#             # Only for visualization
#             HR = np.empty(h*w*num_scales).reshape(num_scales, h, w)
#             HL = np.empty(h*w*num_scales).reshape(num_scales, h, w)


            # Compute Hessian for all the scales
            k = 0
            for sigma in sigma_list:

                sig_sqr = sigma**2
                if( sigma == 0 ):
                    sig_sqr = 1.0

                # Smoothing of u
                u_sig = self.image_convolve_gaussian(u, sigma)
                # Compute gradient of u
                der_u = self.image_der_d(u_sig, hx, hy, hz)

                # Normalizing factor for Hessian at scale sigma
                denom=1.0/( np.sqrt( 1.0 + der_u[:,:,:,0]**2 +\
                der_u[:,:,:,1]**2 + der_u[:,:,:,2]**2 ))

                # Second derivatives of u ( smoothed version of f )
                u_der_xx = self.image_der_dd(u_sig, 'xx', hx, hy, hz)
                u_der_yy = self.image_der_dd(u_sig, 'yy', hx, hy, hz)
                u_der_zz = self.image_der_dd(u_sig, 'zz', hx, hy, hz)

                u_der_xy = self.image_der_dd(u_sig, 'xy', hx, hy, hz,\
                            kernel_ab_type = 1)
                u_der_xz = self.image_der_dd(u_sig, 'xz', hx, hy, hz,\
                            kernel_ab_type = 1)
                u_der_yz = self.image_der_dd(u_sig, 'yz', hx, hy, hz,\
                            kernel_ab_type = 1)

                # Copy the values into the 3x3 Hessian matrices
                H[k,:,:,:,0,0] = np.copy(u_der_xx) * denom
                H[k,:,:,:,0,1] = np.copy(u_der_xy) * denom
                H[k,:,:,:,0,2] = np.copy(u_der_xz) * denom
                H[k,:,:,:,1,0] = np.copy(u_der_xy) * denom
                H[k,:,:,:,1,1] = np.copy(u_der_yy) * denom
                H[k,:,:,:,1,2] = np.copy(u_der_yz) * denom
                H[k,:,:,:,2,0] = np.copy(u_der_xz) * denom
                H[k,:,:,:,2,1] = np.copy(u_der_yz) * denom
                H[k,:,:,:,2,2] = np.copy(u_der_zz) * denom

                G[k,:,:,:,:] = np.copy(der_u)

                # Apply smoothing on structure tensor. smooth each channel
                # separately.
                if(rho > 0.):
                    H[k,:,:,:,0,0] =\
                    self.image_convolve_gaussian(H[k,:,:,:,0,0], rho)
                    H[k,:,:,:,0,1] =\
                    self.image_convolve_gaussian(H[k,:,:,:,0,1], rho)
                    H[k,:,:,:,0,2] =\
                    self.image_convolve_gaussian(H[k,:,:,:,0,2], rho)
                    H[k,:,:,:,1,0] =\
                    self.image_convolve_gaussian(H[k,:,:,:,1,0], rho)
                    H[k,:,:,:,1,1] =\
                    self.image_convolve_gaussian(H[k,:,:,:,1,1], rho)
                    H[k,:,:,:,1,2] =\
                    self.image_convolve_gaussian(H[k,:,:,:,1,2], rho)
                    H[k,:,:,:,2,0] =\
                    self.image_convolve_gaussian(H[k,:,:,:,2,0], rho)
                    H[k,:,:,:,2,1] =\
                    self.image_convolve_gaussian(H[k,:,:,:,2,1], rho)
                    H[k,:,:,:,2,2] =\
                    self.image_convolve_gaussian(H[k,:,:,:,2,2], rho)

                    G[k,:,:,:,0] =\
                    self.image_convolve_gaussian(G[k,:,:,:,0], rho)
                    G[k,:,:,:,1] =\
                    self.image_convolve_gaussian(G[k,:,:,:,1], rho)

#                 hess = np.empty(h*w*4).reshape(h, w, 2, 2)
#                 hess[:,:,0,0] = np.copy(u_der_xx) * sig_sqr
#                 hess[:,:,1,1] = np.copy(u_der_yy) * sig_sqr
#                 hess[:,:,0,1] = hess[:,:,1,0] = np.copy(u_der_xy) * sig_sqr

                hess = H[k,:,:,:,:,:]*sig_sqr

                evls, evcs = np.linalg.eigh(hess)

                # Reshape the array for finding the max norm of hessians.
                # To convert (i,j,k) index to linear index
                # (k*row*col + i*col + j) can be used.
                one_d_evls = evls.reshape(w*h*d,3)

                if( c_var == [] ):
                    # Find the maximum norm of Hessians to compute c
                    max_norm = max(np.linalg.norm(one_d_evls, ord = 2,\
                        axis = 1))
                    c = max_norm * 0.5
                else:
                    c = c_var[k]

                eps = 1.e-20

                # -- Sort evals & evecs based on evalue modulus increasingly --
                Evls[k,:,:,:,:]   = np.copy(evls)
                Evcs[k,:,:,:,:,:] = np.copy(evcs)

                sorting_idx = np.argsort(np.abs(evls))

                # Linearize the array of indices
                idx         = sorting_idx.flatten() +\
                    np.repeat(np.arange(d*w*h) * 3, 3)
                lin_evls    = Evls[k,:,:,:,:].flatten()[idx]
                Evls[k,:,:,:,:] = lin_evls.reshape(( d, h, w, 3 ))

                # Sort the eigenvectors according to eigenvalues
                evcs_tmp    = np.transpose(Evcs[k,:,:,:,:,:],\
                    axes = (0,1,2,4,3))
                lin_evcs    = evcs_tmp.reshape( ( d*w*h*3, 3 ) )[idx,:]
                Evcs[k,:,:,:,:,:] = np.transpose(\
                    lin_evcs.reshape((d,h,w,3,3)), axes = (0,1,2,4,3))
                # ----- End of eval/evec sorting ----


                RB = abs(Evls[k,:,:,:,0]) / (np.sqrt(abs(Evls[k,:,:,:,1] \
                    * Evls[k,:,:,:,2])+eps))
                RA = abs(Evls[k,:,:,:,1]) / (abs(Evls[k,:,:,:,2])+eps)
                S  = np.sqrt((Evls[k,:,:,:,0]**2 + Evls[k,:,:,:,1]**2 \
                    + Evls[k,:,:,:,2]**2))

                if( detect_planes ):
                    v_measure = (np.exp(-1.0*(RB**2/2.0*beta_var**2))) \
                                *(1.0-np.exp(-1.0*(S**2/(2.0*c**2))))
                else:
                    v_measure = (1.0-np.exp(-1.0*(RA**2/(2.0*alpha_var**2))))\
                                * (np.exp(-1.0*(RB**2/2.0*beta_var**2)))\
                                * (1.0-np.exp(-1.0*(S**2/(2.0*c**2))))

                if( crease_type == "r" ):
                    c_1 = Evls[k,:,:,:,1] <= 0.0
                    c_2 = Evls[k,:,:,:,2] <= 0.0
                    # If(evl1 > 0 or evl2 > 0): V = 0.0 else V = v_measure
                    # Here "*" is boolean operator
                    V[k,:,:,:] = c_1 * c_2 * v_measure

                elif( crease_type == "v" ):
                    c_1 = Evls[k,:,:,:,1] >= 0.0
                    c_2 = Evls[k,:,:,:,2] >= 0.0
                    # If(evl1 < 0 or evl2 < 0): V = 0.0 else V = v_measure
                    # Here "*" is boolean operator
                    V[k,:,:,:] = (c_1 * c_2) * v_measure

                elif ( crease_type == "rv" ):
                    V[k,:,:,:] = v_measure
                else:
                    raise Exception("Error in function vesselness_measure():\
                    Invalid crease_type. Crease_type must be from the \
                    set {'r','v','rv'}")

                k += 1


            scale_arr = np.empty(h*w*d).reshape(d,h,w)
            scale_arr.fill(0)

            max_idx = np.argmax(V, axis = 0)
            max_val = np.max(V, axis = 0)

            scale_l = np.asarray(sigma_list)
            mask    = max_val >= theta_var
            max_scl = scale_l[max_idx] * mask
            max_idx = max_idx * mask

            # A set of point indices that lie on the vessels
            on_vess_idx = np.arange(d*h*w)
            on_vess_idx = set(on_vess_idx * mask.flatten())
            if (not mask[0,0,0]):
                on_vess_idx.discard(0)

            scale = max_scl
            if( postprocess ):
                scale = self.post_filter_scales(max_scl, scale_l,\
                        on_vess_idx, max_idx, Evcs)

            # Update max_idx according to post_filter_scales
            for k in range(len(sigma_list)):
                max_idx[np.where(scale==scale_l[k])] = k


            # Extract only eigenvalues and eigenvectors of the max_scale for
            # each voxel in a dxhxw grid from Evls and Evcs
            mask_base_evl = np.repeat(max_idx.flatten(),3).\
                reshape((d, h, w, 3))
            mask_base_evc = np.repeat(max_idx.flatten(),9).\
                reshape((d, h, w, 3, 3))


            e_val = np.empty((d, h, w, 3))
            e_vec = np.empty((d, h, w, 3, 3))
            e_val.fill(0.0)
            e_vec.fill(0.0)


            for k in range(len(sigma_list)):

                mask = mask_base_evl == k
                e_val = e_val + (mask * Evls[k,:,:,:,:])

                mask = mask_base_evc == k
                e_vec = e_vec + (mask * Evcs[k,:,:,:,:,:])
        else:
            raise Exception('In vesselness_measure function: Invalid\
            dimensionality of file. Input file must be either 2D or 3D.')

        return scale, e_val, e_vec, on_vess_idx


###############################################################################
#-------------------- Vesselness Filter by Jerman et al. ---------------------#
###############################################################################
    def Jerman_Vesselness_measure(self, f, sigma_list, rho,\
        theta_var, tau, crease_type = "r",\
        postprocess = True, detect_planes=False, hx = 1.0, hy = 1.0, hz = 1.0,\
        RETURN_HESSIAN = False):
        """
        -------Vesselness Filter by Jerman et al.-------

        f : 2D input image
        sigma_list : a set of possible scales in the image.
        E.g. {0.5, 1.0, 1.5, 2.0}
        theta_var : A float between 0 and 1. Used for segmenting the vessels
            from backgournd according to the computed vesselness measurements
        crease_type : "r" for ridge detection, "v" for vessel detection,
            "rv" for both.

        """
        dim = f.shape

        # To return
        scale = []
        e_val = []
        e_vec = []

    

        # Vesselness measure for 2D Images
        if( len(dim) == 2 ):

            # height, width
            h, w = dim

            u = np.copy(f)

            num_scales = len(sigma_list)
          
            # Allocate memory for Vesselness measure for different scales and
            # Hessian2
            V = np.empty(h*w*num_scales).reshape(num_scales, h, w)
            G = np.empty(num_scales*h*w*2).reshape(num_scales,  h, w, 2)
            H = np.empty(num_scales*h*w*4).reshape(num_scales, h, w, 2, 2)
            Evls = np.empty(num_scales*h*w*2).reshape(num_scales,  h, w, 2)
            Evls_n = np.empty(num_scales*h*w*2).reshape(num_scales,  h, w, 2)
            Evcs = np.empty(num_scales*h*w*4).reshape(num_scales,  h, w,\
                2, 2)
            lambda_rho = np.empty(num_scales*h*w).reshape(num_scales,  h, w)
            H_mts = np.empty(num_scales*h*w*4).reshape(num_scales,h, w, 2, 2)
           
#             # Only for visualization
#             HR = np.empty(h*w*num_scales).reshape(num_scales, h, w)
#             HL = np.empty(h*w*num_scales).reshape(num_scales, h, w)


            # Compute Hessian for all the scales
            k = 0
            for sigma in sigma_list:


                sig_sqr = (sigma**2)

                # Smoothing of u
                u_sig = self.image_convolve_gaussian(u, sigma)
                # Compute gradient of u
                der_u = self.image_der_d(u_sig, hx, hy)

                # Normalizing factor for Hessian at scale sigma a la Canero
                denom=1.0/( np.sqrt( 1.0 + der_u[:,:,0]**2 + der_u[:,:,1]**2 ))
#                denom=sig_sqr/2.0

                # Second derivatives of u ( smoothed version of f )
                u_der_xx = self.image_der_dd(u_sig, 'xx', hx, hy)
                u_der_yy = self.image_der_dd(u_sig, 'yy', hx, hy)
                u_der_xy = self.image_der_dd(u_sig, 'xy', hx, hy,\
                            kernel_ab_type = 1)


                # Copy the values into the 3x3 Hessian matrices
                H[k,:,:,0,0] = np.copy(u_der_xx) * denom
                H[k,:,:,0,1] = np.copy(u_der_xy) * denom
                H[k,:,:,1,0] = np.copy(u_der_xy) * denom
                H[k,:,:,1,1] = np.copy(u_der_yy) * denom
                
                if(rho > 0.):
                    H[k,:,:,0,0] =\
                    self.image_convolve_gaussian(H[k,:,:,0,0], rho)
                    H[k,:,:,0,1] =\
                    self.image_convolve_gaussian(H[k,:,:,0,1], rho)
                    H[k,:,:,1,0] =\
                    self.image_convolve_gaussian(H[k,:,:,1,0], rho)
                    H[k,:,:,1,1] =\
                    self.image_convolve_gaussian(H[k,:,:,1,1], rho)

                G[k,:,:,:] = np.copy(der_u)
                
                hess = np.empty(h*w*4).reshape(h, w, 2, 2)
                hess[:,:,0,0] = np.copy(u_der_xx) * sig_sqr
                hess[:,:,1,1] = np.copy(u_der_yy) * sig_sqr
                hess[:,:,0,1] = hess[:,:,1,0] = np.copy(u_der_xy) * sig_sqr

                #hess = np.copy(H[k,:,:,:,:])*sig_sqr
                H_mts[k,:,:,:,:] = np.copy(hess)
                evls, evcs = np.linalg.eigh(hess)


        

                # -- Sort evals & evecs based on evalue modulus increasingly --
                Evls[k,:,:,:]   = np.copy(evls)
                Evcs[k,:,:,:,:] = np.copy(evcs)

                sorting_idx = np.argsort(np.abs(evls))

                # Linearize the array of indices
                idx         = sorting_idx.flatten() +\
                    np.repeat(np.arange(w*h) * 2, 2)
                lin_evls    = Evls[k,:,:,:].flatten()[idx]
                Evls[k,:,:,:] = lin_evls.reshape((  h, w, 2 ))

                # Sort the eigenvectors according to eigenvalues
                evcs_tmp    = np.transpose(Evcs[k,:,:,:,:],\
                    axes = (0,1,3,2))
                lin_evcs    = evcs_tmp.reshape( ( w*h*2, 2 ) )[idx,:]
                Evcs[k,:,:,:,:] = np.transpose(\
                    lin_evcs.reshape((h,w,2,2)), axes = (0,1,3,2))
                # ----- End of eval/evec sorting ----
                Evls_n[k,:,:,:] = -1.0 * Evls[k,:,:,:]
                                
                # lambda_rho = ambda_2
                lambda_rho[k,:,:] = Evls_n[k,:,:,1]
                max_lambda = np.max(lambda_rho[k,:,:])
                
                mask_1 = lambda_rho[k,:,:] > 0.0
                mask_2 = mask_1 * ( lambda_rho[k,:,:] <= (tau * max_lambda) )
                lambda_rho[k,:,:] = mask_1 * np.logical_not(mask_2) * lambda_rho[k,:,:] +\
                             mask_2 * tau * max_lambda
                                 
                
                tmp = Evls_n[k,:,:,1] + lambda_rho[k,:,:]
                tmp[np.where(tmp==0.0)] = 1.e-20
                
                tmp = ((3.0/tmp)**3) * (Evls_n[k,:,:,1]**2) * (lambda_rho[k,:,:]-\
                        Evls_n[k,:,:,1])
                sub_m1 = Evls_n[k,:,:,1] <= 0.0
                sub_m2 = lambda_rho[k,:,:] <= 0.0
                sub_m3 = Evls_n[k,:,:,1] >= (lambda_rho[k,:,:]/2.0)
                mask_1 = np.logical_or(sub_m1, sub_m2)
                mask_2 = sub_m3 * np.logical_not(sub_m1)
                ones = np.ones((h,w))
                V[k,:,:] = tmp * np.logical_not(mask_1) * np.logical_not(mask_2)+\
                            ones * mask_2
                
               # plot.show_image(lambda_rho[k,:,:])
               # V[k,:,:]=np.copy(VV[k,:,:])
                H_mts[k,:,:,:,:] = np.copy(H[k,:,:,:,:])
          

                k += 1
               #       time.time()-s_t, "sec"


            scale_arr = np.empty(h*w).reshape(h,w)
            scale_arr.fill(0)
         
            max_idx = np.argmax(V, axis = 0)
            max_val = np.max(V, axis = 0)
           # plot.multi_plot_with_extermum_vlines( u_array=[V[:,ii,jj], V[:,ii1,jj1],V[:,ii2,jj2]],\
           # r=1, c=3, u_ext_array = [V[:,ii,jj], V[:,ii1,jj1],V[:,ii2,jj2]],\
           # base_u =[V[:,ii,jj]], base_ext = [V[:,ii,jj]],show_axis_values =True)
            scale_l = np.asarray(sigma_list)
            mask_on = max_val >= theta_var
            max_scl = scale_l[max_idx] * mask_on
            max_idx = max_idx * mask_on

            # A set of point indices that lie on the vessels
            on_vess_idx = np.arange(h*w)
            on_vess_idx = set(on_vess_idx * mask_on.flatten())
            if (not mask_on[0,0]):
                on_vess_idx.discard(0)

          #  plot.show_image_with_scales(max_scl)

            scale = max_scl
            if( postprocess ):
                #scale = self.post_filter_scales(max_scl, scale_l,\
                #        on_vess_idx, max_idx, Evcs)
                self.post_filter_scales_2( scale_l,max_scl, H_mts)
            #plot.show_image_with_scales(max_scl)
            # Update max_idx according to post_filter_scales
            for k in range(len(sigma_list)):
                max_idx[np.where(scale==scale_l[k])] = k

            
            # Extract only eigenvalues and eigenvectors of the max_scale for
            # each voxel in a dxhxw grid from Evls and Evcs
            
            mask_base_evc = np.repeat(max_idx.flatten(),4).\
                reshape(( h, w, 2, 2))

            h_mts = np.empty((h, w, 2, 2))
            h_mts.fill(0.0)
            for k in range(len(sigma_list)):
                
                mask = mask_base_evc == k
                h_mts = h_mts + (mask * H_mts[k,:,:,:,:])

            e_val = np.empty((h, w, 2))
            e_vec = np.empty((h, w, 2, 2))
            e_val.fill(0.0)
            e_vec.fill(0.0)

            # smooth *after* scale selection to remove discontinuities
            # that can be introduced by scale selection
            if(rho > 0.):
                h_mts[:,:,0,0] =\
                    self.image_convolve_gaussian(h_mts[:,:,0,0], rho)
                h_mts[:,:,0,1] =\
                    self.image_convolve_gaussian(h_mts[:,:,0,1], rho)
                h_mts[:,:,1,0] =\
                    self.image_convolve_gaussian(h_mts[:,:,1,0], rho)
                h_mts[:,:,1,1] =\
                    self.image_convolve_gaussian(h_mts[:,:,1,1], rho)
            
            # compute evals/evecs of regularized result
            evls, evcs = np.linalg.eigh(h_mts[:,:,:,:])
            
            # -- Sort evals & evecs based on evalue modulus increasingly --
            e_val[:,:,:]   = np.copy(evls)
            e_vec[:,:,:,:] = np.copy(evcs)
            
            sorting_idx = np.argsort(np.abs(e_val))
            
            # Linearize the array of indices
            idx         = sorting_idx.flatten() +\
                np.repeat(np.arange(w*h) * 2, 2)
            lin_evls    = e_val[:,:,:].flatten()[idx]
            e_val[:,:,:] = lin_evls.reshape((  h, w, 2 ))
            
            # Sort the eigenvectors according to eigenvalues
            evcs_tmp    = np.transpose(e_vec[:,:,:,:],\
                                           axes = (0,1,3,2))
            lin_evcs    = evcs_tmp.reshape( ( w*h*2, 2 ) )[idx,:]
            e_vec[:,:,:,:] = np.transpose(\
                lin_evcs.reshape((h,w,2,2)), axes = (0,1,3,2))
            
           # plot.show_image(np.abs(e_val[:,:,0]))
          #  plot.draw_eigen_vector_field_evecs(f, e_vec, on_vess_idx, arrow_scale = 1.0)

           # plot.show_image(np.abs(e_val[:,:,1]))


            if( postprocess ):
                if( RETURN_HESSIAN ):
                    return scale, e_val, e_vec, h_mts, on_vess_idx
                return scale, e_val, e_vec, on_vess_idx
            else:
                if( RETURN_HESSIAN ):
                    return scale, e_val, e_vec, h_mts, mask_on
                return scale, e_val, e_vec, mask_on
        elif ( len(dim) == 3 ):
            print("3D version of Jerman vesselness measure is to be implemented.")
        else:
            raise Exception('In vesselness_measure function: Invalid\
            dimensionality of file. Input file must be either 2D or 3D.')

        return scale, e_val, e_vec, on_vess_idx
        
###############################################################################
#---                                AFOD                                 -----#
###############################################################################
    def fourth_order_anisotropic_diffusion_filter(self, f, hx, hy, sigma, rho,\
           var_lambda, var_lambda_valley, tau, ignore_ridge, ignore_valley,\
           symm = True, smooth_max_crease = False):

        h, w = f.shape

        lambda_sqr = var_lambda*var_lambda;
        u = np.copy(f)
        # Presmoothing of u
        if(sigma > 0.0):
            u = self.image_convolve_gaussian(u, sigma)

        # Second derivatives of f
        image_der_xx = self.image_der_dd(f, 'xx')
        image_der_yy = self.image_der_dd(f, 'yy')
        if(symm):
            image_der_xy = self.image_second_derivative(f, 'xy')
            image_der_yx = image_der_xy
        else:
            image_der_xy = self.image_der_dd(f, 'xy')
            image_der_yx = self.image_der_dd(f, 'yx')

        #---------Compute Diffusivity------------------
        # Second derivatives of u ( smoothed version of f )
        u_der_xx = self.image_der_dd(u, 'xx')
        u_der_yy = self.image_der_dd(u, 'yy')
        if(symm):
            u_der_xy = self.image_second_derivative(u, 'xy')
            u_der_yx = u_der_xy
        else:
            u_der_xy = self.image_der_dd(u, 'xy')
            u_der_yx = self.image_der_dd(u, 'yx')
        der_u = self.image_der_d(u)
        denom=1.0/(np.sqrt(1.0+der_u[:,:,0]**2+der_u[:,:,1]**2))

        hess = np.dstack((u_der_xx*denom, u_der_xy*denom, u_der_yx*denom,\
                u_der_yy*denom))
        hess = hess.reshape((h, w, 2, 2))

        if(symm):
            hess_evl, hess_evc = np.linalg.eigh(hess)
        else:
            hess_evl, hess_evc = np.linalg.eig(hess)

        isqrt2= 1.0/sqrt(2.0)
        ridge_strength = np.empty(w)
        for i in range(h):
            for j in range(w):
                # Order eigenvalues with decreasing modulus.
                self.sort_based_on_evalues_modulus(hess_evl[i,j],\
                        hess_evc[i,j])

                hess_evl1 = hess_evl[i,j,0]
                hess_evl2 = hess_evl[i,j,1]
                hess_evc1 = hess_evc[i,j,:,0]
                hess_evc2 = hess_evc[i,j,:,1]
                if( i == 2 ):
                    if( hess_evl1 <= 0 ):
                        ridge_strength[j] = hess_evl1
                    else:
                        ridge_strength[j] = 0
                evals = np.empty(4)
                evecs = np.empty(16).reshape(4,2,2)
                # Principal curvature direction
                # hess_evl1 > 0 means, in hess_evc1 direction we are
                # orthogonal to a valley, then smooth in this direction
                if(ignore_valley and hess_evl1 > 0):
                    evals[0] = 1.0
#                     evals[0] = 1.0/(1.0+((hess_evl1*hess_evl1)/lambda_sqr))
#                     evals[0] = 1.0/(1.0+((hess_evl1*hess_evl1)/\
                      #(var_lambda_valley**2)))
                # hess_evl1 < 0 means, in hess_evc1 direction we are orthogonal
                # to a ridge, then smooth in this direction
                elif(ignore_ridge and hess_evl1 < 0):
                    evals[0] = 1.0


                else:
                    evals[0] = 1.0/(1.0+((hess_evl1*hess_evl1)/lambda_sqr))

                evecs[0,:,:] = np.outer(hess_evc1, hess_evc1)

                # Smaller curvature direction
                # hess_evl2 > 0 means, in hess_evc2 directopn we are orthogonal
                # to a valley, and if ignore_valley ==1
                # then smooth maximally in this directio
                if(ignore_valley and hess_evl2 > 0):
                    evals[1] = 1.0
                elif(ignore_ridge and hess_evl2 < 0):
                    evals[1] = 1.0
                else:
                    evals[1] = 1.0/(1.0+((hess_evl2*hess_evl2)/lambda_sqr))

                evecs[1,:,:] = np.outer(hess_evc2, hess_evc2)
                evals[2] = 0.5 * (evals[0] + evals[1])
                evecs[2,:,:] = isqrt2 * (np.outer(hess_evc1, hess_evc2) +\
                                np.outer(hess_evc2, hess_evc1))

                # Antisymmetric tangent -> set to zero (shouldn't matter, too)
                evals[3] = 0.0
                evecs[3,:,:] = isqrt2 * (np.outer(hess_evc1, hess_evc2) -\
                                np.outer(hess_evc2, hess_evc1))

                evecs = evecs.reshape(4,4)

                evals_matrix = np.diag(evals)
                local_st = self.matrix_from_evals_evecs(evals_matrix, evecs.T)

                #Matrix-Vector multiplication
                local_der_xx = local_st[0,0] * image_der_xx[i,j] +\
                               local_st[0,1] * image_der_xy[i,j] +\
                               local_st[0,2] * image_der_yx[i,j] +\
                               local_st[0,3] * image_der_yy[i,j];
                local_der_xy = local_st[1,0] * image_der_xx[i,j] +\
                               local_st[1,1] * image_der_xy[i,j] +\
                               local_st[1,2] * image_der_yx[i,j] +\
                               local_st[1,3] * image_der_yy[i,j];
                local_der_yx = local_st[2,0] * image_der_xx[i,j] +\
                               local_st[2,1] * image_der_xy[i,j] +\
                               local_st[2,2] * image_der_yx[i,j] +\
                               local_st[2,3] * image_der_yy[i,j];
                local_der_yy = local_st[3,0] * image_der_xx[i,j] +\
                               local_st[3,1] * image_der_xy[i,j] +\
                               local_st[3,2] * image_der_yx[i,j] +\
                               local_st[3,3] * image_der_yy[i,j];

                #save the result
                image_der_xx[i,j] = local_der_xx
                image_der_xy[i,j] = local_der_xy
                image_der_yx[i,j] = local_der_yx
                image_der_yy[i,j] = local_der_yy

        incr_xx = self.image_second_derivative(image_der_xx, 'xx')
        incr_yy = self.image_second_derivative(image_der_yy, 'yy')
        incr_xy = self.image_second_derivative(image_der_xy, 'xy')
        incr_yx = self.image_second_derivative(image_der_yx, 'yx')
        u = f - tau * (incr_xx + incr_yy + incr_xy + incr_yx)
        return u
    def map_degree_to_pixel_dir_struct_tensor(self, degree):
        if( degree < 22.5 or degree >= 337.5 ):
            i = 0
            j =  1
        elif( degree < 67.5 and degree >= 22.5 ):
            i = -1
            j =  1
        elif( degree < 112.5 and degree >= 67.5 ):
            i = -1
            j = 0
        elif( degree < 157.5 and degree >= 112.5 ):
            i = -1
            j = -1
        elif( degree < 202.5 and degree >= 157.5 ):
            i =  0
            j = -1
        elif( degree < 247.5 and degree >= 202.5 ):
            i =  1
            j = -1
        elif( degree < 292.5 and degree >= 247.5 ):
            i =  1
            j =  0
        elif( degree < 337.5 and degree >= 292.5 ):
            i = 1
            j = 1
        return i, j

    def map_degree_to_pixel_dir(self, degree):
        i = 0
        j = 0
        if( degree < 22.5 or degree >= 337.5 ):
            j =  1
        elif( degree < 67.5 and degree >= 22.5 ):
            i =  1
            j =  1
        elif( degree < 112.5 and degree >= 67.5 ):
            i =  1
        elif( degree < 157.5 and degree >= 112.5 ):
            i =  1
            j = -1
        elif( degree < 202.5 and degree >= 157.5 ):
            j = -1
        elif( degree < 247.5 and degree >= 202.5 ):
            i = -1
            j = -1
        elif( degree < 292.5 and degree >= 247.5 ):
            i = -1
        elif( degree < 337.5 and degree >= 292.5 ):
            i = -1
            j =  1
        return i, j

    def find_pixels_in_max_evector_dir_struct_tensor( self, i, j, scale_list,\
            scale_img, status_img, G ):
        h, w = scale_img.shape
        status = np.empty(h*w).reshape(h,w)
        status.fill(0)

        pixels = np.asarray([[i,j,scale_img[i,j]]])
        pixels_size = 1
        counter = 0
        status[i,j] = 1

        while counter < pixels_size :
            p = pixels[counter]
            curr_i = p[0]
            curr_j = p[1]
            s_indx = np.where(scale_list == p[2])[0][0]
            curr_g = G[s_indx, curr_i, curr_j,:]
            curr_h = np.outer(curr_g, curr_g)
            evl, evc = np.linalg.eigh(curr_h)
            self.sort_based_on_evalues_modulus(evl, evc)
            direction = evc[:,0]

            if( direction[0] == 0.0 ):
                # Multiply with dir_r-y to consider sign of the infinity
                m_1 = np.inf * direction[1]
            else:
                m_1 = direction[1]/direction[0]

            deg_1 = np.arctan(m_1) * (180.0/np.pi)

            if( deg_1 <0 ):
                deg_1 = 360.0 + deg_1

            deg_2 = deg_1 + 180 if ( deg_1 + 180 < 360 ) else (deg_1 - 180)

            di_1, dj_1 = self.map_degree_to_pixel_dir_struct_tensor( deg_1 )
            di_2, dj_2 = self.map_degree_to_pixel_dir_struct_tensor( deg_2 )

            i_new1 = np.min([np.max([0, curr_i+di_1]), h-1])
            j_new1 = np.min([np.max([0, curr_j+dj_1]), w-1])
            i_new2 = np.min([np.max([0, curr_i+di_2]), h-1])
            j_new2 = np.min([np.max([0, curr_j+dj_2]), w-1])
            s1 = scale_img[i_new1, j_new1]
            s2 = scale_img[i_new2, j_new2]
            if( s1 != 0 and status[i_new1, j_new1] == 0 and status_img[i_new1,\
                    j_new1] == 0):# and status[i_new1, j_new1] == 0 ):
                pixels = np.append( pixels, [[i_new1, j_new1,s1]], axis = 0)
                status[i_new1, j_new1] = 1
            if( s2 != 0 and status[i_new2, j_new2] == 0 and status_img[i_new2,\
                    j_new2] == 0):
                pixels = np.append( pixels, [[i_new2, j_new2,s2]], axis = 0)
                status[i_new2, j_new2] = 1
            pixels_size = pixels.shape[0]
            counter += 1

        return pixels

    def post_filter_scales_struct_tensor( self, scale_list, scale_img, G ):
        h, w = scale_img.shape
        status_img = np.empty(h*w).reshape(h,w)
        # Zero for each pixels means it has not been seen, 1 means seen
        status_img.fill(0)
        for i in range(h):
            for j in range(w):
                if( scale_img[i,j] > 0.0 and status_img[i,j] == 0 ):
                    # contains an array of {[x,y,scale]} which is the position
                    # and scale of the pixels lying in direction
                    # of the eigen vector coresponding to the largest eigen
                    # value modulus

                    ridge_perpendicular_cut =\
                        self.find_pixels_in_max_evector_dir_struct_tensor( i,\
                            j, scale_list, scale_img, status_img, G )

                    max_scale = np.max(ridge_perpendicular_cut[:,2])
                    scale_array = ridge_perpendicular_cut[:,2]
                    if( len(scale_list) > 1 ):
                        frequency_of_scales = np.histogram(scale_array,\
                            bins=scale_list)
                        max_index = np.argmax(frequency_of_scales[0])
                        freq_scale = np.max(frequency_of_scales[1][max_index])
                    else:
                        freq_scale = max_scale
                    #decision based on average
                    avg_scale = np.average(scale_array)
                    if( np.abs(max_scale-avg_scale) <\
                            np.abs(freq_scale-avg_scale)):
                        scale = max_scale
                    else:
                        scale = freq_scale
                    for p in ridge_perpendicular_cut:
                        if(status_img[p[0],p[1]] == 0):
                            scale_img[p[0],p[1]]  = scale
                            status_img[p[0],p[1]] = 1

        return

    def find_pixels_in_max_evector_dir_2( self, i, j, evec, scale_img ):
        h, w = scale_img.shape
        status = np.empty(h*w).reshape(h,w)
        status.fill(0)

        pixels = np.asarray([[i,j,scale_img[i,j]]])
        pixels_size = 1
        counter = 0
        status[i,j] = 1
        while counter < pixels_size :
            p = pixels[counter]
            curr_i = p[0]
            curr_j = p[1]

            direction = evec

            if( direction[0] == 0.0 ):
                #multiply with dir_r-y to consider sign of the infinity
                m_1 = np.inf * direction[1]
            else:
                m_1 = direction[1]/direction[0]

            deg_1 = np.arctan(m_1) * (180.0/np.pi)

            if( deg_1 <0 ):
                deg_1 = 360.0 + deg_1

            deg_2 = deg_1 + 180 if ( deg_1 + 180 < 360 ) else (deg_1 - 180)

            di_1, dj_1 = self.map_degree_to_pixel_dir( deg_1 )
            di_2, dj_2 = self.map_degree_to_pixel_dir( deg_2 )

            i_new1 = int(np.min([np.max([0, curr_i+di_1]), h-1]))
            j_new1 = int(np.min([np.max([0, curr_j+dj_1]), w-1]))
            i_new2 = int(np.min([np.max([0, curr_i+di_2]), h-1]))
            j_new2 = int(np.min([np.max([0, curr_j+dj_2]), w-1]))

            s1 = scale_img[i_new1, j_new1]
            s2 = scale_img[i_new2, j_new2]

            if( s1 != 0 and status[i_new1, j_new1] == 0):
                pixels = np.append( pixels, [[i_new1, j_new1,s1]], axis = 0)
                status[i_new1, j_new1] = 1
            if( s2 != 0 and status[i_new2, j_new2] == 0):
                pixels = np.append( pixels, [[i_new2, j_new2,s2]], axis = 0)
                status[i_new2, j_new2] = 1
            pixels_size = pixels.shape[0]
            counter += 1

        return pixels

    def find_pixels_in_max_evector_dir( self, i, j, scale_list, scale_img,\
            status_img, H ):
        h, w = scale_img.shape
        status = np.empty(h*w).reshape(h,w)
        status.fill(0)

        pixels = np.asarray([[i,j,scale_img[i,j]]])
        pixels_size = 1
        counter = 0
        status[i,j] = 1
        while counter < pixels_size :
            p = pixels[counter]
            curr_i = p[0]
            curr_j = p[1]
            s_indx = np.where(scale_list == p[2])[0][0]
            curr_h = H[int(s_indx), int(curr_i), int(curr_j),:,:]

            evl, evc = np.linalg.eigh(curr_h)
            self.sort_based_on_evalues_modulus(evl, evc)
            direction = evc[:,0]
            if( direction[0] == 0.0 ):
                # multiply with dir_r-y to consider sign of the infinity
                m_1 = np.inf * direction[1]
            else:
                m_1 = direction[1]/direction[0]

            deg_1 = np.arctan(m_1) * (180.0/np.pi)

            if( deg_1 <0 ):
                deg_1 = 360.0 + deg_1

            deg_2 = deg_1 + 180 if ( deg_1 + 180 < 360 ) else (deg_1 - 180)

            di_1, dj_1 = self.map_degree_to_pixel_dir( deg_1 )
            di_2, dj_2 = self.map_degree_to_pixel_dir( deg_2 )

            i_new1 = np.min([np.max([0, curr_i+di_1]), h-1])
            j_new1 = np.min([np.max([0, curr_j+dj_1]), w-1])
            i_new2 = np.min([np.max([0, curr_i+di_2]), h-1])
            j_new2 = np.min([np.max([0, curr_j+dj_2]), w-1])
            s1 = scale_img[int(i_new1), int(j_new1)]
            s2 = scale_img[int(i_new2), int(j_new2)]
            if( s1 != 0 and status[int(i_new1), int(j_new1)] == 0 and\
                    status_img[int(i_new1),\
                    int(j_new1)] == 0):# and status[i_new1, j_new1] == 0 ):
                pixels = np.append( pixels, [[i_new1, j_new1,s1]], axis = 0)
                status[int(i_new1), int(j_new1)] = 1
            if( s2 != 0 and status[int(i_new2), int(j_new2)] == 0 and\
                    status_img[int(i_new2),\
                    int(j_new2)] == 0):
                pixels = np.append( pixels, [[i_new2, j_new2,s2]], axis = 0)
                status[int(i_new2), int(j_new2)] = 1
            pixels_size = pixels.shape[0]
            counter += 1

        return pixels

    def find_closest_value(self, list, value):
        min_indx = -1
        min_diff = np.inf

        n = len(list)
        for i in range(n):
            v = list[i]
            diff  = abs(value-v)
            if ( diff < min_diff ):
                min_diff = diff
                min_indx = i
        if(min_indx >= 0 and min_diff !=0 and min_indx < n-1):
            return list[min_indx+1]
        elif(min_indx >= 0):
            return list[min_indx]
        else:
            return value

    def post_filter_scales_2( self, scale_list, scale_img, H ):
        h, w = scale_img.shape
        status_img = np.empty(h*w).reshape(h,w)
        status_img.fill(0) #zero for each pixels means it has not been seen, 1 means seen
        for i in range(h):
            for j in range(w):
                if( scale_img[i,j] > 0.0 and status_img[i,j] == 0 ):
                    #contains an array of {[x,y,scale]} which is the position and scale of the pixels lying in direction
                    #of the eigen vector coresponding to the largest eigen value modulus

                    ridge_perpendicular_cut =\
                        self.find_pixels_in_max_evector_dir( i, j, scale_list,\
                        scale_img, status_img, H )

                    max_scale = np.max(ridge_perpendicular_cut[:,2])
                    scale_array = ridge_perpendicular_cut[:,2]
                    if( len(scale_list) > 1 ):
                        frequency_of_scales = np.histogram(scale_array,\
                            bins=scale_list)
                        max_index = np.argmax(frequency_of_scales[0])
                        freq_scale = np.max(frequency_of_scales[1][max_index])
                    else:
                        freq_scale = max_scale
                    #decision based on average
                    avg_scale = np.average(scale_array)
                    if( np.abs(max_scale-avg_scale)<np.abs(freq_scale-avg_scale)):
                        scale = max_scale
                    else:
                        scale = freq_scale
                    #avg_scale = np.average(scale_array)
                    scale = self.find_closest_value(scale_list, avg_scale)
                    for p in ridge_perpendicular_cut:

                        if(status_img[int(p[0]),int(p[1])] == 0):
                            scale_img[int(p[0]),int(p[1])]  = scale
                            status_img[int(p[0]),int(p[1])] = 1


        return

    def multiscale_fourth_order_anisotropic_diffusion_filter_3d(self, orig, f,\
            hx, hy, hz, sigma_list, general_sigma, rho, var_lambda, T, M,\
            c_vars, alpha_var, betta_var, theta_var, crease_type = 'r',\
            it = -1, path_for_notes = "", detect_planes = False):

        d, h, w = f.shape
        # ---------- Initialization --------
        u = np.copy(f)

        tau_max  = 0.5 ** 3 # dim = 3
        tau_max  = 0.001
        fed = FED.FED()
        tau = fed.fed_tau_by_process_time( T, M, tau_max, reordering = 1)
        tau = np.asarray(tau)
        #tau.fill(tau_max)
        # ------ End of Initialization -----

        coeff = 1.0/np.sqrt(2.0)

        # Filtering loop
        for k in range(M):

            # ----------------- Compute P(u^k) before cycle -------------------

            # Find scales for regularization
            scale_image, evals, evecs,vess_loc = self.vesselness_measure(u,\
                sigma_list, general_sigma, 0.0, c_vars, alpha_var, betta_var,\
                theta_var, crease_type, detect_planes = detect_planes)

            # --- Compute diffusion tensor D ---
            evec_0 = evecs[:,:,:,:,0].reshape((d*w*h,3))
            evec_1 = evecs[:,:,:,:,1].reshape((d*w*h,3))
            evec_2 = evecs[:,:,:,:,2].reshape((d*w*h,3))
            
            e_tnsr = np.empty((d,h,w, 9, 9))

            # Use einstein sum notation for outer products to avoid "for" loops
            e_tnsr[:,:,:,:,0] = np.einsum('ij...,i...->ij...', evec_0, evec_0)\
                  .reshape((d,h,w,3*3))
            e_tnsr[:,:,:,:,1] = np.einsum('ij...,i...->ij...', evec_1, evec_1)\
                  .reshape((d,h,w,3*3))
            e_tnsr[:,:,:,:,2] = np.einsum('ij...,i...->ij...', evec_2, evec_2)\
                  .reshape((d,h,w,3*3))
            e_tnsr[:,:,:,:,3] = (np.einsum('ij...,i...->ij...', evec_0, evec_1)\
                  .reshape((d,h,w,3*3)) +\
                  np.einsum('ij...,i...->ij...', evec_1, evec_0).\
                  reshape((d,h,w,3*3))) * coeff
            e_tnsr[:,:,:,:,4] = (np.einsum('ij...,i...->ij...', evec_0, evec_2)\
                  .reshape((d,h,w,3*3)) +\
                  np.einsum('ij...,i...->ij...', evec_2, evec_0).\
                  reshape((d,h,w,3*3))) * coeff
            e_tnsr[:,:,:,:,5] = (np.einsum('ij...,i...->ij...', evec_1, evec_2)\
                  .reshape((d,h,w,3*3)) +\
                  np.einsum('ij...,i...->ij...', evec_2, evec_1).\
                  reshape((d,h,w,3*3))) * coeff
            e_tnsr[:,:,:,:,6] = (np.einsum('ij...,i...->ij...', evec_0, evec_1)\
                  .reshape((d,h,w,3*3)) -\
                  np.einsum('ij...,i...->ij...', evec_1, evec_0).\
                  reshape((d,h,w,3*3))) * coeff
            e_tnsr[:,:,:,:,7] = (np.einsum('ij...,i...->ij...', evec_0, evec_2)\
                  .reshape((d,h,w,3*3)) -\
                  np.einsum('ij...,i...->ij...', evec_2, evec_0).\
                  reshape((d,h,w,3*3))) * coeff
            e_tnsr[:,:,:,:,8] = (np.einsum('ij...,i...->ij...', evec_1, evec_2)\
                  .reshape((d,h,w,3*3)) -\
                  np.einsum('ij...,i...->ij...', evec_2, evec_1).\
                  reshape((d,h,w,3*3))) * coeff



            # Calculate eigenvalues from eigenvalues of the Hessian matrices
            m_evl = np.zeros((d,h,w,9,9))
            ones  = np.ones((d,h,w))

            var_lambda_sqr = var_lambda ** 2
            mask = evals[:,:,:,0] >= 0.0

            PM_diff = 1.0 / (((evals[:,:,:,0] ** 2) / var_lambda_sqr) + 1.0)
            PM_diff = np.copy(ones) * 1.0
            m_evl[:,:,:,0,0] = (mask * ones) + (np.logical_not(mask) * PM_diff)

            mask = evals[:,:,:,1] >= 0.0
            PM_diff = 1.0 / (((evals[:,:,:,1] ** 2) / var_lambda_sqr) + 1.0)
            m_evl[:,:,:,1,1] = (mask * ones) + (np.logical_not(mask) * PM_diff)

            mask = evals[:,:,:,2] >= 0.0
            PM_diff = 1.0 / (((evals[:,:,:,2] ** 2) / var_lambda_sqr) + 1.0)
            #PM_diff = 1.0
            m_evl[:,:,:,2,2] = (mask * ones) + (np.logical_not(mask) * PM_diff)

            m_evl[:,:,:,3,3] = (m_evl[:,:,:,0,0] + m_evl[:,:,:,1,1]) * 0.5
            m_evl[:,:,:,4,4] = (m_evl[:,:,:,0,0] + m_evl[:,:,:,2,2]) * 0.5
            m_evl[:,:,:,5,5] = (m_evl[:,:,:,1,1] + m_evl[:,:,:,2,2]) * 0.5

            m_evl[:,:,:,6,6] = np.zeros((d,h,w))
            m_evl[:,:,:,7,7] = np.zeros((d,h,w))
            m_evl[:,:,:,8,8] = np.zeros((d,h,w))

            # Compute diffusion tensor D from its eigenvalues and eigentensors
            tnsr_lin = e_tnsr.reshape((d*h*w,9,9))
            mevl_lin = m_evl.reshape((d*h*w,9,9))
            tnsr_T   = np.transpose(tnsr_lin, axes = (0,2,1))

            mvT = np.einsum('fij,fjk->fik', mevl_lin, tnsr_T)
            D   = np.einsum('fij,fjk->fik', tnsr_lin, mvT)
            # ------------ End of Compute P(u^k) before cycle -----------------

            # Within cycle steps
            u_i = np.copy(u)
            for t in tau:

                # Compute Hessian matrix of u_i in a vectorized form
                H = np.empty((d,h,w,9))

                H[:,:,:,0] = self.image_der_dd(u_i, 'xx')
                H[:,:,:,4] = self.image_der_dd(u_i, 'yy')
                H[:,:,:,8] = self.image_der_dd(u_i, 'zz')

                H[:,:,:,1] = self.image_der_dd(u_i, 'xy',\
                    kernel_ab_type = 1)
                H[:,:,:,2] = self.image_der_dd(u_i, 'xz',\
                    kernel_ab_type = 1)
                H[:,:,:,5] = self.image_der_dd(u_i, 'yz',\
                    kernel_ab_type = 1)
                H[:,:,:,3] = H[:,:,:,1]
                H[:,:,:,6] = H[:,:,:,2]
                H[:,:,:,7] = H[:,:,:,5]

                H_lin = H.reshape((d*h*w, 9))
                D_H   = np.einsum('fij,fj->fi', D, H_lin).reshape((d,h,w,9))
                # Explicit Euler Scheme
                u_i = u_i - t * (\
                    self.image_der_dd(D_H[:,:,:,0], 'xx', kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,:,3], 'xy', kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,:,1], 'yx', kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,:,4], 'yy', kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,:,2], 'zx', kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,:,6], 'xz', kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,:,8], 'zz', kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,:,5], 'zy', kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,:,7], 'yz', kernel_ab_type = 1) \
                    )
            u = np.copy(u_i)

        return u


        
    def multiscale_guassian_filter(self, orig, f,\
             hx, hy,\
             sigma_list, general_sigma, rho,\
             c_vars=[], betta_var=0.0, theta_var=0.0, alpha_var = 0,crease_type="r",\
             gamma = 2.0, selector_type = 'F',te=5, it = -1,\
             RETURN_VESSELNESS_MEASURE=False, path_for_notes=""):
        """
        selector_type : {'F', 'M', 'N', 'A'}
                        'F': Uses Frangi et al. filter for scale selection
                        'M': Uses Lindeberg, ML as ridge strength
                        'N': Uses Lindeberg, N as ridge strength
                        'A': Uses Lindeberg, A as ridge strength
        """
        h, w = f.shape
        u = np.copy(f)

        if( selector_type == 'F' ):
            # Find scales for regularization
            scale_image, evals, evecs,vess_loc = self.vesselness_measure(u,\
                sigma_list, general_sigma, rho, c_vars, alpha_var, betta_var,\
                theta_var, crease_type, hx = hx, hy = hy,postprocess = False)
        else:
            # Convert the t values to sigma values 
            sigma_list = np.empty(te)
            for i in range(te):
                t = i + 1
                sigma_list[i] = np.sqrt( 2.0 *  t)
                
            scale_image, H_mts = self.lindeberg_scale_selection( u, sigma_list,\
                 gamma=gamma, hx=hx, hy=hy, selector_type = selector_type)
    
        
        res = np.zeros(f.shape)
        # Background
        mask = scale_image == 0.0
        
        u_s = self.image_convolve_gaussian(u, general_sigma) 
        res = res + ( mask * u_s )
        t = 1
        for s in sigma_list:
            coeff = 1.0
            '''
            if( selector_type == 'F' ):
                coeff = s ** 2
            elif( selector_type == 'M' ):
                coeff = ((s**gamma)/2.0)
                   
                
            elif( selector_type == 'N'):
                coeff = s**(4.0*gamma)
            
            elif( selector_type == 'A'):
                coeff = s**(2.0*gamma)
            '''
            mask = scale_image == s
            u_s = self.image_convolve_gaussian(u, s) * coeff
            u_s = self.normalize(u_s)
            res = res + ( mask * u_s )
            t += 1
        if( rho > 0.0 ):
            res = self.image_convolve_gaussian(res, rho)
        return res

    def multiscale_fourth_order_anisotropic_diffusion_filter_mirror_boundary(self, orig, f,\
             hx, hy,\
             sigma_list, general_sigma, rho, var_lambda,\
             c_vars, betta_var, theta_var,T, M,\
             crease_type = 'r',  alpha_var = 0,\
             smooth_max_crease = False,\
             it = -1, RETURN_VESSELNESS_MEASURE=False, path_for_notes="",
             auto_stop = False, EULER_EX_STEP = False):

        h, w = f.shape
        # ---------- Initialization --------
        u = np.copy(f)

        tau_max  = 0.1 ** 2 # dim = 2
        fed = FED.FED()
        
        if( EULER_EX_STEP ):
            tau = [M]
            M = T
        else:
            tau = fed.fed_tau_by_process_time( T, M, tau_max, reordering = 1)
        tau = np.asarray(tau)

        # ------ End of Initialization -----

        coeff = 1.0/np.sqrt(2.0)


        # Filtering loop
        for k in range(M):
            # ----------------- Compute P(u^k) before cycle -------------------

            # Find scales for regularization
            scale_image, evals, evecs,vess_loc = self.vesselness_measure(u,\
                sigma_list, general_sigma, rho, c_vars, alpha_var, betta_var,\
                theta_var, crease_type, hx = hx, hy = hy)
                
            # --- Compute diffusion tensor D ---
            evec_0 = evecs[:,:,:,0].reshape((w*h,2))
            evec_1 = evecs[:,:,:,1].reshape((w*h,2))

            e_tnsr = np.empty((h,w, 4, 4))

            # Use einstein sum notation for outer products to avoid "for" loops
            # reshape vectorizes the matrix row-wise (xx, yx, xy, yy)
            e_tnsr[:,:,:,0] = np.einsum('ij...,i...->ij...', evec_0, evec_0)\
                  .reshape((h,w,2*2))
            e_tnsr[:,:,:,1] = np.einsum('ij...,i...->ij...', evec_1, evec_1)\
                  .reshape((h,w,2*2))
            e_tnsr[:,:,:,2] = (np.einsum('ij...,i...->ij...', evec_0, evec_1)\
                  .reshape((h,w,2*2)) +\
                  np.einsum('ij...,i...->ij...', evec_1, evec_0).\
                  reshape((h,w,2*2))) * coeff
            e_tnsr[:,:,:,3] = (np.einsum('ij...,i...->ij...', evec_0, evec_1)\
                  .reshape((h,w,2*2)) -\
                  np.einsum('ij...,i...->ij...', evec_1, evec_0).\
                  reshape((h,w,2*2))) * coeff

            # Calculate eigenvalues from eigenvalues of the Hessian matrices
            m_evl = np.zeros((h,w,4,4))
            ones  = np.ones((h,w))

            var_lambda_sqr = var_lambda ** 2

            mask = evals[:,:,0] >= 0.0
            mask = np.zeros((h,w))
            PM_diff = 1.0 / (((evals[:,:,0] ** 2) / var_lambda_sqr) + 1.0)

            m_evl[:,:,0,0] = (mask * ones) + (np.logical_not(mask) * PM_diff)

            mask = evals[:,:,1] >= 0.0
            mask = np.zeros((h,w))
            PM_diff = 1.0 / (((evals[:,:,1] ** 2) / var_lambda_sqr) + 1.0)
            m_evl[:,:,1,1] = (mask * ones) + (np.logical_not(mask) * PM_diff)

            m_evl[:,:,2,2] = (m_evl[:,:,0,0] + m_evl[:,:,1,1]) * 0.5

            m_evl[:,:,3,3] = np.zeros((h,w))

            # Compute diffusion tensor D from its eigenvalues and eigentensors
            tnsr_lin = e_tnsr.reshape((h*w,4,4))
            mevl_lin = m_evl.reshape((h*w,4,4))
            tnsr_T   = np.transpose(tnsr_lin, axes = (0,2,1))

            mvT = np.einsum('fij,fjk->fik', mevl_lin, tnsr_T)
            D   = np.einsum('fij,fjk->fik', tnsr_lin, mvT)
            """
            # Average D on local neighborhood
            if( rho > 0.0 ):
                D[:,0,0] = self.image_convolve_gaussian(D[:,0,0], rho)
                D[:,0,1] = self.image_convolve_gaussian(D[:,0,1], rho)
                D[:,0,2] = self.image_convolve_gaussian(D[:,0,2], rho)
                D[:,0,3] = self.image_convolve_gaussian(D[:,0,3], rho)
                D[:,1,0] = self.image_convolve_gaussian(D[:,1,0], rho)
                D[:,1,1] = self.image_convolve_gaussian(D[:,1,1], rho)
                D[:,1,2] = self.image_convolve_gaussian(D[:,1,2], rho)
                D[:,1,3] = self.image_convolve_gaussian(D[:,1,3], rho)
                D[:,2,0] = self.image_convolve_gaussian(D[:,2,0], rho)
                D[:,2,1] = self.image_convolve_gaussian(D[:,2,1], rho)
                D[:,2,2] = self.image_convolve_gaussian(D[:,2,2], rho)
                D[:,2,3] = self.image_convolve_gaussian(D[:,2,3], rho)
                D[:,3,0] = self.image_convolve_gaussian(D[:,3,0], rho)
                D[:,3,1] = self.image_convolve_gaussian(D[:,3,1], rho)
                D[:,3,2] = self.image_convolve_gaussian(D[:,3,2], rho)
                D[:,3,3] = self.image_convolve_gaussian(D[:,3,3], rho)

            """
            # ------------ End of Compute P(u^k) before cycle -----------------
            # Within cycle steps
            u_i = np.copy(u)
            for t in tau:
                # Compute Hessian matrix of u_i in a vectorized form
                H = np.empty((h,w,4))

                H[:,:,0] = self.image_der_dd(u_i, 'xx', hx, hy)
                H[:,:,3] = self.image_der_dd(u_i, 'yy', hx, hy)
                H[:,:,1] = self.image_der_dd(u_i, 'xy', hx, hy,\
                    kernel_ab_type = 1)
                H[:,:,2] = H[:,:,1]

                # Compute D(u):H(u_i) where : is the double contraction
                H_lin = H.reshape((h*w, 4))
                D_H   = np.einsum('fij,fj->fi', D, H_lin).reshape((h,w,4))

                # Explicit Euler Scheme
                u_i = u_i - t * (\
                    self.image_der_dd(D_H[:,:,0], 'xx', hx, hy,\
                            kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,2], 'xy', hx, hy,\
                            kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,1], 'yx', hx, hy,\
                            kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,3], 'yy', hx, hy,\
                            kernel_ab_type = 1) )
            u = np.copy(u_i)
       
        return u


    def multiscale_fourth_order_anisotropic_diffusion_filter(self, orig2, f2,\
             hx, hy,\
             sigma_list, general_sigma, rho, var_lambda,\
             c_vars, betta_var, theta_var,T, M,\
             crease_type = 'r',  alpha_var = 0,\
             smooth_max_crease = False,\
             it = -1, RETURN_VESSELNESS_MEASURE=False, path_for_notes="",
             auto_stop = False, EULER_EX_STEP = False, STABILITY_CHECK = False):

        f    = np.lib.pad(f2   , (2, 2), mode = 'reflect')        
        
        h, w = f.shape
        # ---------- Initialization --------
        u = np.copy(f)
        
        bx = 2
        by = 2
        
        tau_max  = 0.05 #0.1 ** 2 # dim = 2
       # tau_max  = 0.03
        fed = FED.FED()
        
        if( EULER_EX_STEP ):
            tau = [M]
            #M = int(T/0.03)
            M = T
            
        else:
            tau = fed.fed_tau_by_process_time( T, M, tau_max, reordering = 1)
        tau = np.asarray(tau)
        
        # ------ End of Initialization -----
        coeff = 1.0/np.sqrt(2.0)

        diffusion_time = 0

        # Filtering loop
        for k in range(M):
            # ----------------- Compute P(u^k) before cycle -------------------
            # Find scales for regularization
            scale_image, evals, evecs,vess_loc = self.vesselness_measure(u[by:h-by,bx:w-bx],\
                sigma_list, general_sigma, rho, c_vars, alpha_var, betta_var,\
                theta_var, crease_type, hx = hx, hy = hy, ignore_boundary=True,\
                BOUNDARY_CONDITION='natural')
            # --- Compute diffusion tensor D ---
            evec_0 = evecs[:,:,:,0].reshape((w*h,2))
            evec_1 = evecs[:,:,:,1].reshape((w*h,2))

            e_tnsr = np.empty((h,w, 4, 4))

            # Use einstein sum notation for outer products to avoid "for" loops
            # reshape vectorizes the matrix row-wise (xx, yx, xy, yy)
            e_tnsr[:,:,:,0] = np.einsum('ij...,i...->ij...', evec_0, evec_0)\
                  .reshape((h,w,2*2))
            e_tnsr[:,:,:,1] = np.einsum('ij...,i...->ij...', evec_1, evec_1)\
                  .reshape((h,w,2*2))
            e_tnsr[:,:,:,2] = (np.einsum('ij...,i...->ij...', evec_0, evec_1)\
                  .reshape((h,w,2*2)) +\
                  np.einsum('ij...,i...->ij...', evec_1, evec_0).\
                  reshape((h,w,2*2))) * coeff
            e_tnsr[:,:,:,3] = (np.einsum('ij...,i...->ij...', evec_0, evec_1)\
                  .reshape((h,w,2*2)) -\
                  np.einsum('ij...,i...->ij...', evec_1, evec_0).\
                  reshape((h,w,2*2))) * coeff

            # Calculate eigenvalues from eigenvalues of the Hessian matrices
            m_evl = np.zeros((h,w,4,4))
            ones  = np.ones((h,w))

            var_lambda_sqr = var_lambda ** 2

            mask = evals[:,:,0] >= 0.0
            mask = np.zeros((h,w))
            PM_diff = 1.0 / (((evals[:,:,0] ** 2) / var_lambda_sqr) + 1.0)

            m_evl[:,:,0,0] = (mask * ones) + (np.logical_not(mask) * PM_diff)

            mask = evals[:,:,1] >= 0.0
            mask = np.zeros((h,w))
            PM_diff = 1.0 / (((evals[:,:,1] ** 2) / var_lambda_sqr) + 1.0)
            #PM_diff = ones
            m_evl[:,:,1,1] = (mask * ones) + (np.logical_not(mask) * PM_diff)

            m_evl[:,:,2,2] = (m_evl[:,:,0,0] + m_evl[:,:,1,1]) * 0.5

            m_evl[:,:,3,3] = np.zeros((h,w))

            # Compute diffusion tensor D from its eigenvalues and eigentensors
            tnsr_lin = e_tnsr.reshape((h*w,4,4))
            mevl_lin = m_evl.reshape((h*w,4,4))
            tnsr_T   = np.transpose(tnsr_lin, axes = (0,2,1))

            mvT = np.einsum('fij,fjk->fik', mevl_lin, tnsr_T)
            D   = np.einsum('fij,fjk->fik', tnsr_lin, mvT)
            D   = D.reshape((h, w, 4, 4))
            D[0:by+1,:,:,:] = 0.0
            D[:,0:bx+1,:,:] = 0.0
            D[h-by-1:,:,:,:] = 0.0
            D[:,w-bx-1:,:,:] = 0.0
            D = D.reshape((h*w,4,4))
            # ------------ End of Compute P(u^k) before cycle -----------------

            # Within cycle steps
            u_i = np.copy(u)
            for t in tau:
                
                diffusion_time += t
                # Compute Hessian matrix of u_i in a vectorized form
                H = np.empty((h,w,4))
                u_der_xx = self.image_der_dd(u_i, 'xx', hx, hy)
                u_der_yy = self.image_der_dd(u_i, 'yy', hx, hy)
                u_der_xy = self.image_der_dd(u_i, 'xy', hx, hy,\
                            kernel_ab_type = 1)
                u_der_xx[0:by+1, :    ] = 0.0
                u_der_xx[ :    ,0:bx+1] = 0.0
                u_der_xx[h-by-1:  ,      :  ] = 0.0
                u_der_xx[      :  ,w-bx-1:  ] = 0.0
                
                u_der_yy[0:by+1, :  ] = 0.0
                u_der_yy[ :    ,0:bx+1] = 0.0
                u_der_yy[h-by-1:  ,      :  ] = 0.0
                u_der_yy[      :  ,w-bx-1:  ] = 0.0
                
                u_der_xy[0:by+1, :    ] = 0.0
                u_der_xy[ :    ,0:bx+1] = 0.0
                u_der_xy[h-by-1:  ,      :  ] = 0.0
                u_der_xy[      :  ,w-bx-1:  ] = 0.0
                
                H[:,:,0] = u_der_xx
                H[:,:,3] = u_der_yy
                H[:,:,1] = u_der_xy
                
                H[:,:,2] = H[:,:,1]

                # Compute D(u):H(u_i) where : is the double contraction
                H_lin = H.reshape((h*w, 4))
                D_H   = np.einsum('fij,fj->fi', D, H_lin).reshape((h,w,4))
                
                incr_xx = self.image_der_dd(D_H[:,:,0], 'xx', hx, hy,\
                            kernel_ab_type = 1,ignore_boundary = True, size = 1)
                incr_xy = self.image_der_dd(D_H[:,:,2], 'xy', hx, hy,\
                            kernel_ab_type = 1,ignore_boundary = True, size = 1)
                incr_yx = self.image_der_dd(D_H[:,:,1], 'yx', hx, hy,\
                            kernel_ab_type = 1,ignore_boundary = True, size = 1)
                incr_yy = self.image_der_dd(D_H[:,:,3], 'yy', hx, hy,\
                            kernel_ab_type = 1,ignore_boundary = True, size = 1)
                            
                incr_xx[0:by-1, :    ] = 0.0
                incr_xx[ :    ,0:bx-1] = 0.0
                incr_xx[h-by+1:  ,      :  ] = 0.0
                incr_xx[      :  ,w-bx+1:  ] = 0.0
                
                incr_xy[0:by-1, :  ] = 0.0
                incr_xy[ :    ,0:bx-1] = 0.0
                incr_xy[h-by+1:  ,      :  ] = 0.0
                incr_xy[      :  ,w-bx+1:  ] = 0.0
                
                incr_yx[0:by-1, :    ] = 0.0
                incr_yx[ :    ,0:bx-1] = 0.0
                incr_yx[h-by+1:  ,      :  ] = 0.0
                incr_yx[      :  ,w-bx+1:  ] = 0.0   
                
                incr_yy[0:by-1, :    ] = 0.0
                incr_yy[ :    ,0:bx-1] = 0.0
                incr_yy[h-by+1:  ,      :  ] = 0.0
                incr_yy[      :  ,w-bx+1:  ] = 0.0  
                         
                # Explicit Euler Scheme
                u_i = u_i - t * ( incr_xx + incr_xy + incr_yx + incr_yy )
             
            u[by:h-by,bx:w-bx] = np.copy(u_i[by:h-by,bx:w-bx])
            print("Cycle ", k+1)
        return u[by:h-by,bx:w-bx]


    def CED_fourth_order_anisotropic_diffusion_filter(self, orig, f,\
             hx, hy,\
             sigma_list, general_sigma, rho, var_lambda,\
             c_vars, betta_var, theta_var,T, M,\
             crease_type = 'r',  alpha_var = 0,\
             smooth_max_crease = False,\
             it = -1, RETURN_VESSELNESS_MEASURE=False, path_for_notes="",\
             auto_stop = False, EULER_EX_STEP = False):

        h, w = f.shape
        # ---------- Initialization --------
        u = np.copy(f)

        tau_max  = 0.1 ** 2 # dim = 2
        tau_max = 0.001
        #tau_max  = 0.03
        fed = FED.FED()
        tau = fed.fed_tau_by_process_time( T, M, tau_max, reordering = 1)
        if( EULER_EX_STEP ):
            M = int(T/0.03)
            tau = [0.03]
        tau = np.asarray(tau)

        # ------ End of Initialization -----


#        coeff = 1.0/np.sqrt(2.0)



        # Filtering loop
        for k in range(M):
            # ----------------- Compute P(u^k) before cycle -------------------

            # Find scales for regularization
            scale_image, evals, evecs,h_mts, vess_loc =\
                self.vesselness_measure(u,\
                sigma_list, general_sigma, 0.0, c_vars, alpha_var, betta_var,\
                theta_var, crease_type, hx = hx, hy = hy, RETURN_HESSIAN=True)

            # Compute the hessian matrices at different scales
            MH = np.copy(h_mts)
            MH = MH.reshape((h, w, 2 * 2))
            MH_norm = np.sqrt(MH[:,:,0]**2 + MH[:,:,1]**2 + MH[:,:,2]**2 +\
                            MH[:,:,3]**2)
           # MH_norm = np.sqrt(evals[:,:,0]**2 + evals[:,:,1]**2)
            MH_norm = np.repeat(MH_norm.flatten(), 4, axis = -1).\
                        reshape((h, w, 4))
            MH_norm_dnum = np.copy(MH_norm)
            MH_norm_dnum[np.where(MH_norm == 0.0)] = 1.e-20
            MH = MH/MH_norm_dnum

            # Compute the structure tensors from the Hessians
            HH = np.einsum('fij...,fi...->fij...', MH, MH)

            MH_norm = np.repeat(MH_norm.flatten(), 4, axis = -1).\
                        reshape((h, w, 4, 4))
            MH_norm_sqr = np.power(MH_norm, 2)
           # plot.show_image(MH_norm_sqr[:,:,0,0])
            var_lambda_sqr = var_lambda ** 2
            PM_diff = 1.0 / (((MH_norm_sqr) / var_lambda_sqr) + 1.0)
            """
            if( rho > 0.0 ):
                HH[:,:,0,0] = self.image_convolve_gaussian(HH[:,:,0,0], rho)
                HH[:,:,0,1] = self.image_convolve_gaussian(HH[:,:,0,1], rho)
                HH[:,:,0,2] = self.image_convolve_gaussian(HH[:,:,0,2], rho)
                HH[:,:,0,3] = self.image_convolve_gaussian(HH[:,:,0,3], rho)
                HH[:,:,1,0] = self.image_convolve_gaussian(HH[:,:,1,0], rho)
                HH[:,:,1,1] = self.image_convolve_gaussian(HH[:,:,1,1], rho)
                HH[:,:,1,2] = self.image_convolve_gaussian(HH[:,:,1,2], rho)
                HH[:,:,1,3] = self.image_convolve_gaussian(HH[:,:,1,3], rho)
                HH[:,:,2,0] = self.image_convolve_gaussian(HH[:,:,2,0], rho)
                HH[:,:,2,1] = self.image_convolve_gaussian(HH[:,:,2,1], rho)
                HH[:,:,2,2] = self.image_convolve_gaussian(HH[:,:,2,2], rho)
                HH[:,:,2,3] = self.image_convolve_gaussian(HH[:,:,2,3], rho)
                HH[:,:,3,0] = self.image_convolve_gaussian(HH[:,:,3,0], rho)
                HH[:,:,3,1] = self.image_convolve_gaussian(HH[:,:,3,1], rho)
                HH[:,:,3,2] = self.image_convolve_gaussian(HH[:,:,3,2], rho)
                HH[:,:,3,3] = self.image_convolve_gaussian(HH[:,:,3,3], rho)
            """
            I = np.empty((h, w, 4, 4))
            I[:,:] = np.identity(4)

            # Diffusion Tensor
            D = I + ( PM_diff - 1.0 ) * HH
            # ------------ End of Compute P(u^k) before cycle -----------------
            # Within cycle steps
            u_i = np.copy(u)
            for t in tau:
                # Compute Hessian matrix of u_i in a vectorized form
                H = np.empty((h,w,4))

                H[:,:,0] = self.image_der_dd(u_i, 'xx', hx, hy)
                H[:,:,3] = self.image_der_dd(u_i, 'yy', hx, hy)
                H[:,:,1] = self.image_der_dd(u_i, 'xy', hx, hy,\
                    kernel_ab_type = 1)
                H[:,:,2] = H[:,:,1]

                # Compute D(u):H(u_i) where : is the double contraction
                D_H   = np.einsum('fkij,fkj->fki', D, H)

                # Explicit Euler Scheme
                u_i = u_i - t * (\
                    self.image_der_dd(D_H[:,:,0], 'xx', hx, hy,\
                            kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,2], 'xy', hx, hy,\
                            kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,1], 'yx', hx, hy,\
                            kernel_ab_type = 1)+\
                    self.image_der_dd(D_H[:,:,3], 'yy', hx, hy,\
                            kernel_ab_type = 1) )
            u = np.copy(u_i)
        return u

    def find_max(self, a):
        eps = 1.e-3
        max_index = np.argmax(a)
        return max_index
        new_i = max_index
        counter = 0
        for v in a:
            if( abs(a[new_i] - v) < eps ):
                new_i = counter
            counter += 1
        return new_i

    # Sort based on eigenvalues. Order eigenvalues with decreasing modulus.
    # For 2d
    def sort_based_on_evalues_modulus(self, evl, evc):
        if( len(evl.shape) == 2 ):
            list_size, k = evl.shape
            for i in range( list_size ):
                abs_evl = np.absolute(evl[i,:])
                if(abs_evl[0] < abs_evl[1]):
                    tmp    = evl[i,0]
                    evl[i,0] = evl[i,1]
                    evl[i,1] = tmp

                    tmp      = np.copy(evc[i,:,0])
                    evc[i,:,0] = np.copy(evc[i,:,1])
                    evc[i,:,1] = np.copy(tmp)
        else:
            abs_evl = np.absolute(evl)
            if(abs_evl[0] < abs_evl[1]):
                tmp    = evl[0]
                evl[0] = evl[1]
                evl[1] = tmp

                tmp      = np.copy(evc[:,0])
                evc[:,0] = np.copy(evc[:,1])
                evc[:,1] = np.copy(tmp)
        return
    def sort_based_on_evalues_modulus_increasingly(self, evl, evc):
        arg_sort_evl = np.argsort(np.abs(evl))
        return evl[arg_sort_evl], evc[:,arg_sort_evl]



