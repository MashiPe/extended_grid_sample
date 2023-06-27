import numpy as np

def extended_grid_sample(image,grid,real_h=240,real_w=240,y_shift=None,x_shift=None):

    h_limit = image.shape[0]
    w_limit = image.shape[1]

    iw = grid.shape[1]
    ih = grid.shape[0]
    c = image.shape[2]

    out_h = grid.shape[0]
    out_w = grid.shape[1]

    # points = grid.reshape((-1-1,2))
    
    ix = grid[:,:,0]
    iy = grid[:,:,1]


    output = np.zeros((out_h,out_w,c))

    # Create extended img to enable out of bounds values
    aux_img = np.zeros((h_limit+1,w_limit+1,3))
    aux_img[:h_limit,:w_limit] = image


    ix = ( (ix+1)/2 )*real_w
    iy = ( (iy+1)/2 )*real_h

    '''
    The reference zone refers to a limited plane where the main transformation takes place.
    Ex: An homography takes and image and changes the perspective, in this case the reference
    zone are the outline of the image, but we can apply the homography on a smaller portion 
    of the image. That would be an image with size (240,240) and the homography takes place
    on an square with size (120,120) in the center of the image, that smaller square is the reference
    zone

    This shift (in pixels) accounts for the information available in the image in case that the given image
    is bigger than the reference zone of the transformation
    Ex: If the reference zone is a (240,240) square and an image with same dimentions then in 
    the normalize space we have the information from -1 to 1, but if the image is bigger than the
    reference zone then we have information beyond -1 to 1. Assuming a symmetrical diference
    betwen the zone and the image adding that diference accounts for the extra information
    in the real image.
    '''

    if x_shift == None:
        x_shift = (w_limit-real_w)//2
    
    if y_shift == None:
        y_shift = (h_limit-real_h)//2

    ix+=x_shift
    iy+=y_shift

    ix_nw = np.floor(ix).astype(np.int64)
    iy_nw = np.floor(iy).astype(np.int64)
    ix_ne = ix_nw +1 
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1 
    iy_se = iy_nw + 1
    
    # ix_nw = ix_nw.astype(np.int32) 
    # iy_nw = iy_nw.astype(np.int32)
    # ix_ne = ix_ne.astype(np.int32) 
    # iy_ne = iy_ne.astype(np.int32)
    # ix_sw = ix_sw.astype(np.int32) 
    # iy_sw = iy_sw.astype(np.int32)
    # ix_se = ix_se.astype(np.int32) 
    # iy_se = iy_se.astype(np.int32)


    nw = (ix_se - ix)    * (iy_se - iy);
    ne = (ix - ix_sw) * (iy_sw - iy);
    sw = (ix_ne - ix)    * (iy    - iy_ne);
    se = (ix    - ix_nw) * (iy    - iy_nw);    
    
    nw_ = np.repeat(np.expand_dims(nw,axis=2),3,axis=2)
    ne_ = np.repeat(np.expand_dims(ne,axis=2),3,axis=2)
    sw_ = np.repeat(np.expand_dims(sw,axis=2),3,axis=2)
    se_ = np.repeat(np.expand_dims(se,axis=2),3,axis=2)
    
    # for h in range(ih):
    #     for w in range(iw):

    #         nw_val = (image[iy_nw[h,w],ix_nw[h,w]] if ix_nw[h,w]>=0 and ix_nw[h,w]<w_limit and  
    #                         iy_nw[h,w]>=0 and iy_nw[h,w]<h_limit else np.zeros((c)))

    #         ne_val = (image[iy_ne[h,w],ix_ne[h,w]] if ix_ne[h,w]>=0 and ix_ne[h,w]<w_limit and  
    #                         iy_ne[h,w]>=0 and iy_ne[h,w]<h_limit else np.zeros((c)))

    #         sw_val = (image[iy_sw[h,w],ix_sw[h,w]] if ix_sw[h,w]>=0 and ix_sw[h,w]<w_limit and
    #                         iy_sw[h,w]>=0 and iy_sw[h,w]<h_limit else np.zeros((c)))

    #         se_val = (image[iy_se[h,w],ix_se[h,w]] if ix_se[h,w]>=0 and ix_se[h,w]<w_limit and
    #                         iy_se[h,w]>=0 and iy_se[h,w]<h_limit else np.zeros((c)))

    #         out_val = nw_val * nw[h,w] +ne_val*ne[h,w] + sw_val * sw[h,w] + se_val*se[h,w]

    #         output[h,w] = out_val


    # cv.imwrite('flex1.jpg',output)

    ix_nw[(ix_nw<0)|(ix_nw>=w_limit) ] = w_limit
    iy_nw[(iy_nw<0)|(iy_nw>=h_limit) ] = h_limit
    ix_ne[(ix_ne<0)|(ix_ne>=w_limit) ] = w_limit
    iy_ne[(iy_ne<0)|(iy_ne>=h_limit) ] = h_limit
    ix_sw[(ix_sw<0)|(ix_sw>=w_limit) ] = w_limit
    iy_sw[(iy_sw<0)|(iy_sw>=h_limit) ] = h_limit
    ix_se[(ix_se<0)|(ix_se>=w_limit) ] = w_limit
    iy_se[(iy_se<0)|(iy_se>=h_limit) ] = h_limit

    nw_val = aux_img[iy_nw,ix_nw]
    ne_val = aux_img[iy_ne,ix_ne]
    sw_val = aux_img[iy_sw,ix_sw]
    se_val = aux_img[iy_se,ix_se]

    out_val = nw_val*nw_ + ne_val*ne_ + sw_val*sw_ + se_val*se_

    output = out_val
    
    return output