from extended_grid_sample import extended_grid_sample
from skimage import io,transform
import numpy as np
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


'''
    In this example it is shown a transformation that is applied to a smaller
    portion of an image but cropping is not necessary and the transformation 
    is projected to all the image
'''

def extend_axis(axis: np.ndarray, left: int ,right: int):
    
    min_limit, max_limit = axis[0],axis[-1]
    n = len(axis)
    step = (max_limit - min_limit) / (n-1)

    left_sign = math.copysign(1,min_limit)
    left_extension = np.array( [ min_limit +(x*step * left_sign ) for x in range(1,left+1) ] )

    right_sign = math.copysign(1,max_limit)
    right_extension = np.array( [ max_limit +(x*step * right_sign ) for x in range(1,right+1) ] )

    return np.concatenate( [ np.flip(left_extension),axis,right_extension ] )


def main():
    
    # Example homography:
    #  (1 ,0 ,0.5)
    #  (0 ,1 ,0.5)
    #  (0 ,0 ,1 )
    h0= 1
    h1= 0
    h2= 0.5
    h3= 0
    h4= 1
    h5= 0.5
    h6= 0
    h7= 0
    h8= 1

    imgPath = "./images/base_imge.jpg"    

    img = io.imread(imgPath)

    img = transform.resize(img,(240,240))
    
    # Create normalize axis
    x_axis = np.linspace(-1,1,120)
    y_axis = np.linspace(-1,1,120)


    '''
        Pytorch demostration where in order to apply the transformation on a smaller
        portion of the image a crop is necessary
    '''

    # Generate X and Y grids
    grid_X , grid_Y = np.meshgrid(x_axis,y_axis)
    grid_X = np.expand_dims(grid_X,2)
    grid_Y = np.expand_dims(grid_Y,2)

    # Apply homography 
    grid_Xp = grid_X*h0 + grid_Y*h1 + h2
    grid_Yp = grid_X*h3 + grid_Y*h4 + h5
    k = grid_X*h6 + grid_Y*h7 + h8

    grid_Xp /= k
    grid_Yp /= k

    # Generate pair-wise grid with extra dimension for pytorch
    sampling_grid = np.array([np.concatenate((grid_Xp,grid_Yp) , axis=2)])

    # Use Pytorch grid_sample
    sampling_grid = torch.FloatTensor(sampling_grid)
    # Using a zone Z with shape (120,120) at coordinates (100,100)
    tensor_img = torch.Tensor(np.array([img[100:220,100:220].transpose((2,0,1))],dtype=np.float32))

    new_img = F.grid_sample(tensor_img,sampling_grid,align_corners=True)

    new_img_pytorch = new_img.transpose(1,2).transpose(2,3).numpy()[0]

    '''
        Demostration where in order to apply the transformation on a smaller
        portion of the image we only need to provide extra parameters and 
        more information is kept
    '''

    # Extend axis
    x_axis = extend_axis(x_axis,100,20)
    y_axis = extend_axis(y_axis,100,20)

    # Same process to generate the grid
    
    # Generate X and Y grids
    grid_X , grid_Y = np.meshgrid(x_axis,y_axis)
    grid_X = np.expand_dims(grid_X,2)
    grid_Y = np.expand_dims(grid_Y,2)

    # Apply homography 
    grid_Xp = grid_X*h0 + grid_Y*h1 + h2
    grid_Yp = grid_X*h3 + grid_Y*h4 + h5
    k = grid_X*h6 + grid_Y*h7 + h8

    grid_Xp /= k
    grid_Yp /= k

    # Generate pair-wise grid with extra dimension for pytorch
    sampling_grid = np.concatenate((grid_Xp,grid_Yp) , axis=2)

    # When calling the proposed implementation just the Z dimensions and the
    # the offset are necessary, the transformation is applied correctly in the 
    # reference zone and projected as necessary to the rest of the image
    new_img_proposal = extended_grid_sample(img,sampling_grid,120,120,100,100)

    io.imsave("./images/example1_pytorch.jpg",(new_img_pytorch*255).astype(np.uint8))
    io.imsave("./images/example1_proposal.jpg",(new_img_proposal*255).astype(np.uint8))

    fig, axs = plt.subplots(1,3)
    axs[0].imshow(img)
    axs[0].set_title('src')
    axs[1].imshow(new_img_pytorch)
    axs[1].set_title('pytorch_implementation')
    axs[2].imshow(new_img_proposal)
    axs[2].set_title('proposal')

    fig.set_dpi(150)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.5)
    plt.show()

    # pass


if __name__ == "__main__":
    main()