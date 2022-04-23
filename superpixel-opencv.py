import warnings
warnings.filterwarnings("ignore")


from  sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries 
from skimage.util import img_as_float 
from skimage import io 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2


# Load the Image and grab itss width and heights 

image = cv2.imread("/home/asma/Documents/Programing/Superpixels/codes/Superpixels-code/image1.jpeg")
(h , w) = image.shape[:2]

print(h , w)
plt.imshow(image)
plt.show()

# Convert image to floating point 
image = img_as_float(image)

print(image)
print(type(image))

### Loop over the number of segments 
# 200 , 300 ,500 ,800 are the number of detected superpixels

"""Segments an image using k-means clustering in Color-(x,y,z) space.

    Args:
        image: The image.
        num_segments: The (approiximate) number of segments in the segmented
          output image (optional).
        compactness: Balances color-space proximity and image-space-proximity.
          Higher values give more weight to image-space proximity (optional).
        max_iterations: Maximum number of iterations of k-means.
        sigma: Width of Gaussian kernel used in preprocessing (optional).
        min_size_factor: Proportion of the minimum segment size to be removed
          with respect to the supposed segment size
          `depth*width*height/num_segments` (optional).
        max_size_factor: Proportion of the maximum connected segment size
          (optional).
        enforce_connectivitiy: Whether the generated segments are connected or
          not (optional).

    Returns:
        Integer mask indicating segment labels.
    """

for segmentnum in (200,300,500,800):
    # apply slic algorithm
    segments = slic(image , n_segments= segmentnum , sigma= 5) 

    # show the output of slic 
    fig = plt.figure("Superpixel -- %d segments " %(segmentnum))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(mark_boundaries(image , segments))
    plt.axis("off")
plt.show()