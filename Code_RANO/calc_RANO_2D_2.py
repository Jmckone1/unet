import numpy as np

import matplotlib.pyplot as plt

import nibabel as nib
from os import walk
import time

from skimage.morphology import label
from skimage.measure import regionprops, mesh_surface_area, marching_cubes
from skimage.morphology import label, binary_erosion, disk, binary_opening, binary_dilation
from skimage.measure import regionprops, mesh_surface_area, marching_cubes, find_contours
from scipy import signal
from scipy.spatial.distance import cdist
from collections import namedtuple

from deepneuro.utilities.conversion import read_image_files


def _get_pixdim(pixdim, affine, input_affine, verbose=True):

    """ Currently only functional for 3D images.
    """

    if pixdim is None:
        if affine is not None:
            pixdim = np.abs(affine.diagonal()[0:-1])
        elif input_affine is not None:
            pixdim = np.abs(input_affine.diagonal()[0:-1])
        else:
            if verbose:
                print('Warning -- no resolution provided. Assuming isotropic.')
            return [1, 1, 1]

    return pixdim

class Point(namedtuple('Point', 'x y')):

    __slots__ = ()
    @property
    def length(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)
    
    def __str__(self):
        return 'Point: x=%6.3f  y=%6.3f  length=%6.3f' % (self.x, self.y, self.length)


def plot_contours(contours, lw=4, alpha=0.5):
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=lw, alpha=alpha)


def vector_norm(p):
    length = p.length
    return Point(p.x / length, p.y / length)


def compute_pairwise_distances(P1, P2, min_length=10):
    euc_dist_matrix = cdist(P1, P2, metric='euclidean')
    #print("CP euc dist",euc_dist_matrix)
    indices = []
    for x in range(euc_dist_matrix.shape[0]):
        for y in range(euc_dist_matrix.shape[1]):

            p1 = Point(*P1[x])
            p2 = Point(*P1[y])
            d = euc_dist_matrix[x, y]
            
            #print("CP p1", p1)
            #print("CP p2", p2)
            #print("CP d", d)
            
            if p1 == p2:
                continue
            elif d < min_length:
                continue
            else:
                indices.append([p1, p2, d])
            
    return euc_dist_matrix, sorted(indices, key=lambda x: x[2], reverse=True)


def interpolate(p1, p2, d):
    
    X = np.linspace(p1.x, p2.x, round(d)).astype(int)
    Y = np.linspace(p1.y, p2.y, round(d)).astype(int)
    XY = np.asarray(list(set(zip(X, Y))))
    return XY


def find_largest_orthogonal_cross_section(pairwise_distances, img, tolerance=0.01):
    #print("PD",len(pairwise_distances))
    for i, (p1, p2, d1) in enumerate(pairwise_distances):

        # Compute intersections with background pixels
        XY = interpolate(p1, p2, d1)
        intersections = sum(img[x, y] == 0 for x, y in XY)
        if intersections == 0:

            V = vector_norm(Point(p2.x - p1.x, p2.y - p1.y))
            
            # Iterate over remaining line segments
            for j, (q1, q2, d2) in enumerate(pairwise_distances[i:]):
                
                W = vector_norm(Point(q2.x - q1.x, q2.y - q1.y))
                if abs(np.dot(V, W)) < tolerance:
                    
                    XY = interpolate(q1, q2, d2)
                    intersections = sum(img[x, y] == 0 for x, y in XY)
                    if intersections == 0:
                        return [p1, p2, q1, q2],[d1,d2]
                    
def calc_2D_RANO_measure(input_data, pixdim=None, affine=None, mask_value=0, axis=2, calc_multiple=False, background_image=None, output_filepath=None, verbose=True):
    
    input_data, input_affine = read_image_files(input_data, return_affine=True)
    #input_affine = None
    pixdim = _get_pixdim(pixdim, affine, input_affine)

    #print("labels",component_labels)
    
    Bidimensional_output=np.zeros([8,155])
    #print(Bidimensional_output)
    #print(Bidimensional_output.shape)
    #input("")
    
    max_2ds = []
    max_2d_images = []

    major_diameter = None

    #major_diameter = None
    for z_slice in range(input_data.shape[2]): # here it wants to be not all the objects but the largest of the objects

        connected_components= label(input_data[:,:,z_slice], connectivity=1)

        (unique, counts) = np.unique(connected_components, return_counts=True)
        counts[0] = 0
        largest_component = (connected_components == np.argmax(counts)).astype(int)
        
        #plt.imshow(largest_component)
        #plt.show()
        #input("")
        
        #print("coomponent sum", sum(sum(largest_component[:,:,z_slice])))
        if sum(sum(largest_component)) > 10:
            #plt.imshow(largest_component[:,:,z_slice])
            #plt.show()
            label_slice = binary_dilation(largest_component).astype(largest_component.dtype)

            label_properties = regionprops(label_slice)
            #print("length",len(label_properties))
            #print("length",label_idx)

            #print("length",label_properties[0].major_axis_length)
            current_major = label_properties[0].major_axis_length
            #print("CM",current_major)
            current_orientation = label_properties[0].orientation
            #print("CO",current_orientation)

            if np.sum(label_slice) == 0:
                continue

            p,d = calc_rano_points(label_slice)

            if p != None:
                x_dim = abs(np.cos(current_orientation) * current_major)
                y_dim = abs(np.sin(current_orientation) * current_major)
                current_major = ((x_dim * pixdim[0])**2 + (y_dim * pixdim[1])**2)**.5

                if major_diameter is None:
                    major_diameter = current_major
                elif current_major > major_diameter:
                    major_diameter = current_major
                
                if p[0] != 0:
                    
                    D1 = np.asarray([[p[0].x, p[1].x], [p[0].y, p[1].y]])
                    D2 = np.asarray([[p[2].x, p[3].x], [p[2].y, p[3].y]])
                    
                    data_out = []
                    for i in range(4):
                        data_out = np.append(data_out, p[i].x)
                        data_out = np.append(data_out, p[i].y)
                        
                    #print("save",data_out) 
                    
                    #print("test_out", data_out[1],data_out[3],data_out[0],data_out[2],data_out[5],data_out[7],data_out[4],data_out[6]) #
                    #this is saved into the same format as p - so to plot it would need to be arranged ##################################
                    #D1 = np.asarray([[data_out[1],data_out[3],[data_out[0],data_out[2]]) ###############################################
                    #D2 = np.asarray([[data_out[5],data_out[7],[data_out[4],data_out[6]]) ###############################################
                    
   #                 plt.imshow(largest_component, cmap='gray')
    #                plt.plot(D1[1, :], D1[0, :], lw=2, c='r')
     #               plt.plot(D2[1, :], D2[0, :], lw=2, c='b')
      #              plt.show()
    
                    print(z_slice, D1[1, :], D1[0, :], D2[1, :], D2[0, :])

                else:
                    D1 = np.asarray([[0, 0], [0, 0]])
                    D2 = np.asarray([[0, 0], [0, 0]])
                    print(z_slice, D1[1, :], D1[0, :], D2[1, :], D2[0, :])
                    data_out = [0,0,0,0,0,0,0,0]
                
                Bidimensional_output[:,z_slice] = data_out
                #print(Bidimensional_output[:,z_slice])
                
                if major_diameter is not None:
                    max_2ds += [major_diameter]
    
    return Bidimensional_output

    #if max_2ds == None:
    #    max_2ds = 0
    #    
    #if calc_multiple:
    #    return max_2ds
    #else:
    #    return max(max_2ds)


def calc_rano_points(binary_image, tol=0.01, output_file=None, background_image=None, verbose=False):

    """ Code developed by James Brown, postdoctoral fellow at the QTIM lab.
    """

    # Estimate lesion height and width
    height, width = np.sum(np.max(binary_image > 0, axis=1)), np.sum(np.max(binary_image > 0, axis=0))
    #print("height",height)
    #print("width",height)
    # Dilate slightly to prevent self-intersections, and compute contours
    dilated = binary_erosion(binary_image, disk(radius=1)).astype('uint8') * 255
    contours = find_contours(dilated, level=1)

    if len(contours) == 0:
        if verbose:
            print("No lesion contours > 1 pixel detected.")
        return [0,0,0,0],[0,0]

    # Calculate pairwise distances over boundary
    outer_contour = np.round(contours[0]).astype(int)  # this assumption should always hold...
    
    #print(" ")
    #print("Width", width)
    #print(" ")
    
    #euc_dist_matrix, ordered_diameters = compute_pairwise_distances(outer_contour, outer_contour, min_length=width)
    euc_dist_matrix, ordered_diameters = compute_pairwise_distances(outer_contour, outer_contour, min_length=10)

    # Exhaustive search for longest valid line segment and its orthogonal counterpart
    try:
        q = [0,0,0,0]
        d = [0,0]
        q,d = find_largest_orthogonal_cross_section(ordered_diameters, binary_image, tolerance=tol)
        #print("Q",q)
    except TypeError:
        if verbose:
            print("Error: unable to compute RANO measurement")
        return [0,0,0,0],[0,0]
    return q,d

def dataread(path):
    d = []
    f = []
    
    for (dir_path, dir_names, file_names) in walk(path):
        # gets rid of any pesky leftover .ipynb_checkpoints files
        if not dir_names == []:
            if not dir_names[0].startswith("."):
                f.extend(file_names)
                d.extend(dir_names)
    return d

if __name__ == "__main__":
    
    path = "Brats_2018_data_split/Training/LGG/"

    d = dataread(path)
    
    output_size = len(d)
    print(output_size)
    
    # output_size = 1
    
    for x in range(output_size):
        print("Data item:",x)
        data_Plot = nib.load(path + d[x] + "/" + d[x] + "_whseg.nii.gz")
        input_2 = data_Plot.get_fdata()
        
        Bi_output = calc_2D_RANO_measure(input_2, pixdim=(240,240), affine=None, mask_value=0, axis=2, calc_multiple=False, background_image=None, output_filepath="output_rano", verbose=False)
        
        np.savez(path + d[x] + "/" + d[x] + "_RANO",RANO=Bi_output)
        