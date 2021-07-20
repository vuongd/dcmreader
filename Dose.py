import logging
import sys, os
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interpn
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# own modules
import dcmreader
from geom import Voxel
from skimage.transform import rescale

class Dose:
    def __init__(self, dcm):
        
        self.logger = logging.getLogger("Dose")
        self.dcm = dcm
        self.unit = dcm.DoseUnits
        self.sumType = dcm.DoseSummationType
        self.type = dcm.DoseType
        self.res = Voxel(float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]), float(dcm.SliceThickness))
        # x and z are swapped in the pixel_array
        cube = dcm.pixel_array * float(dcm.DoseGridScaling) # get dose in gray units
        cube = np.swapaxes(cube, 0, 2).transpose(1, 0, 2)
        self.offset = np.array(dcm.ImagePositionPatient, dtype="float")
        print(self.offset)
        self.dim = Voxel(cube.shape)
        self.cube = cube
        self.origin = Voxel(float(dcm.ImagePositionPatient[0]), float(dcm.ImagePositionPatient[1]), float(dcm.ImagePositionPatient[2]))

    def interpolateCube(self, res, origin, dim):
        """ interpolates the dose cube
        
        Arguments:
            res         resolution the dose cube should be interpolated to
            origin      origin of object to be interpolated to
            dim         dimension of the object to be interpolated to
        Returns:
            None
        """
        scale = self.res.divide(res).getArray()
        cube_i = rescale(
            self.cube,
            scale=scale,
            order=1,
            mode='symmetric',
            preserve_range=True,
            multichannel=False
        )
        print("cube shape", self.cube.shape)
        print("cube_i shape", cube_i.shape)
        print("cube res", self.res.getArray())
        print("cube_i res", res.getArray())
        print("cube origin", self.origin.getArray())
        print("cube_i origin", origin.getArray())
        cube_i_tmp = np.empty(dim.getArray())
        
        origin_shift = self.origin.subtract(origin)
        origin_shift = origin_shift.divide(res).getints()
        print("origin shift", origin_shift.getArray())

        if origin_shift.z < 0:
            cube_i_tmp[origin_shift.y:cube_i.shape[0]+origin_shift.y, 
                       origin_shift.x:cube_i.shape[1]+origin_shift.x,
                       0:dim.z+1] = cube_i[:,:,:dim.z]
        else:
            cube_i_tmp[origin_shift.y:cube_i.shape[0]+origin_shift.y, 
                   origin_shift.x:cube_i.shape[1]+origin_shift.x,
                   origin_shift.z:cube_i.shape[2]+origin_shift.z] = cube_i
        print(origin_shift.getArray())
        
        self.cube_i = cube_i_tmp

    def calculateDVH(self, mask):
        """ calculates the dose volume histogram 
        
        Arguments:
            mask        binary mask of region the dose should be evaluated from
        Returns:
            dvh         dose volume histogram
        """
        if np.isnan(mask).any() or 0 in np.unique(mask[:]):
            mask[np.isnan(mask)] = 0.
            mask[mask!=0] = 1.
        maxDose = np.nanmax(self.cube_i)
        mask_cube = self.cube_i * mask
        dvh = [np.count_nonzero(mask_cube >= float(thres)) / float(np.count_nonzero(mask_cube)) for thres in range(0, maxDose)]
        return dvh
    
    def getMeanDose(self, mask):
        """ calculates the mean dose for a given mask.  
        
        Arguments:
            mask        binary mask of region the dose should be evaluated from
        Returns:
            mean_dose        mean dose 
        """
        
        mean_dose = np.nanmean(self.cube_i * mask)
        return mean_dose
