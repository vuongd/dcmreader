import logging
import sys, os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# own modules
import dcmreader
from geom import Voxel

class Dose:
    def __init__(self, dcm):
        """ readin in structure and create polydata according to
        https://github.com/malaterre/GDCM/blob/master/Utilities/VTK/vtkGDCMPolyDataReader.cxx
        """
        self.logger = logging.getLogger("Dose")
        self.dcm = dcm
        self.unit = dcm.DoseUnits
        #print(dcm)
        self.sumType = dcm.DoseSummationType
        self.type = dcm.DoseType
        self.res = Voxel(float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]), float(dcm.SliceThickness))
        cube = dcm.pixel_array * float(dcm.DoseGridScaling) # get dose in gray units
        self.dim = Voxel(cube.shape)
        self.cube = cube
        #print(np.unique(dcm.pixel_array))
        #print("cube")
        #print(np.unique(cube))
        self.origin = Voxel(float(dcm.ImagePositionPatient[0]), float(dcm.ImagePositionPatient[1]), float(dcm.ImagePositionPatient[2]))


    def getInterpolatedCube(self, res):
        old_x = self.origin.x + self.res.x * np.arange(self.dim.x)
        old_y = self.origin.y + self.res.y * np.arange(self.dim.y)
        old_z = self.origin.z + self.res.z * np.arange(self.dim.z)

        interpol_function = RegularGridInterpolator((old_x, old_y, old_z), self.cube, bounds_error=False, fill_value = 0.)
        interpolated_cube = interpol_function((res.x, res.y, res.z))
        self.interpolated_cube = interpolated_cube
        #print("interpolated cube")
        #print(interpolated_cube)

    def calculateDVH(self, mask):
        if np.isnan(mask).any() or 0 in np.unique(mask[:]):
            mask[np.isnan(mask)] = 0.
            mask[mask!=0] = 1.
        maxDose = max(self.interpolated_cube)
        tmp_cube = self.interpolated_cube * mask
        dvh = [np.count_nonzero(tmp_cube >= float(thres)) / float(np.count_nonzero(tmp_cube)) for thres in range(0, maxDose)]
        return dvh
