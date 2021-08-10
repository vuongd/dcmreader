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
        self.debug = []
        self.dcm = dcm
        self.unit = dcm.DoseUnits
        self.sumType = dcm.DoseSummationType
        self.type = dcm.DoseType
        self.res = Voxel(float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]), float(dcm.SliceThickness))
        # x and z are swapped in the pixel_array
        pixelData = np.frombuffer(dcm.PixelData, dtype=np.float16)
        #self.dim = Voxel(int(dcm.Rows), int(dcm.Columns), int(pixelData.shape[0] / (int(dcm.Rows) * int(dcm.Columns))))
        #print(self.dim.getArray())
        cube = dcm.pixel_array * float(dcm.DoseGridScaling)#pixelData.reshape((self.dim.getArray())) * float(dcm.DoseGridScaling) # get dose in gray units

        cube = np.swapaxes(cube, 0, 2) # for some reason the dose cube has swoped axis y and z
        self.dim = Voxel(cube.shape)
        self.cube = cube
        self.origin = Voxel(float(dcm.ImagePositionPatient[0]), float(dcm.ImagePositionPatient[1]), float(dcm.ImagePositionPatient[2]))
        self.nFrames = dcm.NumberOfFrames

        if "ReferencedRTPlanSequence" in dcm:
            plan = dcm.ReferencedRTPlanSequence[0]
            if "ReferencedFractionGroupSequence" in plan:
                fractionGroup = plan.ReferencedFractionGroupSequence[0]
                self.fractionNumber = int(fractionGroup.ReferencedFractionGroupNumber)
                if "ReferencedBeamSequence" in fractionGroup:
                    self.beamNumber = int(fractionGroup.ReferencedBeamSequence[0].ReferencedBeamNumber)
                    if "ReferencedControlPointSequence" in fractionGroup.ReferencedBeamSequence[0]:
                        controlPointSequence = fractionGroup.ReferencedBeamSequence[0].ReferencedControlPointSequence[0]
                        self.controlPointStart = int(controlPointSequence.ReferencedStartControlPointIndex)
                        self.controlPointStop = int(controlPointSequence.ReferencedStopControlPointIndex)
                
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
        print(np.unique(self.cube))
        print(np.unique(cube_i))
        print("image shape", dim.getArray())
        print("cube shape", self.cube.shape)
        print("cube_i shape", cube_i.shape)
        print("cube res", self.res.getArray())
        print("cube_i res", res.getArray())
        print("cube origin", self.origin.getArray())
        print("image origin", origin.getArray())
        cube_i_tmp = np.empty(dim.getArray())
        
        origin_shift = self.origin.subtract(origin)
        origin_shift = origin_shift.divide(res).getints()
        print("origin shift", origin_shift.getArray())

        xmin = origin_shift.x if origin_shift.x > 0 else 0
        ymin = origin_shift.y if origin_shift.y > 0 else 0
        zmin = origin_shift.z if origin_shift.z > 0 else 0
        
        xmax = cube_i.shape[0]+origin_shift.x if cube_i.shape[0]+origin_shift.x < dim.x else dim.x
        ymax = cube_i.shape[1]+origin_shift.y if cube_i.shape[1]+origin_shift.y < dim.y else dim.y
        zmax = cube_i.shape[2]+origin_shift.z if cube_i.shape[2]+origin_shift.z < dim.z else dim.z
        print((xmax, xmin), (ymax, ymin), (zmax, zmin))
        cube_i_tmp[xmin:xmax,
                   ymin:ymax,
                   zmin:zmax] = cube_i[:(xmax-xmin),:(ymax-ymin),:(zmax-zmin)]
        
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
    
    
