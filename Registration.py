# -*- coding: utf-8 -*-

import logging
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))    
from geom import Voxel, Grid

class Registration:
    
    def __init__(self, dcm):
        self.logger = logging.getLogger("Reg")
        self.UID = ['1.2.840.10008.5.1.4.1.1.66.3','1.2.840.10008.5.1.4.1.1.66.1', "REG"]
        assert dcm.SOPClassUID in self.UID, "Caution: Wrong Modality input"
        
        self.PatientName = dcm.PatientName
        self.type = ""

class Deformable(Registration):
    
    def __init__(self, dcm):
        Registration.__init__(self, dcm)
        self.logger = logging.getLogger("DeformReg")
        assert dcm.SOPClassUID in self.UID, "Caution: Wrong Modality input"
        self.dcm = dcm
        self.type = "deformable"
        
    def load(self):
        """ load deformable registration file and extract deformation grid source and target
        Arguments: 
            None
        Returns:
            sourceGrid      Grid of source frame
            targetGrid      Grid of target frame
        """
        
        # get the dcm deformable image registration sequence
        deformReg = self.dcm.data_element("DeformableRegistrationSequence")[1]
        
        # get post rigid matrix transformation
        M_post_seq = deformReg.data_element("PostDeformationMatrixRegistrationSequence")[0]
        M_post = np.array(M_post_seq[0x3006,0x00C6].value).reshape(4,4)
        self.logger.debug ("M_post matrix %s", M_post)
        
        # get pre rigid matrix transformation
        M_pre_seq = deformReg.data_element("PreDeformationMatrixRegistrationSequence")[0]
        M_pre = np.array(M_pre_seq[0x3006,0x00C6].value).reshape(4,4)
        self.logger.debug ("M_pre matrix %s", M_pre)

        # get deformable vector field grid origin and res
        gridSeq = deformReg.data_element("DeformableRegistrationGridSequence")[0]
        self.gridOrigin = Voxel(np.array(gridSeq.data_element("ImagePositionPatient").value)) # per definition the center of the voxel
        self.gridRes = Voxel(gridSeq.data_element("GridResolution").value)

        self.logger.debug("Grid Origin:\t\t %s", self.gridOrigin.getRounds())
        self.logger.debug("Grid Res:\t\t %s", self.gridRes.getRounds())
        
        self.dim = Voxel(np.array(gridSeq.data_element("GridDimensions").value))
        
        # get deformable vector field grid
        # number of bytes given as XD * YD * ZD * 3 * 4
        # https://dicom.innolitics.com/ciods/deformable-spatial-registration/deformable-spatial-registration/00640002/00640005/00640009
        if isinstance(gridSeq.data_element("VectorGridData").value, list):
            data = np.array(gridSeq.data_element("VectorGridData").value) # python2 reads it in as list
        else: 
            data = np.fromstring(gridSeq.data_element("VectorGridData").value, np.float32) # python2 reads it in as byte array

        data = data.reshape(int(np.prod(self.dim.getRounds())), 3)
        deltas = np.zeros((self.dim.x, self.dim.y, self.dim.z, 3))
        nextVoxel = 0
        for k in range(self.dim.z):
            for i in range(self.dim.x):
                for j in range(self.dim.y):
                    deltas[i,j,k,:] = data[nextVoxel]
                    nextVoxel +=1
        del data
        
        # determine corresponding deformation grid in source frame
        targetGrid = Grid(self.gridOrigin, self.gridRes, self.dim)
        rigidVoxels = M_pre.dot(targetGrid.getPoints("homogeneous"))
        gridDeltas = [deltas[:,:,:,0].ravel(), deltas[:,:,:,1].ravel(), deltas[:,:,:,2].ravel(), np.zeros([len(deltas[:,:,:,2].ravel())])]
        sourceGrid_data = M_post.dot(rigidVoxels + gridDeltas)[:-1]
        
        sourceGrid = Grid(self.gridOrigin, targetGrid.res, self.dim)
        sourceGrid.setPoints([sourceGrid_data[0], sourceGrid_data[1], sourceGrid_data[2]])
        return sourceGrid, targetGrid
    
    def transformToSourceFrame(self, idx):
        """ transformation of coordinates from target RCS frame to source frame
            according to DICOM Equation C.20.3-1
        
        Arguments:
            idx     index of point in the target frame
        Return:
            pts     list of flattened x, y, z coordinates of the deformation
                    grid to the source frame
        """
        sourceGrid, targetGrid = self.load()
        self.logger.info("\n\t\t\t Grid voxel before\t%s", targetGrid.getPoint(idx).getRounds())
        self.logger.info("Grid voxel after \t%s", sourceGrid.getPoint(idx).getRounds())
        pt = sourceGrid.getPoint(idx)
        return pt
        
    def transformToTargetFrame(self, vox):
        """ transformation of coordinates from source frame to target frame (RCS)
            according to DICOM Equation C.20.3-1
        
        Arguments:
            vox             3D Voxel point object which should be transformed to target frame
        Returns:
            vox_target      point in target frame
        """
        sourceGrid, targetGrid = self.load()
        from functools import reduce
        assert sourceGrid.isin(vox), "Voxel is not inside deformation grid. Extent %s" + sourceGrid.getExtent()
        
        # determine neighbouring candidate voxels    
        divisor = self.gridRes.z
        voxUpperBound = vox.add(Voxel(divisor, divisor, divisor)).getints()
        voxDownBound = vox.subtract(Voxel(divisor, divisor, divisor)).getints()
        
        ix = np.where(np.isin(sourceGrid.points[0].astype(int), np.arange(voxDownBound.x, voxUpperBound.x)))
        iy = np.where(np.isin(sourceGrid.points[1].astype(int), np.arange(voxDownBound.y, voxUpperBound.y)))
        iz = np.where(np.isin(sourceGrid.points[2].astype(int), np.arange(voxDownBound.z, voxUpperBound.z)))
        
        commonIndex = []
        commonIndex = reduce(np.intersect1d, (ix, iy, iz))
        
        if len(commonIndex) > 0:
            dist = []
            for i in commonIndex:
                testVoxel = Voxel(sourceGrid.points[0][i], sourceGrid.points[1][i], sourceGrid.points[2][i])
                dist.append(np.linalg.norm(testVoxel.getArray() - vox.getArray()))
            closestMatchIndex =  np.where(dist == min(dist))[0][0]
            vox_target = targetGrid.getPoint(commonIndex[closestMatchIndex])
            return vox_target

        else: 
            self.logger.warning("No corresponding Voxel found! Check your indices!")
            self.logger.warning("Found index in x %s", ix)
            self.logger.warning("Found index in y %s", iy)
            self.logger.warning("Found index in z %s", iz)
            self.logger.warning("common index %s", commonIndex)
            self.logger.info("extent of source grid %s", sourceGrid.extent)
        
    def getIndices(self, vox, origin, res): 
        _v = vox.subtract(origin).divide(res).getints()
        return _v
    
    
class Rigid(Registration): # inherit from Registration file
    def __init__(self, dcm):
        """ Rigid registration object, reads in mainly coordinate transformation
        matrix
        """
        Registration.__init__(self, dcm)
        
        self.logger = logging.getLogger("RigidReg")
        assert dcm.SOPClassUID in self.UID, "Caution: Wrong Modality input"
        self.load(dcm)
        self.type = "rigid"

    def load(self, dcm):
        """ load rigid registration file and extract transformation matrices
        Arguments: 
            dcm         pydicom object
        Returns:
            None
        """
        
        regSeq = dcm.data_element("RegistrationSequence")[1]
        matrixRegSeq = regSeq.data_element("MatrixRegistrationSequence")[0]
        matrixSeq = matrixRegSeq.data_element("MatrixSequence")[0]
        
        self.src2targetMatrix = np.array([matrixSeq.data_element("FrameOfReferenceTransformationMatrix").value]).reshape(4,4)
    
    def transformToSourceFrame(self, vox):
        """ transformation of coordinates from target frame to source frame
        Arguments:
            vox         to be transformed voxel (Voxel object) to source frame
        Return:
            vox_target      transformed point in the source frame
        """
        vox_target = np.linalg.inv(self.src2targetMatrix).dot([vox.x, vox.y, vox.z, 1])

        return vox_target
        
    def transformToTargetFrame(self, vox):
        """ transformation of coordinates from source frame to target frame (RCS)
    
        Arguments:
            vox             to be transformed voxel (Voxel object) to target frame
        Return:
            vox_target      transformed point in the target frame
        """
        
        return self.src2targetMatrix.dot([vox.x, vox.y, vox.z, 1])