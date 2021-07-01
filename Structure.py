# -*- coding: utf-8 -*-

import logging
import vtk
from vtk.util import numpy_support

import numpy as np

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# own modules
from dcmreader import Contour
from geom import Voxel


class Structure:

    def __init__(self, dcm, structure):
        """ readin in structure and create polydata according to
        https://github.com/malaterre/GDCM/blob/master/Utilities/VTK/vtkGDCMPolyDataReader.cxx
        """
        self.logger = logging.getLogger("Structure")
        self.index = structure["index"]
        self.name = structure["name"]
        self.logger.info("Load structure " + self.name)
        self.debug = []
        cnts_slices, cnts = self.readContourData(dcm)
        if self.debug == []:
            self.slices = cnts_slices[:]
            self.slices = [round(x, 2) for x in self.slices]
            pts_subcontours = []
            voxels = []

            for iSlice in range(len(self.slices)):
                for iSubContour in range(1, len(cnts[iSlice])): # first entry is the slice position
                    pts_subcontours.append([iSlice, iSubContour, len(cnts[iSlice][iSubContour][0])])
                    for iPoint in range(len(cnts[iSlice][iSubContour][0])):
                        voxels.append([round(cnts[iSlice][iSubContour][0][iPoint], 2),
                                round(cnts[iSlice][iSubContour][1][iPoint], 2),
                                self.slices[iSlice]])
            # unreasonable reasons but some times there are CTs with different resolution
            uniques = list(set(abs(np.round(np.ediff1d(np.array(self.slices)), 3))))
            uniques.sort()
            if len(uniques) > 1:
                # check for missing slices using modulo:
                mod_result = [round(uniques[x] % uniques[0], 1) for x in range(1, len(uniques))]
                if np.array(mod_result).any():
                    self.logger.info("Unique Slices \t\t%s", uniques)
                    self.debug.append(["Contour slices at different resolution ", uniques])

            self.voxels = voxels
            self.pts_subcontours = pts_subcontours
            self.createPolyData()
            self.getExtent()
            self.centerOfMass = self.calculateCenterofMass()

    def createPolyPoints(self):
        """ creates a poly points object 

        Arguments:
            None
        Returns:
            None"""
        appendFilter = vtk.vtkAppendPolyData()

        beforeSlice = -1
        countPoints = 0
        for iSlice, iSubContour, nPoints in self.pts_subcontours:
            vertexIndices = np.where(np.array(self.voxels)[:,2] == self.slices[iSlice])[0]
            if iSlice != beforeSlice:
                countPoints = 0
                beforeSlice = iSlice
            subContour = Contour()
            subContour.createPolyPoints(vertexIndices[countPoints:countPoints + nPoints], self.voxels)
            appendFilter.AddInputData(subContour.polyData)
            countPoints += nPoints

        appendFilter.Update()
        contours = appendFilter.GetOutput()

        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        contourPoints = contours.GetPoints()
        numPoints = contourPoints.GetNumberOfPoints()
        points.SetNumberOfPoints(numPoints);
        for i in range(numPoints):
            pt = list(contourPoints.GetPoint(i))
            points.SetPoint(i, pt);
        poly.SetPolys(contours.GetPolys())
        poly.SetPoints(points)
        self.polyData = poly # self.convertToLines(poly)
        self.modifiedPolyData = self.polyData
        self.savePolyData(r"K:\RAO_Physik\Research\1_FUNCTIONAL IMAGING\7_SAKK_Lung_Study\P4_spatial_distribution\results")
        self.calculateCenterofMass()
        self.logger.info("Center of Mass %s", self.centerOfMass.getRounds())

    def savePolyData(self, outPath):
        """ save poly data as vtp file

        Arguments:
            outPath         path of the output
        Returns:
            None
        """
        polyDataWriter = vtk.vtkXMLPolyDataWriter()
        polyDataWriter.SetInputData(self.modifiedPolyData)
        polyDataWriter.SetFileName(outPath + "\\" + self.name + ".vtp")
        polyDataWriter.SetCompressorTypeToNone()
        polyDataWriter.SetDataModeToAscii()
        polyDataWriter.Write()

    def createMask(self, res, origin, dim):
        """ creates a binary mask array from structure using vtk module 
        
        Arguments:
            res         image resolution
            origin      image origin
            dim         image dimensions
        Returns:
            None
        """
        self.logger.info("Init empty mask")
        dim = dim.getArray()
        mask = vtk.vtkImageData()
        mask.SetSpacing(res.getArray())
        mask.SetOrigin(origin.getArray())
        mask.SetDimensions(dim)
        mask.SetExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1)

        mask.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,1);

        count = mask.GetNumberOfPoints()
        for i in range(count):
            mask.GetPointData().GetScalars().SetTuple1(i, 1)
        self.logger.info("extruding")
        extruder = vtk.vtkLinearExtrusionFilter()
        extruder.SetInputData(self.modifiedPolyData)
        extruder.SetExtrusionTypeToNormalExtrusion()
        extruder.CappingOn();  # capp on top
        extruder.SetVector(res.getArray());
        extruder.Update();
            
        self.logger.info("stencil mask")
        # Just get the outer extent
        dataToStencil = vtk.vtkPolyDataToImageStencil()
        dataToStencil.SetInputConnection(extruder.GetOutputPort()) # stripper
        dataToStencil.SetTolerance(1e-3)
        dataToStencil.SetInformationInput(mask)
        dataToStencil.SetOutputSpacing(res.getArray())
        dataToStencil.SetOutputOrigin(origin.getArray())
        dataToStencil.Update()

        stencil = vtk.vtkImageStencil()
        stencil.SetInputData(mask)
        stencil.SetStencilConnection(dataToStencil.GetOutputPort())
        stencil.ReverseStencilOff()
        stencil.SetBackgroundValue(0)
        stencil.Update()
        
        # flip the image in Y and Z directions
        flip = vtk.vtkImageReslice() 
        flip.SetInputConnection(stencil.GetOutputPort())
        flip.SetResliceAxesDirectionCosines(1,0,0, 0,-1,0, 0,0,-1);
        flip.Update();    

        # convert mask image data to numpy array with correct dimensions
        vtk_data = flip.GetOutput().GetPointData().GetScalars()
        temp_data = np.zeros((dim[2], dim[1], dim[0]), dtype = np.int16)
        temp_data = numpy_support.vtk_to_numpy(vtk_data).reshape(dim[2], dim[1], dim[0]).astype(np.int16)
        temp_data = temp_data.transpose(2,1,0)
        temp_data = np.flip(temp_data, 2)
        self.mask = temp_data

    def calculateVoxVolume(self, res, origin, dim):
        """ calculates the volume (number of voxels) of structure mask
        
        Arguments: :
            res         image resolution
            origin      image origin
            dim         image dimensions
        Returns:
            None
        """
  
        if not hasattr(self, "mask"):
            self.createMask(res, origin, dim)

        if len(np.unique(self.mask)) < 2:
            print ("Values ", np.unique(self.mask))
            print ("No correct mask stenciled. Exit")
            return
        self.voxVolume = len(np.where(self.mask == 1)[0])

    def getVoxVolume(self, res, origin, dim):
        self.voxVolume = 0.
        self.calculateVoxVolume(res, origin, dim)
        return self.voxVolume

    def createPolyData(self, scale_factor=Voxel(1.0, 1.0, 1.0)):
        """ To create a polydata in vtk of the contour points we need two things:
        1. a geometry:  Describes single entities, such as points
        2. a topology:  Describes the connections of single entities, i.e.
                            how are the points connected?
        Arguments:
            scale_factor    factor (x,y,z) to scale the polydata
        Returns:
            None
        """
        append = vtk.vtkAppendPolyData()

        beforeSlice = -1
        for iSlice, iSubContour, nPoints in self.pts_subcontours:
            if iSlice != beforeSlice:
                countPoints = 0
                beforeSlice = iSlice

            vertexIndices = np.where(np.array(self.voxels)[:,2] == self.slices[iSlice])[0]
            linePoints = vertexIndices[countPoints:countPoints + nPoints]
            linePoints = np.append(linePoints, linePoints[0])
            # create a contour for each subcontour
            subContour = Contour()
            subContour.createPolyLine(linePoints, self.voxels)
            subContour.scale(scale_factor)
            append.AddInputData(subContour.modifiedPolyData)
            countPoints += nPoints

        append.Update()
        polyData = append.GetOutput()
        if set(scale_factor.getArray()) == set([1.0, 1.0, 1.0]):
            self.polyData = polyData
        self.modifiedPolyData = polyData
        self.calculateCenterofMass()
        self.scaleZ(scale_factor)
        self.logger.info("Center of Mass \t\t%s", self.centerOfMass.getRounds())


    def createSurface(self):
        """ create surface object from polydata using ruled Surface filter

        Arguments:
            None
        Returns:
            ruledSurfaceFilter      filter object containing the surface

        """
        ruledSurfaceFilter = vtk.vtkRuledSurfaceFilter()
        ruledSurfaceFilter.SetInputData(self.modifiedPolyData)
        ruledSurfaceFilter.SetResolution(50, 50)
        ruledSurfaceFilter.SetRuledModeToResample()
        ruledSurfaceFilter.Update()
        self.getExtent()
        return ruledSurfaceFilter

    def readContourData(self, dcm):
        """ process dcm file to extract contour points of structure set
        Arguments:
            dcm                 pydicom object of RTST
        Returns:
            slices              list of slice position of the structure
            contourdataPoints   physical word coordinates of the contour points
        """
        contourData = []
        if "ContourSequence" in dcm.ROIContourSequence[self.index]:
            for iSubContour in range(len(dcm.ROIContourSequence[self.index].ContourSequence)):
                contourData.append([float(dcm.ROIContourSequence[self.index].ContourSequence[iSubContour].ContourData[2]),
                                    dcm.ROIContourSequence[self.index].ContourSequence[iSubContour].ContourData[::3],
                                    dcm.ROIContourSequence[self.index].ContourSequence[iSubContour].ContourData[1::3]])
            slices = [contourData[iSlice][0] for iSlice in range(len(contourData))]
            if len(slices) > 1:
                if slices[1] - slices[0] < 0:
                    contourData.sort()
                contourdataPoints = self.multiContour(contourData)
                slices = [contourdataPoints[iSlice][0] for iSlice in range(len(contourdataPoints))]
                return slices[:], contourdataPoints[:]
            else:
                self.logger.info("Only one Slice found")
                self.debug.append("Contour contains one single slice")
                return slices[:], []
        else:
            self.logger.info("No Contour sequence found")
            self.debug.append(["Structure " + self.name + " does not contain a contour sequence"])
            return [], []

    def multiContour(self, contourData):
        """ account for mutlicontours in one slice,
        checks z positions in each sublist of the list and if the have the same z then creats a new sublist
        for example input l = [[z1, [x1, x2],[y1,y2]], [z1, [x3, x4, x5],[y3, y4, y5]], [z2, [x1, x2],[y1,y2]]] - 3 contours on 2 slices
        output l = [[z1, [[x1, x2],[y1,y2]], [[x3, x4, x5],[y3, y4, y5]]], [z2, [[x1, x2],[y1,y2]]]]

        Arguments:
            contourData         point coordinates of contours in old format
        Returns:
            kontur              points coordinates in new format
        """

        listap=[]
        lista_nr=[]
        for contourpoint in contourData:
            listap.append(contourpoint[0]) # skice
            if contourpoint[0] not in lista_nr:
                lista_nr.append(contourpoint[0]) # insert slice point into list
        counts = []
        for i in lista_nr:
            counts.append(listap.count(i)) #how many times a ceratin z position occurs on the list

        listka=[]
        nr=0
        kontur = []
        for i in range(len(contourData)):
            if contourData[i][0] not in listka:
                m=[contourData[i][0]]
                for j in range(counts[nr]):
                    m.append([np.array(contourData[i+j][1], dtype=np.float), np.array(contourData[i+j][2], dtype=np.float)])
                    listka.append(contourData[i][0])
                kontur.append(m)
                nr+=1
        return kontur

    def calculateCenterofMass(self):
        """ Compute center of mass using polyData

        Arguments:
            None
        Return:
            center of mass
        """
        centerOfMassFilter = vtk.vtkCenterOfMass()
        centerOfMassFilter.SetInputData(self.modifiedPolyData)
        centerOfMassFilter.SetUseScalarsAsWeights(False)
        centerOfMassFilter.Update()

        com = np.ones([3])
        com = centerOfMassFilter.GetCenter()
        self.centerOfMass = Voxel(com[0], com[1], com[2])

        return self.centerOfMass

    def processPolyData(self, shift, factor):
        """ processes Polydata using shift and factor to scale and translate

        Arguments:
            shift           Voxel object shift in x, y, z
            factor          Voxel object scale factor in x, y, z
        Returns:
            None
        """
        self.logger.info(self.name + ": Process PolyData")

        self.scalePolyData(factor)
        self.translate(shift)
        self.calculateCenterofMass()

    def translate(self, shift):
        """ translate polydata to new coordinate and update modifiedPolyData

        Arguments:
            shift           Voxel object shift in x, y, z
        Returns:
            None
        """
        translation = vtk.vtkTransform()
        translation.Translate(shift.x, shift.y, shift.z)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(self.modifiedPolyData)
        transformFilter.SetTransform(translation)
        transformFilter.Update()
        self.modifiedPolyData = transformFilter.GetOutput()
        self.centerOfMass = self.calculateCenterofMass()

    def scalePolyData(self, factor):
        """ scale polydata by factor in 2 dimensions

        Arguments:
            factor           Voxel object scale in x, y
        Returns:
            None
        """
        self.calculateCenterofMass()
        self.createPolyData(factor)

    def scaleZ(self, factor):
        """ scale polydata by factor in z dimensions

        Arguments:
            factor           Voxel object stretch in z
        Returns:
            None
        """
        # since the scaling is performed in terms of the origin (0,0,0), the structure is
        # first translated to origin by subtracting center of mass, then scaled
        # and finally translate back by add the center of mass
        # see https://stackoverflow.com/questions/55813955/how-to-scale-a-polydata-in-vtk-without-translating-it

        com = self.centerOfMass

        self.translate(com.negate())

        scale = vtk.vtkTransform()
        scale.Scale(1.0, 1.0, factor.z)
        scale_TF = vtk.vtkTransformPolyDataFilter()
        scale_TF.SetInputData(self.modifiedPolyData);
        scale_TF.SetTransform(scale);
        scale_TF.Update()
        self.modifiedPolyData = scale_TF.GetOutput()

        self.translate(com)
        self.centerOfMass = self.calculateCenterofMass()


    def calculateProperties(self):
        """ calculates properties of the polygon

        Arguments:
            None
        Return:
            None
        """
        self.logger.info("  " + self.name + ": Calculate shape properties" )
        self.centerOfMass = self.calculateCenterofMass()
        self.volume, self.surface = self.calculateMassProperties()

    def getExtent(self):
        """ get extent of polydata

        Arguments:
            None
        Return:
            None
        """
        self.bounds = self.modifiedPolyData.GetBounds()




