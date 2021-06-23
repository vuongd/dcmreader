# -*- coding: utf-8 -*-
import vtk
import logging
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from geom import Voxel

class Contour:

    def __init__(self):
        """
        Contour object consists of stacked 2D contour slices of structures
        """
        self.logger = logging.getLogger("Contour")
        return

    def setPolyData(self, pd):
        """ set polydata of contour object manually
        Arguments:
            pd          vtk PolyData object
        Returns:
            None
        """
        self.polyData = pd
        self.modifiedPolyData = pd

    def createPolyPoints(self, pts, voxels):
        """ create poly data using by inserting points
        Arguments:
            pts             list of points
            voxels          array of all points in 3D
        Returns:
            None
        """
        polyData = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        points.SetNumberOfPoints(len(pts))
        cells.Allocate(1, len(pts))
        cells.InsertNextCell(len(pts))

        for iPoint, point in enumerate(pts):
            points.SetPoint(iPoint, voxels[point][0], voxels[point][1], voxels[point][2])
            cells.InsertCellPoint(iPoint)
        polyData.Initialize()
        polyData.SetPolys(cells)
        polyData.SetPoints(points)
        self.polyData = polyData
        self.modifiedPolyData = polyData
        self.centerOfMass = self.calculateCenterofMass()

    def createPolyLine(self, linePoints, voxels):
        """ create poly data using by creating for each contour a polyline
        Arguments:
            linePoints      list of points of each contour
            voxels          array of all points in 3D
        Returns:
            None
        """
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        lineArray = vtk.vtkCellArray()
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(linePoints));

        for i, point in enumerate(linePoints):
            points.InsertNextPoint(voxels[point])
            line.GetPointIds().SetId(i, i)
        lineArray.InsertNextCell(line)

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(lineArray)
        self.polyData = polyData
        self.modifiedPolyData = polyData
        self.centerOfMass = self.calculateCenterofMass()

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

    def scale(self, factor):
        """ scale polydata by factor in 2 dimensions

        Arguments:
            factor           Voxel object scale in x, y
        Returns:
            None
        """
        # since the scaling is performed in terms of the origin (0,0,0), the structure is
        # first translated to origin by subtracting center of mass, then scaled
        # and finally translate back by add the center of mass
        # see https://stackoverflow.com/questions/55813955/how-to-scale-a-polydata-in-vtk-without-translating-it
        com = self.centerOfMass

        self.translate(com.negate())
        scaleVTK = vtk.vtkTransform()
        scaleVTK.Scale(factor.x, factor.y, 1.0)
        scale_TF = vtk.vtkTransformPolyDataFilter()
        scale_TF.SetInputData(self.modifiedPolyData);
        scale_TF.SetTransform(scaleVTK);
        scale_TF.Update()
        self.modifiedPolyData = scale_TF.GetOutput()

        self.translate(com)
        self.centerOfMass = self.calculateCenterofMass()

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
