# -*- coding: utf-8 -*-

import numpy as np

class Voxel:
    def __init__(self, x,y=np.nan,z=np.nan):
        """ init Voxel
        Arguments: 
            x       x position
            y       y position
            z       z position
        Returns:
            None
        """
        
        if isinstance(x, (list,)) or isinstance(x, np.ndarray) or isinstance(x, tuple):
            self.x = x[0]
            self.y = x[1]
            self.z = x[2]
            
        else:
            self.x, self.y, self.z = x,y,z
                
    def add(self, v):
        """ add two Voxel objects
        Arguments: 
            v       voxel obect to be added to self
        Returns:
            _v      sum of self and v
        """
        
        _v = Voxel(self.x + v.x, 
                         self.y + v.y, 
                         self.z + v.z)
        return _v
    
    def mult(self, x):
        """ multiplicate two Voxel objects componentwise of scalar
        Arguments: 
            x          voxel obect or scalar
        Returns:
            _v         scalar or componentwise multipication
        """
        
        if isinstance(x, float) or isinstance(x, int):
            _v = Voxel(self.x * x, 
                       self.y * x, 
                       self.z * x)
        else:
            _v = Voxel(self.x * x.x, 
                       self.y * x.y, 
                       self.z * x.z)
            
        return _v
    def subtract(self, v):
        """ subtract two Voxel objects
        Arguments: 
            v       voxel obect to be subtracted from self
        Returns:
            _v      subtraction of self - v
        """
        
        _v = Voxel(self.x - v.x, 
                   self.y - v.y, 
                   self.z - v.z)
        return _v
    
    def divide(self, denom):
        """ returns scalar division or componentwise division of self by denom
        Arguments: 
            v       voxel obect to be subtracted from self
        Returns:
            _v      division of self / denom either componentwise or by scalar
        """
        
        assert denom 
        if isinstance(denom, Voxel):

            if ~np.isnan(denom.z):
                _v = Voxel(self.x / denom.x, 
                           self.y / denom.y,
                           self.z / denom.z)
            else: 
                _v = Voxel(self.x / denom.x, 
                           self.y / denom.y,
                           self.z)
        else:
            assert denom != 0, "Division by 0 undefined"
            _v = Voxel(self.x / float(denom), 
                       self.y / float(denom),
                       self.z / float(denom))
        return _v
    
    def negate(self):
        
        return Voxel(-self.x, -self.y, -self.z)
    
    def getArray(self):
        """ returns numpy array of self positions
        Arguments: 
            None
        Returns:
            array      numpy array from Voxel positions
        """
        return np.array([self.x, self.y, self.z])
    
    def getRounds(self):
        """ returns numpy array of self positions
        Arguments: 
            None
        Returns:
            array      numpy array from Voxel positions
        """
        return np.array([round(self.x, 2), round(self.y,2), round(self.z, 2)])
    
    def getints(self):
        """ returns Voxel object with integer values
        Arguments: 
            None
        Returns:
            _v      Voxel object with integer values
        """
        
        _v = Voxel(int(self.x), int(self.y), int(self.z))
        return _v
        
    def getLabel(self):
        """ returns Voxel object label
        Arguments: 
            None
        Returns:
            self.label      
        """
        return self.label
    
    
class Grid:
    def __init__(self, origin, res, dim, points=[]):
        """ initialize Grid object: each grid contains a resolution of voxel, 
        its origin in the world frame and the dimension of the grid"
        """
        self.res = res
        self.origin = origin
        self.dim = dim
        indices = self.getIndices()
        if points!=[]:
            self.points = points
        else:
            self.initiate()
        self.totalPoints = len(indices[0])

    def getIndices(self):
        """ get indices of Grid object
        
        Arguments: 
            None
        Returns:
            indices         indices array
        """
        indicesX, indicesY, indicesZ = np.meshgrid(range(int(self.dim.x)), range(int(self.dim.y)), range(int(self.dim.z)))
        indices = np.array([indicesX.ravel(), indicesY.ravel(), indicesZ.ravel()])    
        return indices
    
    def initiate(self):
        """ initiate a Grid
        
        Arguments: 
            None
        Returns:
            None
        """
        indices = self.getIndices()
        pointsX = self.origin.x + indices[0] * self.res.x
        pointsY = self.origin.y + indices[1] * self.res.y
        pointsZ = self.origin.z + indices[2] * self.res.z
        self.points =  np.array([pointsX.ravel(), pointsY.ravel(), pointsZ.ravel()])
        
    def getPoints(self, mode = "array"):
        """ get points of Grid
        
        Arguments: 
            mode        
        Returns:
            None
        """
        if mode == "homogeneous":
            self.points = [self.points[0].ravel(), self.points[1].ravel(), self.points[2].ravel(), np.ones([self.totalPoints])]
        return self.points
    
    def getExtent(self):
        """ get grid extent
        
        Arguments: 
            None
        Returns:
            None
        """
        self.extent = [min(self.points[0]), max(self.points[0]),
                       min(self.points[1]), max(self.points[1]),
                       min(self.points[2]), max(self.points[2])]
    def setPoints(self, points):
        """ set grid points
        
        Arguments: 
            points
        Returns:
            None
        """
        self.points = points
        self.totalPoints = len(self.points[0])
        
    def getPoint(self, index):
        """ set grid point based on index
        
        Arguments: 
            index       index of point to be evaluated
        Returns:
            None        point at this index
        """
        return Voxel([self.points[0][index], self.points[1][index], self.points[2][index]])

    
    def isin(self, point):
        """ checks if point is in grid
        
        Arguments: 
            point       point to be evaluated
        Returns:
            isin        bool in or out of grid
        """
        self.getExtent()
        isin = ((point.x > self.extent[0] and point.x < self.extent[1]) and 
                (point.y > self.extent[2] and point.y < self.extent[3]) and 
                (point.z > self.extent[4] and point.z < self.extent[5]))
        return isin

        