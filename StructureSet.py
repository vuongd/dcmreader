# -*- coding: utf-8 -*-
import logging
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# own modules
import dcmreader

class StructureSet:

    def __init__(self, dcm,  contourNames = []):
        """ read in structure Set

        Arguments:
            dcm             pydicom object of rtstruct
            origin          image origin of corresponding CT
            contourNames    names of structures to be read in
        Returns:
            None
        """
        self.logger = logging.getLogger("StructureSet")

        self.UID = ['1.2.840.10008.5.1.4.1.1.481.3','RT Structure Set Storage']
        self.PatientName = dcm.PatientName
        self.nStructures = len(dcm.StructureSetROISequence)
        self.structureNames = [dcm.StructureSetROISequence[iStructure].ROIName for iStructure in range(len(dcm.StructureSetROISequence))]
        try: self.StructureSetName = dcm.StructureSetName
        except: self.StructureSetName = ""
        self.structures = dict()
        self.dcm = dcm
        if contourNames != []:
            self.readStructures(contourNames)

    def readStructures(self, structureNames):
        """ read structures in RTSTRUCT
        
        Arguments:
            structureNames  list of names of structures
        """
        self.selected_structures = structureNames
        for iStructure in structureNames:
            structure = self.getStructure(self.dcm, iStructure)
            if structure != -1:
                self.structures[iStructure] = structure

    def getStructure(self, dcm, structureName):
        """ read in structure with its contour points

        Arguments:
            dcm             pydicom object of rtstruct
            structureName   name of the structure
        Returns:
            structure       structure object
        """
        structure = -1
        try:
            self.structureNames.index(structureName)
            structure = dcmreader.Structure(dcm, structure = {"index": self.structureNames.index(structureName), "name":structureName})
        except ValueError:
            self.logger.error("Could not find structure in structure set " +  structureName)
        return structure
