# -*- coding: utf-8 -*-

import pydicom as dc
import logging
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# own modules
from geom import Voxel
import dcmreader
import vtk
from glob import glob

class Patient:

    def __init__(self, **kwargs):
        self.logger = logging.getLogger("Patient")
        self.debug = []
        dirDict = kwargs["dirDict"]
        #assert type(dirDict) is dict,  "{} need an dict input for directory".format(self)
        def rename_keys(d, keys):
            return dict([(keys.get(k), v) for k, v in d.items()])
        new_keys = {'img' : 'CT', 'rs' : 'RTSTRUCT', 'reg' : 'REG', 'dose': 'RTDOSE', 'all':'all'}
        renamed_dirDict = rename_keys(dirDict, new_keys)
        self.dirDict = renamed_dirDict if np.any(renamed_dirDict.keys()==None) else dirDict
        self.fileDict = self.parse()
        if "parseModality" in kwargs:
            self.parseModality = kwargs["parseModality"]
        elif self.FoundTypes!=0:
            self.parseModality = self.FoundTypes
        else:
            self.debug.append("No Files found")
            return 
        self.reg = []
        self.rs = []

        if "CT" in self.parseModality:
            if "mode" in kwargs:
                self.load_scan(kwargs["mode"])
            else:
                self.load_scan("full")
        if "RTSTRUCT" in self.parseModality:
            if "contourNames" in kwargs:
                self.load_rs(kwargs["contourNames"])

        if "REG" in self.parseModality:
            self.load_reg()

        if "RTDOSE" in self.parseModality:
            self.load_dose()

        self.info()
        self.name = ""

    def setPatientName(self, name):
        """ set Patient Name

        Arguments:
            name        new patient name
        Returns:
            None
        """
        self.name = name

    def parse(self):
        """ returns directory with found filenames

        Arguments:
            None
        Returns:
            None
        """
        fileDict = {}
        self.FoundTypes = []
        if len(self.dirDict) == 1 and "all" in self.dirDict:
            if os.path.exists(self.dirDict["all"]):
                for ifile in glob(self.dirDict["all"] + os.sep + "*.dcm"):
                    dcm = dc.read_file(ifile)
                    modality = self.getUID(dcm.SOPClassUID)
                    self.FoundTypes.append(modality)
                    if modality not in fileDict:
                        fileDict[modality] = []
                    fileDict[modality].append(ifile.split(os.sep)[-1])
            else:
                self.logger.info("dirDict all directory does not exists")
                return
        else:
            for i in self.dirDict:
                fileDict[i] = []

                if i == "CT":
                    for img in glob(self.dirDict[i] + os.sep + "*.dcm"):
                        dcm = dc.read_file(img)
                        if self.getUID(dcm.SOPClassUID) == i:
                            fileDict[i].append(img.split(os.sep)[-1])
                            self.FoundTypes.append(i)
                elif os.path.exists(self.dirDict[i]):
                    for ifile in os.listdir(self.dirDict[i]):
                        dcm = dc.read_file(self.dirDict[i] + os.sep + ifile)
                        if self.getUID(dcm.SOPClassUID) == i:
                            fileDict[i].append(ifile)
                            self.FoundTypes.append(i)
                else:
                    print("Dict key {} not found while parsing".format(i))
                    self.debug.append("Dict key {} not found while parsing".format(i))
        self.FoundTypes = np.unique(self.FoundTypes)
        self.logger.info("Found DICOM Types: %s", self.FoundTypes)

        self.logger.debug("  Files in dict %s", fileDict)
        return fileDict

    def getUID(self, file_SOPUID):

        uids = {"CT": ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1', '1.2.840.10008.5.1.4.1.1.7', "CT Image Storage"],
            "REG": ['1.2.840.10008.5.1.4.1.1.66.3', '1.2.840.10008.5.1.4.1.1.66.1', "REG"],
            "RTSTRUCT":['1.2.840.10008.5.1.4.1.1.481.3','RT Structure Set Storage'],
            "RTDOSE": ['1.2.840.10008.5.1.4.1.1.481.2', 'RT Dose Storage']}
        foundUID = "not found"
        for key, value in uids.items():
            if file_SOPUID in value:
                foundUID = key
        return foundUID

    def load_scan(self, mode):
        """ loads and reads image dicom files

        Arguments:
            None
        Returns:
            dcm_files       list of pydicom objects of images
            slices          list of slice positions
        """
        if "CT" in self.FoundTypes:
            if "all" in self.dirDict:
                dcm_files = [dc.read_file(self.dirDict["all"] + os.sep + s) for s in self.fileDict["CT"]]
            else:
                dcm_files = [dc.read_file(self.dirDict["CT"] + os.sep + s) for s in self.fileDict["CT"]]
            img_files = [(round(float(dcm_files[iFile].ImagePositionPatient[2]), 3),
                          self.fileDict["CT"][iFile],
                          dcm_files[iFile]) for iFile in range(len(self.fileDict["CT"]))]
            img_files.sort()
            arr = np.array(img_files)
            self.fileDict["CT"] = arr[:,1]
            dcm_files = arr[:,2]

            slices = map(float, arr[:,0])
            if not isinstance(slices, list):
                slices = list(slices)

            self.origin = Voxel(round(float(dcm_files[0].ImagePositionPatient[0]), 2),
                                round(float(dcm_files[0].ImagePositionPatient[1]), 2),
                                round(float(dcm_files[0].ImagePositionPatient[2]), 2))
            self.res = Voxel(round(float(dcm_files[0].PixelSpacing[0]), 2),
                             round(float(dcm_files[0].PixelSpacing[1]), 2),
                             round(abs(slices[1]-slices[0]), 2))
            self.dim = Voxel(int(dcm_files[0].Columns), int(dcm_files[0].Rows), int(len(slices)))
            if mode == "full":
                self.images = self.get_pixels_hu(dcm_files)
            self.name = dcm_files[0].PatientName
            self.slices = slices
            del img_files, dcm_files
        else:
            self.debug.append("No IMG found")

    def load_rs(self, contourNames):
        """ loads and reads image rtstruct files

        Arguments:
            contourNames    Names of the contours to be read in the rtstruct
        Returns:
            None
        """

        if len(self.dirDict) == 1 and "all" in self.dirDict:
            dcm = dc.read_file(self.dirDict["all"] + os.sep + self.fileDict["RTSTRUCT"][0])
        else:
            dcm = dc.read_file(self.dirDict["RTSTRUCT"] + os.sep + self.fileDict["RTSTRUCT"][0])
        if "No IMG found" not in self.debug:
            self.rs = dcmreader.StructureSet(dcm, contourNames)
        del dcm

    def load_reg(self, mode = dict({"deformable": 1, "rigid": 0})):
        """ loads and reads deformable image registration files

        Arguments:
            mode            type of registration to be read in
        Returns:
            None
        """

        for i in range(len(self.fileDict["REG"])):
            dcm = dc.read_file(self.dirDict["REG"] + os.sep + self.fileDict["REG"][i])
            if mode["deformable"] and dcm.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.3":
                self.reg.append(dcmreader.Deformable(dcm))
            elif mode["rigid"] and dcm.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.1":
                self.reg.append(dcmreader.Rigid(dcm))
        del dcm

    def load_dose(self):
        if len(self.dirDict) == 1 and "all" in self.dirDict:
            dcm = dc.read_file(self.dirDict["all"] + os.sep + self.fileDict["RTDOSE"][0])
        else:
            dcm = dc.read_file(self.dirDict["RTDOSE"] + os.sep + self.fileDict["RTDOSE"][0])
        self.dose = dcmreader.Dose(dcm)
        del dcm

    def get_pixels_hu(self, scans):
        """ returns Houndsfield units from the scans

        Arguments:
            scans           list of pydicom objects of the image scans
        Returns:
            images          numpy array of the HU values of scans
        """
        image = np.stack([s.pixel_array for s in scans])
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
        images = np.array(image, dtype=np.int16)
        return images

    def setLabel(self, label):
        """ set Status label for patient

        Arguments:
            label           label = 1 status = 1, status = 0
        Returns:
            None
        """
        self.label = label

    def info(self):
        """ return basic informtion about patient img (origin and resolution)

        Arguments:
            None
        Returns:
            None
        """
        try:
            self.logger.info("Image origin:\t\t\t\t %s",self.origin.getRounds())
            self.logger.info("Image res:\t\t\t\t %s",self.res.getRounds())
        except: pass

    def load_vtkInfos(self):
        """ load vtk Information to set new bounds

        Arguments:
            None
        Returns:
            None
        """
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(self.dirDict["CT"])
        reader.Update()
        # flip the image in Y and Z directions
        flip = vtk.vtkImageReslice()
        flip.SetInputConnection(reader.GetOutputPort())
        flip.SetResliceAxesDirectionCosines(1,0,0, 0,-1,0, 0,0,-1);
        flip.Update();

        self.bounds = flip.GetExecutive().GetWholeExtent(flip.GetOutputInformation(0))
