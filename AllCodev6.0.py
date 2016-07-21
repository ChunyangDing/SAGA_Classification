'''
Classification Code v6.0
Written by Chunyang Ding for Professor Marla Geha
Yale University

Last updated: 7/21/2016

This code is designed to implement an automatic regression and classification algorithm to spectral data contained
in FITS tables of galaxies, given a spectral redshift and a qualification score. It computes the redshifted expected
line locations of common emission and absorption lines, and searches for a small region of data points to fit over.

It takes into account data variance and performs a chi-squared nonlinear least squares best fit model for a Gaussian
curve with a constant term added. This model was selected for through MCMC of several different polynomial and Gaussian
models. In addition a 95% confidence interval on the model parameters is calculated through regular chi-squared techniques.
The model parameters are then used to classify the spectra, taking into account the presence of different emission and
absorption lines.

The final result of this code outputs a classification of the following:
1 - Bad data
5 - Emission Line spectra
6 - Absorption Line spectra
7 - Quasar Object
-5 - Needs review; cannot classify

This can be run for all of the data in a single FITS table at once, as well as for individual objects with supplementary
fitting information. Runtime for a large file (~300 objects) tends to take about 30 seconds. The data in those FITS tables
should be cleaned prior to being run through the code through a user-defined function.
'''


import itertools
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import optimize
import random
import time
from scipy.stats.distributions import  t
from math import isnan
import os 
import itertools


#Define basic gaussian functions to be used throughout the code
def gauss(x, sigma, A, mu): #Basic gaussian model in x, theta form. theta[0] = sigma, theta[1] = A, theta[2] = mu
    if sigma < 0: #Tries to prevent negative sigma from being found with the regression technique.
        return 0*x
    coef = 1 / (sigma * np.sqrt(2 * np.pi))
    return A * coef * np.exp(-((x - mu) ** 2)/(2 * sigma ** 2))

def gaussConst(x, sigma, A, mu, B): #Basic gaussian model. theta[0] = sigma, theta[1] = A, theta[2] = mu, theta[3] = B
    if sigma < 0: 
        return 0*x
    coef = 1 / (sigma * np.sqrt(2 * np.pi))
    return A * coef * np.exp(-((x - mu) ** 2)/(2 * sigma ** 2)) + B

#Basic folder destinations for saving files and retrieving data. 
os.chdir('..')
PARENT = os.getcwd()

DATA_FOLDER = PARENT + "\data\\"
CLASS_FOLDER = PARENT + "\classification\\"
DIAG_FOLDER = PARENT + "\diagonostics\\"

#A small helper function to convert some given wavelength into a location index in the wavelength array.
def convertAngstromToBin(wavelength, xData):
    return np.where(xData == min(xData, key=lambda x:abs(x-wavelength)))[0][0]

#Determine if file is good or not
def isQuality(fileNum, zlog):
    q = int(zlog[fileNum].split()[6]) #Q = 3, 4 is acceptable. 1, 2 is not.
    return q >= 3

#Find a list of possible line locations and then return the redshifted lines
#The redshifted lines can be customized by simply adding in additional lines (in angstroms).
def getVisibleLines(z):
    #Major Absorption lines
    HBETA = 4862.69
    CaIIK = 3934.78
    Mg = 5174.13
    Na = 5891.58
    
    #Major emission lines
    OII = 3726.03
    OIII = 4958.92
    HALPHA = 6562.80
    
    #Concatenated list of lines
    allLines = [HALPHA, HBETA, OII, OIII, CaIIK, Mg, Na]

    #These are used to differentiate how many of each type line is found.
    numTest = 0
    numEmi = 0
    numAbs = 0
    index = 0
    
    goodLines = []
    #Checks if each line is within the allowed range. If not, the line is removed
    for line in allLines:
        if not ((line + z * line) < 4000 or (line + z * line) > 8500):
            goodLines.append(line)

            #These values need to be adjusted if a different set of lines is used
            if index > 3:
                numAbs += 1
            elif index > 1:
                numEmi += 1
            else:
                numTest += 1
        index += 1
    
    if not goodLines:
        return [[-1]] #Condition to check if list is empty, ie, z makes so that no lines can be detected.
    else:
        return [goodLines, numTest, numEmi, numAbs]


#Computes the redshifted result from the lines. Requires a redshift from a certain zlong file associated
#with the original data file.
def getShiftLines(fileNum, zlog):
    z = float(zlog[fileNum].split()[4]) #z from zlog
    usedLines = getVisibleLines(z) #Gets all of the visible lines
    #print usedLines
    if usedLines[0][0] == -1: #Checks if there are no valid lines
        return [-1] #Returns a invalid flag to be used in rest of the code.
    else:
        allShiftedLines = []
        for line in usedLines[0]:
            allShiftedLines.append(line + z * line) #Redshift formula, shifts each valid line
        return [allShiftedLines, usedLines[1], usedLines[2], usedLines[3]] #Returns all of the lines.

#Given a possible line region, we want to find the peak in the actual data.
#We search through a small region in the code (that makes sense given what we know about emission
#and absorption line widths) and find large deviations to the baseline.
def getPeak(fileNum, data, zlog, line):
    
    #Gets all of the data
    wavelength =data[0]
    flux = data[1]
    sig = data[2]

    #Finds the code locations 
    left = min(wavelength, key=lambda x:abs(x-(line - 12))) #scans \pm 12 Angstrom region for peak
    right = min(wavelength, key=lambda x:abs(x-(line + 12)))

    #print "Line: " + str(line)

    #Checks edge conditions and adjust accordingly
    if line < wavelength[0 + 100]:
        left = wavelength[0]
    if line > wavelength[len(wavelength) - 100]:
        right = wavelength[len(wavelength) - 1] 

    #Gets the code location of the bounds
    lb = np.where(wavelength == left)[0][0]
    rb = np.where(wavelength == right)[0][0]
    
    #Gets the code location of the peak - Steps 5, 6
    averageFlux = np.mean(flux[lb-30:rb+30]) #Creates a baseline
    var = np.abs(flux - averageFlux) #Looks for differences to the baseline, in abs. 
    print var
    peakLoc = np.where(var == max(var[lb:rb]))[0][0] #Finds the biggest dip or peak
    
    maxDeltaFlux = np.abs(var[peakLoc] - var[peakLoc + 1])

    #Tries to find a region for the peak by examining when the variation between the flux and the baseline
    #begins to fall off. This section seems to still have unknown bugs. Unclear the best way to reach this
    #value, as limited number of data points.
    halfWindow = 0
        
    while var[peakLoc + halfWindow] > var[peakLoc] * 0.1:
        halfWindow += 1
    
    if (var[peakLoc] < (3 * sig[peakLoc])): #Checks if the peak is greater than 3 sigma above baseline
        return [-1, -1] #Returns peakLoc
    
    if halfWindow < 5: #If the half-width is found to be too small, the default minimal half-width needed for fitting is used.
        return [peakLoc, 5]
    return [peakLoc, halfWindow]

#Selecting the data around the line. 
def getPeakInfo(data, lineInfo):
    #Creating easily accessible data to work with, for each line
    peak = lineInfo[0]
    scan = lineInfo[1]
    data = [data[0][peak - scan:peak + scan], data[1][peak - scan:peak + scan], data[2][peak - scan:peak + scan]]
    return data

#et the gaussian fit of the line, with covariance matrix
def get_gauss_theta(data):
    xValues = np.array(data[0])
    yValues = np.array(data[1])
    sigValues = np.array(data[2])
    
    peak = len(xValues)/2
    
    #Two separate fitting functions because of initial guesses. Otherwise, optimize.curve_fit will not converge properly
    #Try/except blocks put in in case if selected data is over 3 sigma, but still rather bad.
    
    if yValues[peak] > yValues[0]: 
        #print "Peak"
        try: #Initial fit on "cleaned" data, to get good guesses for a few parameters
            dummyGauss = optimize.curve_fit(gauss, xValues - xValues[peak], yValues - yValues[0], p0 = [4, 5, 0], maxfev = 5000)[0]
        except RuntimeError:
            print "Peak, dummyGauss exception"
            return [[-1, -1, -1, -1], [-1, -1, -1, -1]]
        
        dummyGauss[2] += xValues[peak] #Parameter adjutment to account for "real" data.
        thetaG1 = np.append(dummyGauss, yValues[0])
        
        try: #Fit on "real" data with good best guesses.
            G1, pcov = (optimize.curve_fit(gaussConst, xValues, yValues, p0 = thetaG1, sigma = sigValues, maxfev = 5000))
        except RuntimeError:
            print "Peak, GaussConst exception"
            return [[-1, -1, -1, -1], [-1, -1, -1, -1]]
                
    else:
        #print "Dip"
        try: #Same as above, but for dips rather than peaks. Simply has a flipped negative sign.
            dummyGauss = optimize.curve_fit(gauss, xValues - xValues[peak], yValues - yValues[0], p0 = [4, -5, 0], maxfev = 5000)[0]
        except RuntimeError:
            print "Dip, Gauss Exception"
            return [[-1, -1, -1, -1], [-1, -1, -1, -1]]
        
        dummyGauss[2] += xValues[peak]
        thetaG1 = np.append(dummyGauss, yValues[0])
        
        try:
            G1, pcov = (optimize.curve_fit(gaussConst, xValues, yValues, p0 = thetaG1, sigma = sigValues, maxfev = 5000))
        except RuntimeError:
            print "Dip, GaussConst Exception"
            return [[-1, -1, -1, -1], [-1, -1, -1, -1]]
                
    return G1, pcov

#Compute the Chi squared, given the data and the fit parameters
def chiSquaredRed(data, fitParams, dof):
    xValues = np.array(data[0])
    yValues = np.array(data[1])
    sigValues = np.array(data[2])
    
    fittedVals = gaussConst(xValues, fitParams[0], fitParams[1], fitParams[2], fitParams[3])
    
    chiSq = 0
    for i in range(len(yValues)):
        #Compute basic chi-squared, taking into account the individual uncertainties
        chiSq += ((yValues[i] - fittedVals[i]) ** 2) / (sigValues[i] ** 2)
        
    return chiSq/float(dof) #return the reduced chi squared, which is the chi squared divided by the degrees of freedom

#Step 10: Compute the confidence interval given the covariance matrix
def confInterval(params, covar, alpha, dof):
    CIList = []
    for index in range(len(params)):
        #Apply confidence interval formula based on covariance matrix
        tval = t.ppf(1.0-alpha/2., dof)
        covariance = np.diag(covar)[index]
        if np.isinf(covariance): #Checks if there are bad values in the Covariance matrix, and blocks them out.
            CIList.append([-1, 1])
        else:
            region = tval * np.sqrt(np.diag(covar)[
            #Append information to list
            CIList.append([params[index] - region, params[index] + region])
    return CIList

#Provides the fitting for each of the lines in a single object. Requires cleaned data - see below
#for more detail.
def fitPeakDips(fileNum, tr_data0, tr_data1, tr_data2, tr_zlog):   
    alpha = 0.1 #Trying out 90% confidence level instead of 95%...
    
    #Create a list to store all the fits and other information
    collectedFits = []
    
    z = float(tr_zlog[fileNum].split()[4])
    
    #Step 1: Check if file is good or not
    if not isQuality(fileNum, tr_zlog):
        #print "Bad file quality on fileNum " + str(fileNum)
        return [[[-1], -1, -1, [0, 0, 0]]]
    
    else:
        
        data = [tr_data0, tr_data1, tr_data2]
        
        #Step 2 - Get shifted lines
        shiftLineInfo = getShiftLines(fileNum, tr_zlog) #Should have something here to customize lines being given...
        shiftedLines = shiftLineInfo[0]
        #Step 3 - Loop through each of the lines
        if len(shiftedLines) == 1 and shiftedLines[0] == -1:
            return [[[0], 0, 0, [0, 0, 0]]]
        else:
            for line in shiftedLines:
                if not line == -1:
                    #Step 4, 5, 6 - get the code location for the peak near the identified line
                    lineInfo = getPeak(fileNum, data, tr_zlog, line)
                    #Step 6.5: Check if the peak is missing or not. Append bad if needed. 
                    if lineInfo[0] == -1:
                        #print "No peak found at line " + str(line)
                        collectedFits.append([[-1, -1, -1, -1], -1, [[-1, 1], [-1, 1], [-1, 1], [-1, 1]], z])
                    else:
                        #Step 7: Get the range of data as needed
                        lineData = getPeakInfo(data, lineInfo)
                        dof = len(lineData[0]) - 4
                        #Step 8: Get the best theta fit
                        theta, covar = get_gauss_theta(lineData)
                        if theta[0] == -1:
                            collectedFits.append([[-1, -1, -1, -1], -1, [[-1, 1], [-1, 1], [-1, 1], [-1, 1]], z])
                        else:
                            #Step 9: Compute the reduced chi-squared value
                            chiSqRed = chiSquaredRed(lineData, theta, dof)
                            #Step 10: Compute the 95% confidence interval
                            CI = confInterval(theta, covar, alpha, dof)
                            #Step 11: Append to big list and repeat.
                            collectedFits.append([theta, chiSqRed, CI, z])
        collectedFits.append([[-1, -1, -1, -1], -1, [[-1, 1], [-1, 1], [-1, 1], [-1, 1]], shiftLineInfo[1:], z])
    return collectedFits

#The next four functions are created to specifically handle each of the different data formats found in
#the telescopes used in this project. The data should be put into a format where the wavelength data is
#in the first index, the flux data in the second, and the sigma data in the third, while the .txt data from
#zlog is in the fourth.

#First gets all of the basic data for the entire file into one easy place from MMT data.
def MMTDataImport(fits_file, zlog_file):
    train = fits.open(DATA_FOLDER + fits_file)
    
    tr_data0 = train[0].data #Wavelength - Angstrom
    tr_data1 = train[1].data #Flux
    tr_data2 = train[2].data #Inverse Variance!
        
    f1 = open(DATA_FOLDER + zlog_file, 'r')
    tr_zlog = f1.readlines()
    
    return tr_data0, tr_data1, tr_data2, tr_zlog

#Cleans up the data for one specific object. Can be used to convert inverse variance
#into standard deviation to be used. 
def MMTDataClean(all_data, fileNum):
    wavelength = all_data[0][fileNum]
    flux = all_data[1][fileNum]
    inverseVar = all_data[2][fileNum]
    tr_zlog = all_data[3]
    
    for i in range(len(inverseVar)):
        if inverseVar[i] == 0:
            inverseVar[i] = 1
        
    return [wavelength, flux, np.sqrt(1/inverseVar), tr_zlog]

#Similar to above. Wavelength data is not explicit in AAT data, so is calculated from the
#FITS header. 
def AATDataImport(fits_file, zlog_file):
    train = fits.open(DATA_FOLDER + fits_file)
    
    tr_data1 = train[0].data #Flux

    coord_val = train[0].header['CRVAL1']
    increment = train[0].header['CDELT1']
    coord_pix = train[0].header['CRPIX1']
    axis1_length = train[0].header['NAXIS1']
    start_val = coord_val - increment * coord_pix
    end_val = start_val + increment * axis1_length 
    wavelength = np.arange(start_val, end_val - increment, increment)
    
    tr_data0 = list(itertools.repeat(wavelength, len(tr_data1))) #Wavelength - Angstrom
    tr_data2 = train[1].data #Variance!
    
    f1 = open(DATA_FOLDER + zlog_file, 'r')
    tr_zlog = f1.readlines()
    
    return tr_data0, tr_data1, tr_data2, tr_zlog

#cleans up the AAT data by removing the "nan"s from the file.
def AATDataClean(all_data, fileNum):
    wavelength = all_data[0][fileNum]
    flux = all_data[1][fileNum]
    variance = all_data[2][fileNum]
    tr_zlog = all_data[3]
    
    plt.plot(wavelength, flux)
    plt.show()
    
    tempData = [[0],[0],[0]]
    for i in range(len(wavelength)):
        if not isnan(flux[i]):
            tempData[0].append(wavelength[i])
            tempData[1].append(flux[i])
            tempData[2].append(np.sqrt(variance[i]))

    tempData[0].pop(0)
    tempData[1].pop(0)
    tempData[2].pop(0)

    #Checks to ensure that there is data remaining in the file. Otherwise, return error flags.
    if len(tempData[1]) < 2:
        return [[0], 0, 0, 0]
      
    tempData.append(tr_zlog)
    
    return tempData


#Step 12: Go through every single item in a file and compute the fit parameters
def allTogetherNow(fits_file, zlog_file):

    #Get the data from the telescope FITS files. Specialized programs defined above.
    #all_data = MMTDataImport(fits_file, zlog_file)
    all_data = AATDataImport(fits_file, zlog_file)
    
    allObsFits = []
    for i in range(300):
        #print "Filenum: " + str(i)
        
        #file_data = MMTDataClean(all_data, i)
        file_data = AATDataClean(all_data, i)

        #Quickly checks if the data is good or not. If not, the append error flags, else, append good results.
        if file_data[0][0] == 0:
            allObsFits.append([[[0],0,-1,[0,0,0]]])
        else:
            #Call upon the primary fitting method to get a list of fits.
            allObsFits.append(fitPeakDips(i, file_data[0], file_data[1], file_data[2], file_data[3]))
        
    return allObsFits

#Step 13: Do an easy classification scheme, just by summing up the peaks. Not statistically valid, but mostly works
#Sets an arbitrary limit for what is considered a emission or absorption spectra. Requires that the original peak
#fits satisfy some condition, where the 95% confidence interval on the Gaussian amplitude (2 sigmal parameter estimation)
#on both sides is either positive or negative; no cross.
def easyClassification(obj_fits):
    sumA = 0
    if obj_fits[0][0][0] == 0:
        return 0
    if obj_fits[0][0][0] == -1:
        return -1
    for line in obj_fits:
        if not(line[2][1][0] < 0 and line [2][1][1] > 0):
            sumA += line[0][1]
    return sumA

#A slightly more complex method, by taking the area of each peak.
def medClassification(obj_fits):
    intA = 0
    if obj_fits[0][0][0] == 0:
        return 0
    if obj_fits[0][0][0] == -1:
        return -1
    for line in obj_fits:
        #print line[2][1] #VERY USEFUL FOR DEBUG PURPOSES!
        if not (line[2][1][0] < 0 and line [2][1][1] > 0):
            #print "good line"
            intA += np.sqrt(2) * line[0][1] * np.abs(line[0][0]) * np.sqrt(np.pi) #Formula for Gaussian area
    return intA

#This is a more procedural classification scheme where instead of arbitrarily depending on total summed line heights
#or areas, it considers which lines were found and which were not. From there, the algorithm follows some implemented
#logic to determine what is the best classification.
#This method of classification tends to have lots of "unable to classify", especially if some of the lines are particularly
#weak. This is because there is a "mixed" flag implemented. Therefore, if an emission line is found where an absorption line
#is expected, the algorithm interprets this as an error and prevents the code from returning a classification.
def hardClassification(obj_fits, trainNum, emiNum, absNum):
    if len(obj_fits) == 1:
        if obj_fits[0][0][0] == 0:
            return -5
        if obj_fits[0][0][0] == -1:
            return 1
    #Separate the fit data into the train set (hydrogen lines), emiSet, and absSet.
    trainSet = obj_fits[:trainNum]
    emiSet = obj_fits[trainNum:trainNum + emiNum]
    absSet = obj_fits[trainNum+emiNum:trainNum + emiNum + absNum]

    #Creates several different flags to keep track of which lines were found and which were not.
    emiTrue = True
    mixFlag = False
    
    trainEmi = False
    trainAbs = False
    goodTrain = False
    
    emiMix = False
    absMix = False
    
    emiEmi = False
    absAbs = False

    #Each comparison first validates that the line is itself good, before checking if the amplitude is
    #positive or not. 
    
    for line in trainSet:
        if (not (line[2][1][0] < 0 and line[2][1][1] > 0) and line[0][0] > 0.75):
            goodTrain = True
            if line[0][1] > 0:
                trainEmi = True
            if line[0][1] < 0:
                trainAbs = True
                
    for line in emiSet:
        if (not (line[2][1][0] < 0 and line[2][1][1] > 0) and line[0][0] > 0.75):
            if line[0][1] < 0:
                emiMix = True
            if line[0][1] != -1:
                emiEmi = True
                
    for line in absSet:
        if (not (line[2][1][0] < 0 and line[2][1][1] > 0) and line[0][0] > 0.75):
            if line[0][1] > 0:
                absMix = True
            if line[0][1] != -1:
                absAbs = True

    #The implemented logic for classification...
    if trainAbs:
        if absMix:
            return -5
        elif absAbs:
            return 6
        return 6
    
    elif trainEmi:
        if emiMix:
            return -5
        elif emiEmi:
            return 5
        return 5

    else:
        if absAbs and not absMix:
            return 6
        elif emiEmi and not emiMix:
            return 5
        else:
            return -5

#Goes through the entire file of object fits and classifies them one by one. Returns a list of classifications in same
#order as the list of peak fits.
def allClassified2(allObsFits):
    classifications = []
    #I apologize to all my past and future CS teachers for this montrosity of an array. Please forgive me.
    for i in range(len(allObsFits)):
        classifications.append(hardClassification(allObsFits[i], allObsFits[i][-1][3][0], allObsFits[i][-1][3][1], allObsFits[i][-1][3][2]))
    return classifications

#An earlier method of classification, now deprecated. This method /seems/ to work, but there is no good
#scientific justification for choosing certain magic numbers. In addition, this method would be highly
#dependent on the number of lines used, as well as the arbitrary flux standards, so it is NOT recommended.
def allClassified(allObsFits):
    classifications = []
    for i in range(len(allObsFits)):
        #print "fileNum:      " + str(i)
        sumA = easyClassification(allObsFits[i])
        intA = medClassification(allObsFits[i])
        #print "summed A:     " + str(sumA)
        #print "integrated A: " + str(intA)
        #print ""
        if ((sumA < 0 and intA > 0) or (sumA > 0 and intA < 0)):
            classifications.append(-5)
        elif(intA < 20 and intA > -1):
            classifications.append(-5)
        elif intA == -1:
            classifications.append(1)
        elif intA > 0:
            classifications.append(5)
        elif intA < 0:
            classifications.append(6)
        else:
            classifications.append(-5)
    return classifications

'''
This portion of the code actually runs through everything that is needed.
The syntax for running this code is to put in the name of the fits file and then
of the zlog file, and to save that fitted peak data. Afterwards, run "allClassified2" on
that output data file to get classifications.
'''
#allObsFits = allTogetherNow('spHect-2015.0918_1.fits', 'spHect-2015.0918_1.zlog')
allObsFits = allTogetherNow('Aeneid_1.fits', 'Aeneid_1.zlog')
quickClassification = allClassified(allObsFits)
print quickClassification

hardClass = allClassified2(allObsFits)
print hardClass

def getSigmaPlot(allObsFits)
    allSigma = []
    for temp in allObsFits:
        for line in temp:
            if not line[2] == -1:
                if not(line[2][1][0] < 0 and line[2][1][1] > 0):
                    allSigma.append(line[0][0])
                
    samp = plt.hist(allSigma, bins = 40, range =[0, 10], normed = True)
    plt.xlabel("Sigma of gaussian fit (Angstroms)")
    plt.ylabel("Relative Frequency")
    plt.title("Distribution of line widths in file")
    print samp
    plt.show()

'''
Here is a simple diagonostic function that was implemented for both code checking as well
as returning direct information on the fitting for each of the peaks and the dips. It also
generates plots to be saved for each predicted line location and overplots of the best fit
regression. Finally, it also saves the intermediate step information in a .txt file.
'''
diagnosticsFits('2015.0918_1', 232)

#This function is for printing out all of the diagonostics for a single object with a single file.
#It ought to print/save everything line by line
def diagnosticsFits(fitsNum, objNum):
    
    SHOW_PLOTS = False
    SAVE_PLOTS = True
    
    alpha = 0.05 #2 sigma, 95%
    
    zlogFileName = 'spHect-' + str(fitsNum) + ".zlog"
    fitsFileName = 'spHect-' + str(fitsNum) + ".fits"
    idName = str(fitsNum) + '_' + str(objNum + 1).zfill(3)
    diagnosticFile = 'df' + str(idName) + '.dat'
    
    diagFile = open(diagnosticFile, 'w+')
    diagFile.write("Diagnostics for Object " + str(idName) + "\n")
    
    try:
        zlogFile = open(zlogFileName, 'r')
    except IOError:
        print "No file found with fitsNum %s" % fitsNum
        
    zlog = zlogFile.readlines()
    fitsFile = fits.open(fitsFileName)
    
    wavelength = fitsFile[0].data[objNum]
    flux = fitsFile[1].data[objNum]
    invVar = fitsFile[2].data[objNum]
    
    for i in range(len(invVar)):
        if invVar[i] == 0:
            invVar[i] = 1.
        
    sigma = np.sqrt(1/invVar)
          
    #Gets the shifted wavelengths that are still visible
    shiftLineInfo = getShiftLines(objNum, zlog)
    shiftedLines = shiftLineInfo[0]
    diagFile.write("Shifted Line Wavelengths: \n")
    diagFile.write(str(shiftedLines) + "\n")
    
    collectedFits = []
    
    #Step 3 - Loop through each of the lines
    if len(shiftedLines) == 1 and shiftedLines[0] == -1:
        print "No lines found in this file"
        diagFile.write("No lines found in this file")  
    else:
        for line in shiftedLines:
            if not line == -1:
                
                #Plots the region that the algorithm searches for the biggest dip in.
                lb = convertAngstromToBin(line - 50, wavelength)
                rb = convertAngstromToBin(line + 50, wavelength)
                plt.figure(figsize=(20,10))
                plt.errorbar(wavelength[lb:rb], flux[lb:rb], yerr = sigma[lb:rb], fmt = 'o')
                plt.title( "50 Angstrom Region around Predicted Line %s"%line )
                plt.xlabel( 'Wavelength')
                plt.ylabel("Flux")
                if SHOW_PLOTS:
                    plt.show()
                if SAVE_PLOTS:
                    plt.savefig("SearchRegion_" + str(idName) + "_Line_" + str(round(line)) + ".png")
                plt.clf()
                
                diagFile.write('\n')
                diagFile.write('Information for peak located at ' + str(round(line)) + ' angstroms as follows: \n')
                #Step 4, 5, 6 - get the code location for the peak near the identified line
                lineInfo = getPeak(objNum, [wavelength, flux, sigma], zlog, line)
                #Step 6.5: Check if the peak is missing or not. Append bad if needed. 
                if lineInfo[0] == -1:
                    diagFile.write("No peak found at line " + str(line))
                    collectedFits.append([[-1, -1, -1, -1], -1, [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]])
                else:
                    #Step 7: Get the range of data as needed
                    diagFile.write("Found following line info: \n" + str(lineInfo) + "\n")
                    lineData = getPeakInfo([wavelength, flux, sigma], lineInfo)
                    dof = len(lineData[0]) - 4
                    
                    #Step 8: Get the best theta fit
                    theta, covar = get_gauss_theta(lineData)
                    
                    if theta[0] == -1:
                        collectedFits.append([[-1, -1, -1, -1], -1, [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]])
                        diagFile.write("Unable to find best fit to the Gaussian")
                    else:
                        diagFile.write("Found following theta parameters: \n " + str(theta) + "\n")
                        diagFile.write("Found following covariance matrix: \n " + str(covar) + "\n")
                        
                        #Step 9: Compute the reduced chi-squared value
                        chiSqRed = chiSquaredRed(lineData, theta, dof)
                        #Step 10: Compute the 95% confidence interval
                        CI = confInterval(theta, covar, alpha, dof)
                        #Step 11: Append to big list and repeat.
                        
                        diagFile.write("Found chiSquared-Reduced as: " + str(chiSqRed) + "\n")
                        diagFile.write("Found confidence intervals as follows: " + str(CI) + "\n")
                        collectedFits.append([theta, chiSqRed, CI])

                        xSpace = np.linspace(lineData[0][0] - 20, lineData[0][-1] + 20)
                        ySpace = gaussConst(xSpace, theta[0], theta[1], theta[2], theta[3])
                        plt.figure(figsize=(20,10))
                        realLB = convertAngstromToBin(lineData[0][0] - 20, wavelength)
                        realRB = convertAngstromToBin(lineData[0][-1] + 20, wavelength)
                        plt.errorbar(wavelength[realLB:realRB], flux[realLB:realRB], yerr = sigma[realLB:realRB], fmt = 'o', label = 'Region of peak fitting data')
                        #plt.errorbar(lineData[0], lineData[1], yerr = lineData[2], fmt = 'o', label = 'Region of peak fitting data')
                        plt.plot(xSpace, ySpace, '-', label = "Gaussian Fit")
                        plt.title("GaussianFit_" + str(idName) + "_line_" + str(round(line)))
                        plt.xlabel("Wavelength")
                        plt.ylabel("Flux")
                        plt.legend()
                        if SAVE_PLOTS:
                            plt.savefig("GaussianFit_" + str(idName) + "_line_" + str(round(line)) +  ".png")
                        if SHOW_PLOTS:
                            plt.show()
                        plt.clf()
                        
        diagFile.write("\n")
        diagFile.write("Summed Area:        " + str(easyClassification(collectedFits)) + "\n")
        diagFile.write("Integrated Area:    " + str(medClassification(collectedFits)) + "\n")
        diagFile.write("Med Classification: " + str(allClassified([collectedFits])) + "\n")
        #diagFile.write("Hard Classification:" + str(allClassified2(collectedFits)) + "\n")
                        
    diagFile.close()
                        
    return [0]
