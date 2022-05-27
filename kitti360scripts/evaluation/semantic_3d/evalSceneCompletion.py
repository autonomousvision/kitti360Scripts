#!/usr/bin/python
#
# The evaluation script for pixel-level semantic labeling.
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
#
# Please check the description of the "getPrediction" method below
# and set the required environment variables as needed, such that
# this script can locate your results.
# If the default implementation of the method works, then it's most likely
# that our evaluation server will be able to process your results as well.
#
# Note that the script is a faster, if you enable cython support.
# WARNING: Cython only tested for Ubuntu 64bit OS.
# To enable cython, run
# CYTHONIZE_EVAL= python setup.py build_ext --inplace
#
# To run this script, make sure that your results are images,
# where pixels encode the class IDs as defined in labels.py.
# Note that the regular ID is used, not the train ID.
# Further note that many classes are ignored from evaluation.
# Thus, authors are not expected to predict these classes and all
# pixels with a ground truth label that is ignored are ignored in
# evaluation.

# python imports
from __future__ import print_function, absolute_import, division
import os, sys
import platform
import fnmatch
import time

# KITTI-360 imports
from kitti360scripts.helpers.csHelpers import *
from kitti360scripts.helpers.labels import id2label, trainId2label
from kitti360scripts.helpers.ply import read_ply
from sklearn.neighbors import KDTree

# C Support
# Enable the cython support for faster evaluation
# Only tested for Ubuntu 64bit OS
CSUPPORT = True
# Check if C-Support is available for better performance
if CSUPPORT:
    try:
        import addToConfusionMatrix
    except:
        CSUPPORT = False

# helper print function so we can switch from mail to standard printing
def printFunc( printStr, **kwargs ):
    print(printStr)


###################################
# PLEASE READ THESE INSTRUCTIONS!!!
###################################
# Provide the prediction file for the given ground truth file.
#
# The current implementation expects the results to be in a certain root folder.
# This folder is one of the following with decreasing priority:
#   - environment variable KITTI360_RESULTS
#   - environment variable KITTI360_DATASET/results
#   - ../../results/"

# Within the root folder, a matching prediction file is recursively searched.
# A file matches, if the filename follows the pattern
#   {seq:0>4}_{frame:0>10}*.npy
# for a ground truth filename
#   2013_05_28_drive_{seq:0>4}_sync/image_00/semantic/{frame:0>10}.npy
def getPrediction( args, groundTruthFile ):
    # determine the prediction path, if the method is first called
    if not args.predictionPath:
        rootPath = None
        if 'KITTI360_RESULTS' in os.environ:
            rootPath = os.environ['KITTI360_RESULTS']
        elif 'KITTI360_DATASET' in os.environ:
            rootPath = os.path.join( os.environ['KITTI360_DATASET'] , "results" )
        else:
            rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','results')

        if not os.path.isdir(rootPath):
            printError("Could not find a result root folder. Please read the instructions of this method.")

        args.predictionPath = rootPath

    # walk the prediction path, if not happened yet
    if not args.predictionWalk:
        walk = []
        for root, dirnames, filenames in os.walk(args.predictionPath):
            walk.append( (root,filenames) )
        args.predictionWalk = walk

    csFile = getFileInfo(groundTruthFile)
    filePattern = "*{:04d}_{:010d}.npy".format( csFile.sequenceNb , csFile.frameNb )

    predictionFile = None
    for root, filenames in args.predictionWalk:
        for filename in fnmatch.filter(filenames, filePattern):
            if not predictionFile:
                predictionFile = os.path.join(root, filename)
            else:
                printError("Found multiple predictions for ground truth {}".format(groundTruthFile))

    if not predictionFile:
        printError("Found no prediction for ground truth {}".format(groundTruthFile))

    return predictionFile


def getGroundTruth(groundTruthListFile, eval_every=1):
    if 'KITTI360_DATASET' in os.environ:
        rootPath = os.environ['KITTI360_DATASET']
    else:
        rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

    if not os.path.isdir(rootPath):
        printError("Could not find a result root folder. Please read the instructions of this method.")

    if not os.path.isfile(groundTruthListFile):
        printError("Could not open %s. Please read the instructions of this method." % groundTruthListFile)

    with open(groundTruthListFile, 'r') as f:
        lines = f.read().splitlines()

    groundTruthFiles = []
    for i,line in enumerate(lines):
        if i % eval_every == 0:
            accumulatedPcdFile = os.path.join(rootPath, line.split(' ')[0])
            groundTruthFile = os.path.join(rootPath, line.split(' ')[1])
            if os.path.isfile(os.path.join(rootPath, groundTruthFile)):
                groundTruthFiles.append([groundTruthFile, accumulatedPcdFile])
            else:
                if not os.path.isfile(accumulatedPcdFile):
                    printError('Could not find the accumulated point cloud %s' % accumulatedPcdFile)
                if not generateCroppedGroundTruth(rootPath, accumulatedPcdFile, groundTruthFile):
                    printError("Could not open %s. Please read the instructions of this method." % groundTruthFile)
    return groundTruthFiles

# Crop the accumulate point cloud as the ground truth for semantic completion at a given frame
# The gruond truth is within a corridor of 30m around the vehicle poses of a 100m trajectory (50m in each direction).
# If the forward direction of one pose deviates more than 45 degree compared to the heading angle of the given center, 
# it is eliminated from the neighboring poses.
def generateCroppedGroundTruth(rootPath, accumulatedPcdFile, outputFile, disThres=50.0, angleThres=45.0):
    print("Creating %s from %s" % (outputFile, accumulatedPcdFile))
   
    # load the full accumulated window
    groundTruthPcd = read_ply(accumulatedPcdFile)
    groundTruthNpWindow = np.vstack((groundTruthPcd['x'], 
                                    groundTruthPcd['y'],
                                    groundTruthPcd['z'])).T
    groundTruthFullLabel = groundTruthPcd['semantic']
    groundTruthLabelWindow = np.zeros_like(groundTruthFullLabel)
    groundTruthColorWindow = np.zeros((groundTruthFullLabel.shape[0], 3))
    groundTruthConfWindow = groundTruthPcd['confidence']
    # Convert to trainId for evaluation, as some classes should be merged 
    # during evaluation, e.g., building+garage -> building
    for i in np.unique(groundTruthFullLabel):
        groundTruthLabelWindow[groundTruthFullLabel==i] = id2label[i].trainId
        groundTruthColorWindow[groundTruthFullLabel==i] = id2label[i].color
    groundTruthWindowTree = KDTree(groundTruthNpWindow, leaf_size=args.leafSize)

    # load the poses to determine the cropping region
    poseFile = os.path.join(rootPath, 'data_poses', '2013_05_28_drive_%04d_sync' % csWindow.sequenceNb, 'poses.txt')
    poses = np.loadtxt(poseFile)
    frameNb = int(os.path.splitext(os.path.basename(outputFile))[0])
    frameIdx = np.where(poses[:,0]==frameNb)[0]
    pose = poses[frameIdx]
    pose = np.reshape(pose[0,1:], (3,4)) 
    center = pose[:,3]

    # find neighbor points within the same window with the following conditions
    # 1) distance to the current frame within 40meters
    posesNeighbor = poses[np.logical_and(poses[:,0]>=csWindow.firstFrameNb, poses[:,0]<=csWindow.lastFrameNb)]
    dis_to_center = np.linalg.norm(posesNeighbor[:,1:].reshape(-1,3,4)[:,:,3] - center, axis=1)
    posesNeighbor = posesNeighbor[dis_to_center<disThres]
    # 2) curvature smaller than a given threshold (ignore potential occluded points due to large orientation)
    centerPrev = poses[frameIdx-1,1:].reshape(3,4)[:,3]
    centerNext = poses[frameIdx+1,1:].reshape(3,4)[:,3]
    posesPrevIdx = np.where(posesNeighbor[:,0]<=frameNb)[0]
    posesNextIdx = np.where(posesNeighbor[:,0]>=frameNb)[0]
    posesPrevLoc = posesNeighbor[posesPrevIdx,1:].reshape(-1,3,4)[:,:2,3]
    posesNextLoc = posesNeighbor[posesNextIdx,1:].reshape(-1,3,4)[:,:2,3]
    anglePrev = getAngleBetweenVectors(centerPrev[:2]-center[:2], posesPrevLoc[:-1]-posesPrevLoc[1:])
    angleNext = getAngleBetweenVectors(centerNext[:2]-center[:2], posesNextLoc[1:]-posesNextLoc[:-1])
    posesValid = np.concatenate((posesNeighbor[posesPrevIdx[:-1][anglePrev<angleThres]],
                                     posesNeighbor[posesNeighbor[:,0]==frameNb,:],
                                     posesNeighbor[posesNextIdx[1:][angleNext<angleThres]]), axis=0)

    idx_all = []
    for i in range(posesValid.shape[0]):
        idx = groundTruthWindowTree.query_radius(posesValid[i, 1:].reshape(3,4)[:3,3].reshape(1,3), args.radius)
        idx_all.append(idx[0])
    idx_all = np.unique(np.concatenate(idx_all))
    groundTruthNp = groundTruthNpWindow[idx_all,:]
    groundTruthColor = groundTruthColorWindow[idx_all,:]
    groundTruthLabel = groundTruthLabelWindow[idx_all]
    groundTruthConf = groundTruthConfWindow[idx_all]
    os.makedirs(os.path.dirname(outputFile), exist_ok=True)
    np.savez(outputFile, posesValid=posesValid, groundTruthNp=groundTruthNp, groundTruthColor=groundTruthColor, 
             groundTruthLabel=groundTruthLabel, groundTruthConf=groundTruthConf)

    return True

def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)

def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:,0]+ M*cells[:,1] + M**2*cells[:,2]).shape[0]
        
######################
# Parameters
######################


# A dummy class to collect all bunch of data
class CArgs(object):
    pass
# And a global object of that class
args = CArgs()

# Where to look for KITTI-360 
if 'KITTI360_DATASET' in os.environ:
    args.kitti360Path = os.environ['KITTI360_DATASET']
else:
    args.kitti360Path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

if 'KITTI360_EXPORT_DIR' in os.environ:
    export_dir = os.environ['KITTI360_EXPORT_DIR']
    if not os.path.isdir(export_dir):
        raise ValueError("KITTI360_EXPORT_DIR {} is not a directory".format(export_dir))
    args.exportFile = "{}/resultPixelLevelSemanticLabeling.json".format(export_dir)
else:
    args.exportFile = os.path.join(args.kitti360Path, "evaluationResults", "resultPixelLevelSemanticLabeling.json")

# Parameters that should be modified by user
args.groundTruthListFile = os.path.join(args.kitti360Path, 'data_3d_semantics', 'train', '2013_05_28_drive_val_frames.txt')

# Remaining params
args.evalInstLevelScore = False
args.evalPixelAccuracy  = True
args.evalLabels         = []
args.printRow           = 5
args.normalized         = True
args.colorized          = hasattr(sys.stderr, "isatty") and sys.stderr.isatty() and platform.system()=='Linux'
args.bold               = colors.BOLD if args.colorized else ""
args.nocol              = colors.ENDC if args.colorized else ""
args.JSONOutput         = True
args.quiet              = False

# leaf size used in KDtree 
args.leafSize           = 10 
# size of the octree voxel which determines whether a voxel is observed or not
args.observedVoxelSize  = 0.5
# a predicted point is accurate if its distance to the nearest gt point is below the threshold
args.thresholdAcc       = 0.2
# a gt point is complete if its distance to the nearest predicted point is below the threshold
args.thresholdComplete  = 0.2
# we measure completeness and accuracy over discretized voxels such that these metrics are 
# insensitive to the density of the point clouds
args.voxelSize          = 0.02
args.radius             = 30.0

# store some parameters for finding predictions in the args variable
# the values are filled when the method getPrediction is first called
args.predictionPath = None
args.predictionWalk = None


#########################
# Methods
#########################

def getAngleBetweenVectors(vec1, vec2):
    vec1 = vec1.reshape(-1,2)
    vec2 = vec2.reshape(-1,2)
    dot_product = np.sum(vec1 * vec2, axis=1)
    cos_angle = dot_product / (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))
    angle = np.arccos(cos_angle)
    angle[np.isclose(cos_angle, 1.0)] = 0.0 # avoid nan at 1.0
    return angle

# Generate empty confusion matrix and create list of relevant labels
def generateMatrix(args):
    args.evalLabels = []
    for label in labels:
        if (label.trainId < 0): #or (label.trainId==255):
            continue
        if label.name in ['train', 'bus', 'rider', 'sky']:
            continue
        # we append all found labels, regardless of being ignored
        args.evalLabels.append(label.trainId)
    args.evalLabels = list(set(args.evalLabels))
    maxId = max(args.evalLabels) #+ 1 # for ignored_label:255
    #args.evalLabels.append(maxId)
    # We use float type as each pixel is weighted by the ground truth confidence
    return np.zeros(shape=(maxId+1, maxId+1),dtype=np.float64)

def generateEvalStats():
    evalStats = {}
    evalStats["nAccuratePredictionCells"] = 0
    evalStats["nValidPredictionCells"] = 0
    evalStats["nAccuratePredictionPoints"] = 0
    evalStats["nValidPredictionPoints"] = 0

    evalStats["nCompleteGroundTruthCells"] = 0
    evalStats["nValidGroundTruthCells"] = 0
    evalStats["nCompleteGroundTruthPoints"] = 0
    evalStats["nValidGroundTruthPoints"] = 0
    return evalStats

def generateInstanceStats(args):
    instanceStats = {}
    instanceStats["classes"   ] = {}
    instanceStats["categories"] = {}
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            instanceStats["classes"][label.name] = {}
            instanceStats["classes"][label.name]["tp"] = 0.0
            instanceStats["classes"][label.name]["tpWeighted"] = 0.0
            instanceStats["classes"][label.name]["fn"] = 0.0
            instanceStats["classes"][label.name]["fnWeighted"] = 0.0
    for category in category2labels:
        labelIds = []
        allInstances = True
        for label in category2labels[category]:
            if label.id < 0:
                continue
            if not label.hasInstances:
                allInstances = False
                break
            labelIds.append(label.id)
        if not allInstances:
            continue

        instanceStats["categories"][category] = {}
        instanceStats["categories"][category]["tp"] = 0.0
        instanceStats["categories"][category]["tpWeighted"] = 0.0
        instanceStats["categories"][category]["fn"] = 0.0
        instanceStats["categories"][category]["fnWeighted"] = 0.0
        instanceStats["categories"][category]["labelIds"] = labelIds

    return instanceStats


# Get absolute or normalized value from field in confusion matrix.
def getMatrixFieldValue(confMatrix, i, j, args):
    if args.normalized:
        rowSum = confMatrix[i].sum()
        if (rowSum == 0):
            return float('nan')
        return float(confMatrix[i][j]) / rowSum
    else:
        return confMatrix[i][j]

# Calculate and return IOU score for a particular label
def getIouScoreForLabel(label, confMatrix, args):
    if trainId2label[label].ignoreInEval:
        return float('nan')

    # the number of true positive pixels for this label
    # the entry on the diagonal of the confusion matrix
    tp = confMatrix[label,label]

    # the number of false negative pixels for this label
    # the row sum of the matching row in the confusion matrix
    # minus the diagonal entry
    fn = confMatrix[label,:].sum() - tp

    # the number of false positive pixels for this labels
    # Only pixels that are not on a pixel with ground truth label that is ignored
    # The column sum of the corresponding column in the confusion matrix
    # without the ignored rows and without the actual label of interest
    notIgnored = [l for l in args.evalLabels if not trainId2label[l].ignoreInEval and not l==label]
    fp = confMatrix[notIgnored,label].sum()

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate and return IOU score for a particular label
def getInstanceIouScoreForLabel(label, confMatrix, instStats, args):
    if trainId2label[label].ignoreInEval:
        return float('nan')

    labelName = trainId2label[label].name
    if not labelName in instStats["classes"]:
        return float('nan')

    tp = instStats["classes"][labelName]["tpWeighted"]
    fn = instStats["classes"][labelName]["fnWeighted"]
    # false postives computed as above
    notIgnored = [l for l in args.evalLabels if not trainId2label[l].ignoreInEval and not l==label]
    fp = confMatrix[notIgnored,label].sum()

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate prior for a particular class id.
def getPrior(label, confMatrix):
    return float(confMatrix[label,:].sum()) / confMatrix.sum()

# Get average of scores.
# Only computes the average over valid entries.
def getScoreAverage(scoreList, args):
    validScores = 0
    scoreSum    = 0.0
    for score in scoreList:
        if not math.isnan(scoreList[score]):
            validScores += 1
            scoreSum += scoreList[score]
    if validScores == 0:
        return float('nan')
    return scoreSum / validScores

# Calculate and return IOU score for a particular category
def getIouScoreForCategory(category, confMatrix, args):
    # All labels in this category
    labels = category2labels[category]
    # The IDs of all valid labels in this category
    labelIds = [label.trainId for label in labels if not label.ignoreInEval and label.trainId in args.evalLabels]
    # If there are no valid labels, then return NaN
    if not labelIds:
        return float('nan')

    # the number of true positive pixels for this category
    # this is the sum of all entries in the confusion matrix
    # where row and column belong to a label ID of this category
    tp = confMatrix[labelIds,:][:,labelIds].sum()

    # the number of false negative pixels for this category
    # that is the sum of all rows of labels within this category
    # minus the number of true positive pixels
    fn = confMatrix[labelIds,:].sum() - tp

    # the number of false positive pixels for this category
    # we count the column sum of all labels within this category
    # while skipping the rows of ignored labels and of labels within this category
    notIgnoredAndNotInCategory = [l for l in args.evalLabels if not trainId2label[l].ignoreInEval and trainId2label[l].category != category]
    fp = confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum()

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate and return IOU score for a particular category
def getInstanceIouScoreForCategory(category, confMatrix, instStats, args):
    if not category in instStats["categories"]:
        return float('nan')
    labelIds = instStats["categories"][category]["labelIds"]

    tp = instStats["categories"][category]["tpWeighted"]
    fn = instStats["categories"][category]["fnWeighted"]

    # the number of false positive pixels for this category
    # same as above
    notIgnoredAndNotInCategory = [l for l in args.evalLabels if not trainId2label[l].ignoreInEval and trainId2label[l].category != category]
    fp = confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum()

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom


# create a dictionary containing all relevant results
def createResultDict( confMatrix, classScores, classInstScores, categoryScores, categoryInstScores, perImageStats, args ):
    # write JSON result file
    wholeData = {}
    wholeData["confMatrix"] = confMatrix.tolist()
    wholeData["priors"] = {}
    wholeData["labels"] = {}
    for label in args.evalLabels:
        wholeData["priors"][trainId2label[label].name] = getPrior(label, confMatrix)
        wholeData["labels"][trainId2label[label].name] = label
    wholeData["classScores"] = classScores
    wholeData["classInstScores"] = classInstScores
    wholeData["categoryScores"] = categoryScores
    wholeData["categoryInstScores"] = categoryInstScores
    wholeData["averageScoreClasses"] = getScoreAverage(classScores, args)
    wholeData["averageScoreInstClasses"] = getScoreAverage(classInstScores, args)
    wholeData["averageScoreCategories"] = getScoreAverage(categoryScores, args)
    wholeData["averageScoreInstCategories"] = getScoreAverage(categoryInstScores, args)

    if perImageStats:
        wholeData["perImageScores"] = perImageStats

    return wholeData

def writeJSONFile(wholeData, args):
    path = os.path.dirname(args.exportFile)
    ensurePath(path)
    writeDict2JSON(wholeData, args.exportFile)

# Print confusion matrix
def printConfMatrix(confMatrix, args):
    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in args.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "))

    # print label names
    print("\b{text:>{width}} |".format(width=13, text=""), end=' ')
    for label in args.evalLabels:
        print("\b{text:^{width}} |".format(width=args.printRow, text=trainId2label[label].name[0]), end=' ')
    print("\b{text:>{width}} |".format(width=6, text="Prior"))

    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in args.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "))

    # print matrix
    for x in range(0, confMatrix.shape[0]):
        if (not x in args.evalLabels):
            continue
        # get prior of this label
        prior = getPrior(x, confMatrix)
        # skip if label does not exist in ground truth
        if prior < 1e-9:
            continue

        # print name
        name = trainId2label[x].name
        if len(name) > 13:
            name = name[:13]
        print("\b{text:>{width}} |".format(width=13,text=name), end=' ')
        # print matrix content
        for y in range(0, len(confMatrix[x])):
            if (not y in args.evalLabels):
                continue
            matrixFieldValue = getMatrixFieldValue(confMatrix, x, y, args)
            print(getColorEntry(matrixFieldValue, args) + "\b{text:>{width}.2f}  ".format(width=args.printRow, text=matrixFieldValue) + args.nocol, end=' ')
        # print prior
        print(getColorEntry(prior, args) + "\b{text:>{width}.4f} ".format(width=6, text=prior) + args.nocol)
    # print line
    print("\b{text:{fill}>{width}}".format(width=15, fill='-', text=" "), end=' ')
    for label in args.evalLabels:
        print("\b{text:{fill}>{width}}".format(width=args.printRow + 2, fill='-', text=" "), end=' ')
    print("\b{text:{fill}>{width}}".format(width=args.printRow + 3, fill='-', text=" "), end=' ')

# Print intersection-over-union scores for all classes.
def printClassScores(scoreList, instScoreList, args):
    if (args.quiet):
        return
    print(args.bold + "classes          IoU      nIoU" + args.nocol)
    print("--------------------------------")
    for label in args.evalLabels:
        if (trainId2label[label].ignoreInEval):
            continue
        labelName = str(trainId2label[label].name)
        iouStr = getColorEntry(scoreList[labelName], args) + "{val:>5.3f}".format(val=scoreList[labelName]) + args.nocol
        niouStr = getColorEntry(instScoreList[labelName], args) + "{val:>5.3f}".format(val=instScoreList[labelName]) + args.nocol
        print("{:<14}: ".format(labelName) + iouStr + "    " + niouStr)

# Print intersection-over-union scores for all categorys.
def printCategoryScores(scoreDict, instScoreDict, args):
    if (args.quiet):
        return
    print(args.bold + "categories       IoU      nIoU" + args.nocol)
    print("--------------------------------")
    for categoryName in scoreDict:
        if all( label.ignoreInEval for label in category2labels[categoryName] ):
            continue
        iouStr  = getColorEntry(scoreDict[categoryName], args) + "{val:>5.3f}".format(val=scoreDict[categoryName]) + args.nocol
        niouStr = getColorEntry(instScoreDict[categoryName], args) + "{val:>5.3f}".format(val=instScoreDict[categoryName]) + args.nocol
        print("{:<14}: ".format(categoryName) + iouStr + "    " + niouStr)

# Evaluate image lists pairwise.
def evaluateImgLists(predictionImgList, groundTruthImgList, args, vis_every=5):
    if len(predictionImgList) != len(groundTruthImgList):
        printError("List of images for prediction and groundtruth are not of equal size.")
    confMatrix    = generateMatrix(args)
    instStats     = generateInstanceStats(args)
    evalStats     = generateEvalStats()
    perImageStats = {}
    nbConfWeightedPoints      = 0

    if not args.quiet:
        print("Evaluating {} pairs of images...".format(len(predictionImgList)))

    # Evaluate all pairs of images and save them into a matrix
    for i in range(len(predictionImgList)):
        predictionImgFileName = predictionImgList[i]
        groundTruthImgFileName = groundTruthImgList[i]
        #print "Evaluate ", predictionImgFileName, "<>", groundTruthImgFileName
        visualizeIdx = i//vis_every if i % vis_every == 0 else -1
        nbConfWeightedPoints += evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, evalStats, perImageStats, visualizeIdx, args)

        if not args.quiet:
            print("\rImages Processed: {}".format(i+1), end=' ')
            sys.stdout.flush()

    if not args.quiet:
        print("\n")

    accuracy = float(evalStats['nAccuratePredictionCells']) / evalStats['nValidPredictionCells']
    completeness = float(evalStats['nCompleteGroundTruthCells']) / evalStats['nValidGroundTruthCells']
    fmean = 2.0 / (1/accuracy + 1/completeness)
    evalStats['averageAccuracy'] = accuracy
    evalStats['averageCompleteness'] = completeness
    evalStats['F1Score'] = fmean
    print('Overall accuracy %.2f, completeness %.2f, fmean %.2f' % (accuracy*100, completeness*100, fmean*100))

    # sanity check
    if not np.isclose(confMatrix.sum(), nbConfWeightedPoints):
        printError('Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(confMatrix.sum(),nbConfWeightedPoints))

    # print confusion matrix
    if (not args.quiet):
        printConfMatrix(confMatrix, args)

    # Calculate IOU scores on class level from matrix
    classScoreList = {}
    for label in args.evalLabels:
        labelName = trainId2label[label].name
        classScoreList[labelName] = getIouScoreForLabel(label, confMatrix, args)

    # Calculate instance IOU scores on class level from matrix
    classInstScoreList = {}
    for label in args.evalLabels:
        labelName = trainId2label[label].name
        classInstScoreList[labelName] = getInstanceIouScoreForLabel(label, confMatrix, instStats, args)

    # Print IOU scores
    if (not args.quiet):
        print("")
        print("")
        printClassScores(classScoreList, classInstScoreList, args)
        iouAvgStr  = getColorEntry(getScoreAverage(classScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(classScoreList, args)) + args.nocol
        niouAvgStr = getColorEntry(getScoreAverage(classInstScoreList , args), args) + "{avg:5.3f}".format(avg=getScoreAverage(classInstScoreList , args)) + args.nocol
        print("--------------------------------")
        print("Score Average : " + iouAvgStr + "    " + niouAvgStr)
        print("--------------------------------")
        print("")

    # Calculate IOU scores on category level from matrix
    categoryScoreList = {}
    for category in category2labels.keys():
        categoryScoreList[category] = getIouScoreForCategory(category,confMatrix,args)

    # Calculate instance IOU scores on category level from matrix
    categoryInstScoreList = {}
    for category in category2labels.keys():
        categoryInstScoreList[category] = getInstanceIouScoreForCategory(category,confMatrix,instStats,args)

    # Print IOU scores
    if (not args.quiet):
        print("")
        printCategoryScores(categoryScoreList, categoryInstScoreList, args)
        iouAvgStr = getColorEntry(getScoreAverage(categoryScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(categoryScoreList, args)) + args.nocol
        niouAvgStr = getColorEntry(getScoreAverage(categoryInstScoreList, args), args) + "{avg:5.3f}".format(avg=getScoreAverage(categoryInstScoreList, args)) + args.nocol
        print("--------------------------------")
        print("Score Average : " + iouAvgStr + "    " + niouAvgStr)
        print("--------------------------------")
        print("")

    # write result file
    allResultsDict = createResultDict( confMatrix, classScoreList, classInstScoreList, categoryScoreList, categoryInstScoreList, perImageStats, args )
    allResultsDict.update( evalStats )
    writeJSONFile( allResultsDict, args)

    # return confusion matrix
    return True


def id2trainIdImg(img):
    label = np.ones(img.shape) * -1
    ids = np.unique(img)
    for cid in ids:
        labelId = id2label[cid].trainId
        label[img == cid] = labelId
    return label.astype(np.uint8)


def writeResultPointCloud(predictionNp, predictionSemantic, accuracyMask, observedMask, visualizeIdx, maxNumPoint=10000):
    import plotly.offline
    import plotly.graph_objects as go
    interval = np.ceil(predictionNp.shape[0]/maxNumPoint).astype(np.int)
    x,y,z = predictionNp[::interval,:].T
    layout = go.Layout(scene=dict(aspectmode="data"))

    # create destination folders
    geometryPath = os.path.join(args.predictionPath, '..', 'geometry')
    semanticPath = os.path.join(args.predictionPath, '..', 'semantic')
    if not os.path.isdir(geometryPath):
        os.makedirs(geometryPath)
    if not os.path.isdir(semanticPath):
        os.makedirs(semanticPath)

    # visualize accuracy and completeness
    predictionColor = np.zeros((predictionNp.shape[0], 3))
    predictionColor[accuracyMask] = np.array([0.,1.,0.])
    predictionColor[np.logical_not(accuracyMask)] = np.array([1.,0.,0.])
    predictionColor[np.logical_not(observedMask)] = np.array([0.,0.,1.])
    predictionColor = predictionColor[::interval,:]
    data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', 
                       marker=dict(size=2, color=predictionColor))]
    plotlyStr = plotly.offline.plot({"data": data, "layout": layout}, include_plotlyjs=False, output_type='div')
    with open(os.path.join(geometryPath, '%d.php' % visualizeIdx), 'w') as f:
        f.write(plotlyStr)

    # visualize semantic
    if predictionSemantic is not None:
        predictionSemanticSub = predictionSemantic[::interval]
        predictionColor = np.zeros_like(predictionColor)
        for i in np.unique(predictionSemanticSub):
            predictionColor[predictionSemanticSub==i] = trainId2label[i].color
    data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', 
                       marker=dict(size=2, color=predictionColor/255.))]
    plotlyStr = plotly.offline.plot({"data": data, "layout": layout}, include_plotlyjs=False, output_type='div')
    with open(os.path.join(semanticPath, '%d.php' % visualizeIdx), 'w') as f:
        f.write(plotlyStr)



# Main evaluation method. Evaluates pairs of prediction and ground truth
# images which are passed as arguments.
def evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, evalStats, perImageStats, visualizeIdx, args):
    total_start_time = time.time()
    # Loading all resources for evaluation.
    try:
        if predictionImgFileName.endswith('.npy'):
            predictionNp = np.load(predictionImgFileName)
        else:
            predictionNp = np.loadtxt(predictionImgFileName)
        assert (predictionNp.shape[1]==3 or predictionNp.shape[1]==4)
        if predictionNp.shape[1]==4: # also evaluate semantic
            predictionSemanticRaw = predictionNp[:,-1]
            predictionNp = predictionNp[:,:-1]
            predictionColor = np.zeros((predictionSemanticRaw.shape[0], 3))
            predictionSemantic = np.zeros_like(predictionSemanticRaw)
            # convert to trainId
            for i in np.unique(predictionSemanticRaw):
                predictionColor[predictionSemanticRaw==i] = id2label[i].color
                predictionSemantic[predictionSemanticRaw==i] = id2label[i].trainId
        else:
            predictionSemantic = None
    except:
        printError("Unable to load " + predictionImgFileName)

    try:
        groundTruthImgFileName, accumulatedPcdFileName = groundTruthImgFileName
        cached_gt = np.load(groundTruthImgFileName)
        posesValid = cached_gt['posesValid']
        groundTruthNp = cached_gt['groundTruthNp']
        groundTruthColor = cached_gt['groundTruthColor']
        groundTruthLabel = cached_gt['groundTruthLabel']
        groundTruthConf = cached_gt['groundTruthConf']
        groundTruthTree = KDTree(groundTruthNp, leaf_size=args.leafSize)
    except:
        printError("Unable to load " + groundTruthImgFileName)

    nbConfWeightedPoints  = predictionNp.shape[0]

    # Evaluate chamfer distance
    predictionTree = KDTree(predictionNp, leaf_size=args.leafSize)
    print("groundTruthNp: ", groundTruthNp.shape)
    print("predictionNp: ", predictionNp.shape)
    start_time = time.time()
    completeDistance, completeIdx = predictionTree.query(groundTruthNp)
    completeDistance = completeDistance.flatten()
    print("Computed chamfer distance groundTruth to predictions: ", time.time()-start_time)
    start_time = time.time()
    accuracyDistance, _ = groundTruthTree.query(predictionNp)
    accuracyDistance = accuracyDistance.flatten()
    print("Computed chamfer distance predictions to groundTruth: ", time.time()-start_time)

    # Load free+occupied voxel centers to determine unknown region
    occupiedVoxelFile = accumulatedPcdFileName.replace('static', 'octomap').replace('.ply','.npy')
    if os.path.isfile(occupiedVoxelFile):
        observedCells = np.load(occupiedVoxelFile)
        observedTree = KDTree(observedCells[:,:3], leaf_size=args.leafSize)
        observedDistance, _ = observedTree.query(predictionNp)
        observedDistance = observedDistance.flatten()
        observedMask = observedDistance < args.observedVoxelSize
    else:
        print('Warning: no observed centers found! All points will be evaluated, including unobserved region.')
        observedMask = np.ones(predictionNp.shape[0],)

    # Further set regions out of the circle union to unknown region
    poseCenter = posesValid[:,1:].reshape(-1,3,4)[:,:,3]
    poseCenterTree = KDTree(poseCenter, leaf_size=args.leafSize)
    point2CenterDistance, _ = poseCenterTree.query(predictionNp)
    point2CenterDistance = point2CenterDistance.flatten()
    observedMask = np.logical_and(observedMask, point2CenterDistance < args.radius)
    
    # Voxelize to evaluate the accuracy/completeness independent to point cloud density
    groundTruthCells = getCellCoordinates(groundTruthNp, args.voxelSize)
    predictionCells = getCellCoordinates(predictionNp[observedMask], args.voxelSize)
    nValidGroundTruthCells = getNumUniqueCells(groundTruthCells)
    nValidPredictionCells = getNumUniqueCells(predictionCells)

    # evaluate completeness
    completeMask = completeDistance < args.thresholdComplete
    completeGroundTruthPoints = groundTruthNp[ completeMask ]
    completeGroundTruthCells = getCellCoordinates(completeGroundTruthPoints, args.voxelSize)
    nCompleteGroundTruthCells = getNumUniqueCells(completeGroundTruthCells)
    completeness = completeMask.mean()

    # evalute accuracy
    start_time = time.time()
    accuracyMask = accuracyDistance < args.thresholdAcc
    accuratePredictionPoints = predictionNp[ np.logical_and(accuracyMask, observedMask) ]
    accuratePredictionCells = getCellCoordinates(accuratePredictionPoints, args.voxelSize)
    nAccuratePredictionCells = getNumUniqueCells(accuratePredictionCells)
    accuracy = accuracyMask.mean()

    # evaluate semantic
    if predictionSemantic is not None: 
        fullGroundTruthPrediction = predictionSemantic[completeIdx.flatten()]
        fullGroundTruthPrediction[np.logical_not(completeMask)] = 255 # set label of incomplete region to 255

        if CSUPPORT:
            # using cython
            fullGroundTruthPrediction = fullGroundTruthPrediction.reshape(-1,1).astype(np.uint8)
            groundTruthLabel = groundTruthLabel.reshape(-1,1).astype(np.uint8)
            groundTruthConf = groundTruthConf.reshape(-1,1).astype(np.float64)
            confMatrix = addToConfusionMatrix.cEvaluatePair(fullGroundTruthPrediction, groundTruthLabel, groundTruthConf, confMatrix, args.evalLabels)
        else:
            # the slower python way 
            groundTruthConf = groundTruthConf.astype(np.float64)
            encoding_value = max(groundTruthLabel.max(), fullGroundTruthPrediction.max()).astype(np.int32) + 1
            encoded = (groundTruthLabel.astype(np.int32) * encoding_value) + fullGroundTruthPrediction

            values, cnt = np.unique(encoded, return_counts=True)

            for value, c in zip(values, cnt):
                pred_id = int(value % encoding_value)
                gt_id = int((value - pred_id)/encoding_value)
                if not gt_id in args.evalLabels:
                    printError("Unknown label with id {:}".format(gt_id))

                mask = np.logical_and(fullGroundTruthPrediction==pred_id, groundTruthLabel==gt_id)
                # confidence-weighted evaluation
                confMatrix[gt_id][pred_id] += np.sum(groundTruthConf[mask])
    evalStats['nValidGroundTruthCells'] += nValidGroundTruthCells
    evalStats['nAccuratePredictionCells'] += nAccuratePredictionCells
    evalStats['nValidPredictionCells'] += nValidPredictionCells
    evalStats['nCompleteGroundTruthCells'] += nCompleteGroundTruthCells

    print( 'Completeness voxel level %f, point level %f' %( 1.0 * nCompleteGroundTruthCells / nValidGroundTruthCells, completeness ))
    print( 'Accuracy voxel level %f, point level %f' %( 1.0 * nAccuratePredictionCells / nValidPredictionCells, accuracy ))

    if visualizeIdx>=0:
        writeResultPointCloud(predictionNp, predictionSemantic, accuracyMask, observedMask, visualizeIdx)

    print('Total time for one sample: ', time.time() - total_start_time)
    nbConfWeightedPoints = groundTruthConf.sum()
    return nbConfWeightedPoints

# The main method
def main():
    global args
    argv = sys.argv[1:]

    predictionImgList = []
    groundTruthImgList = []

    # we support the no-argument way only
    if len(argv) == 0:
        # use the ground truth search string specified above
        groundTruthImgList = getGroundTruth(args.groundTruthListFile)
        if not groundTruthImgList:
            printFunc("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(args.groundTruthSearch))
        # get the corresponding prediction for each ground truth imag
        for gt,_ in groundTruthImgList:
            predictionImgList.append( getPrediction(args,gt) )

        # evaluate
        success = evaluateImgLists(predictionImgList, groundTruthImgList, args)

    else:
        printError("Please specifiy the dataset and prediction root by setting environment variables KITTI360_DATASET and KITTI360_RESULTS.")

    return

# call the main method
if __name__ == "__main__":
    main()

