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

try:
    from itertools import izip
except ImportError:
    izip = zip

# KITTI-360 imports
import metric as metr
from kitti360scripts.helpers.csHelpers import *
sys.path.append('../semantic_2d')
from evalPixelLevelSemanticLabeling import evaluateImgLists as evaluateImgListsSemantic
import skimage.measure

# C Support
# Enable the cython support for faster evaluation
# Only tested for Ubuntu 64bit OS
CSUPPORT = True
# Check if C-Support is available for better performance
if CSUPPORT:
    try:
        from kitti360scripts.evaluation import addToConfusionMatrix
    except:
        CSUPPORT = False

# helper print function so we can switch from mail to standard printing
def printFunc( printStr, **kwargs ):
    print(printStr)

# Confidence is saved in the format of uint16
# Normalize confidence to [0,1] by dividing MAX_CONFIDENCE
MAX_CONFIDENCE=65535.0

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
#
# Within the root folder, a matching prediction file is recursively searched.
# A file matches, if the filename follows the pattern
#   {seq:0>4}_{frame:0>10}*.png
# for a ground truth filename
#   2013_05_28_drive_{seq:0>4}_sync/image_00/semantic/{frame:0>10}.png
def getPrediction( args, groundTruthFile, return_semantic=False ):
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
    filePattern = "{:04d}_{:010d}*.png".format( csFile.sequenceNb , csFile.frameNb )

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


def getGroundTruth(groundTruthListFile, eval_every=1, return_semantic=False):
    if 'KITTI360_DATASET' in os.environ:
        rootPath = os.environ['KITTI360_DATASET']
    else:
        rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

    if not os.path.isdir(rootPath):
        printError("Could not find a result root folder. Please read the instructions of this method.")

    if not os.path.isfile(groundTruthListFile):
        printError("Could not open %s. Please read the instructions of this method." % groundTruthListFile)

    with open(groundTruthListFile, 'r') as f:
        pairs = f.read().splitlines()

    groundTruthFiles = []
    for i,pair in enumerate(pairs):
        if i % eval_every == 0:
            if not return_semantic: # return RGB image
                groundTruthFile = os.path.join(rootPath, pair.split(' ')[0])
                groundTruthFiles.append(groundTruthFile)
            else: # return semantic image
                groundTruthFile = os.path.join(rootPath, pair.split(' ')[1])
                confidenceFile = os.path.join(os.path.dirname(os.path.dirname(groundTruthFile)),
                                              'confidence', os.path.basename(groundTruthFile))
                groundTruthFiles.append([groundTruthFile, confidenceFile])
            if not os.path.isfile(groundTruthFile):
                printError("Could not open %s. Please read the instructions of this method." % groundTruthFile)
    return groundTruthFiles
        
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
    args.exportNVSFile = "{}/resultNovelViewSynthesis.json".format(export_dir)
    args.exportFile = "{}/resultNovelViewLabelSynthesis.json".format(export_dir)
else:
    args.exportNVSFile = os.path.join(args.kitti360Path, "evaluationResults", "resultNovelViewSynthesis.json")
    args.exportFile = os.path.join(args.kitti360Path, "evaluationResults", "resultNovelViewLabelSynthesis.json")
# Parameters that should be modified by user
args.groundTruthSearch  = os.path.join( args.kitti360Path , "gtFine" , "val" , "*", "*_gtFine_labelIds.png" )
# Where to look for validation frames
args.groundTruthListFile = os.path.join('data', '2013_05_28_drive_train_drop50_frames.txt')

# Remaining params
args.evalPixelAccuracy  = True
args.evalSemantic       = False
args.evalLabels         = []
args.printRow           = 5
args.normalized         = True
args.colorized          = hasattr(sys.stderr, "isatty") and sys.stderr.isatty() and platform.system()=='Linux'
args.bold               = colors.BOLD if args.colorized else ""
args.nocol              = colors.ENDC if args.colorized else ""
args.JSONOutput         = True
args.quiet              = False


# store some parameters for finding predictions in the args variable
# the values are filled when the method getPrediction is first called
args.predictionPath = None
args.predictionWalk = None


#########################
# Methods
#########################


# create a dictionary containing all relevant results
def createResultDict( imageL1List, imagePSNRList, imageSSIMList, imageLPIPSList, args ):
    # write JSON result file
    wholeData = {}
    wholeData["imageL1List"] = imageL1List
    wholeData["imagePSNRList"] = imagePSNRList
    wholeData["imageSSIMList"] = imageSSIMList
    wholeData["imageLPIPSList"] = imageLPIPSList

    return wholeData

def writeJSONFile(wholeData, args):
    path = os.path.dirname(args.exportNVSFile)
    ensurePath(path)
    writeDict2JSON(wholeData, args.exportNVSFile)

def writeStatsFile(wholeData, args, outFile='stats360_nvs.txt'):
    path = os.path.dirname(args.exportNVSFile)
    ensurePath(path)
    with open(os.path.join(path, outFile), 'w') as f:
        f.write('%.4f %.4f %.4f %.4f' % (np.mean(wholeData['imageL1List']), 
                                    np.mean(wholeData['imagePSNRList']),
                                    np.mean(wholeData['imageSSIMList']),
                                    np.mean(wholeData['imageLPIPSList'])))
    
def writeResultImages(predictionImgList, gtImgList, args):
    predictions = np.array(sorted(predictionImgList))
    gts = np.array(sorted(gtImgList))

    # permutate images so the selection has a bit more variety
    np.random.seed(0)
    permutation = np.random.permutation(len(predictions))
    predictions = predictions[permutation]
    gts = gts[permutation]

    # we only visualize the first 10 images
    predictions = predictions[0:10]
    gts = gts[0:10]

    # create destination folders
    dstPred = os.path.join(args.predictionPath, "..", "pred")
    dstErr = os.path.join(args.predictionPath, "..", "err")
    if not os.path.isdir(dstPred):
        os.mkdir(dstPred)
    if not os.path.isdir(dstErr):
        os.mkdir(dstErr)

    # colorize the error images

    for i in range(len(predictions)):
        from shutil import copyfile
        copyfile(predictions[i], os.path.join(dstPred, "{num:010d}.png".format(num=i)))
    
        pred = np.array(Image.open(predictions[i]))
        gt = np.array(Image.open(gts[i]))

        _, err = skimage.measure.compare_ssim(
            gt, pred, multichannel=True, full=True 
        )
        # normalize ssim from [-1,1] to [0,1]
        err = 1 - (1 + err) / 2
        err = np.mean(err, axis=2)

        # save ssim image
        errors = (err*255).astype(np.uint8)
        Image.fromarray(errors).save(os.path.join(dstErr, "{num:010d}.png".format(num=i)))

    return True

def writeSemanticResultImages(predictionImgList, gtImgList, args):
    predictions = np.array(sorted(predictionImgList))
    gts = np.array(sorted([gt[0] for gt in gtImgList]))
    confs = np.array(sorted([gt[1] for gt in gtImgList]))

    # permutate images so the selection has a bit more variety
    np.random.seed(0)
    permutation = np.random.permutation(len(predictions))
    predictions = predictions[permutation]
    gts = gts[permutation]
    confs = confs[permutation]

    # we only visualize the first 10 images
    predictions = predictions[0:10]
    gts = gts[0:10]
    confs = confs[0:10]

    # create destination folders
    dstPred = os.path.join(args.predictionPath, "..", "pred")
    dstErr = os.path.join(args.predictionPath, "..", "err")
    if not os.path.isdir(dstPred):
        os.mkdir(dstPred)
    if not os.path.isdir(dstErr):
        os.mkdir(dstErr)

    # colorize the images
    id2color = np.array([id2label[x].color for x in id2label])
    id2category = np.array([id2label[x].categoryId for x in id2label])
    ignoreInEval = np.array([not id2label[x].ignoreInEval for x in id2label], dtype=np.uint8)

    for i in range(len(predictions)):
        from shutil import copyfile
    
        pred = np.array(Image.open(predictions[i]))
        gt = np.array(Image.open(gts[i]))
        conf = np.array(Image.open(confs[i])) / MAX_CONFIDENCE

        outOfRange = pred > (len(id2category) - 1)
        pred[outOfRange] = 0

        predCat = id2category[pred]
        gtCat = id2category[gt]
        catFalse = np.uint8(predCat != gtCat)
        predFalse = np.uint8(pred != gt)
        valid = ignoreInEval[gt]
        r = (np.uint8((catFalse + predFalse) > 0)*valid*255)*(np.logical_not(outOfRange)) * conf
        g = (np.uint8((catFalse + predFalse) < 2)*valid*255)*(np.logical_not(outOfRange)) * conf
        b = np.zeros(catFalse.shape)

        errors = np.array(np.dstack((r, g, b)), dtype=np.uint8)

        predColored = np.array(id2color[pred], dtype=np.uint8)
        Image.fromarray(predColored).save(os.path.join(dstPred, "{num:010d}.png".format(num=i)))
        Image.fromarray(errors).save(os.path.join(dstErr, "{num:010d}.png".format(num=i)))

    return True

# Evaluate image lists pairwise.
def evaluateImgLists(predictionImgList, groundTruthImgList, args):
    if len(predictionImgList) != len(groundTruthImgList):
        printError("List of images for prediction and groundtruth are not of equal size.")

    if not args.quiet:
        print("Evaluating {} pairs of images...".format(len(predictionImgList)))

    # Evaluate all pairs of images and save them into a matrix
    imageL1List = []
    imagePSNRList = []
    imageSSIMList = []
    imageLPIPSList = []
    for i in range(len(predictionImgList)):
        predictionImgFileName = predictionImgList[i]
        groundTruthImgFileName = groundTruthImgList[i]
        ret = evaluatePair(predictionImgFileName, groundTruthImgFileName, args)

        imageL1List.append(ret['dist1_mean'])
        imagePSNRList.append(ret['distpsnr_mean'])
        imageSSIMList.append(ret['distssim_mean'])
        imageLPIPSList.append(ret['distlpips_mean'])

        if not args.quiet:
            print("\rImages Processed: {}".format(i+1), end=' ')
            sys.stdout.flush()
    if not args.quiet:
        print("\n")

    # write result file
    allResultsDict = createResultDict( imageL1List, imagePSNRList, imageSSIMList, imageLPIPSList, args )
    writeJSONFile( allResultsDict, args )
    writeStatsFile( allResultsDict, args )

    # print results
    print('Average L1   : %.06f' % np.mean(imageL1List))
    print('Average PSNR : %.06f' % np.mean(imagePSNRList))
    print('Average SSIM : %.06f' % np.mean(imageSSIMList))
    print('Average LPIPS: %.06f' % np.mean(imageLPIPSList))

    # return confusion matrix
    return allResultsDict


# Main evaluation method. Evaluates pairs of prediction and ground truth
# images which are passed as arguments.
def evaluatePair(predictionImgFileName, groundTruthImgFileName, args):
    # Loading all resources for evaluation.
    try:
        predictionImg = Image.open(predictionImgFileName)
        predictionNp  = np.array(predictionImg)
    except:
        printError("Unable to load " + predictionImgFileName)
    try:
        groundTruthImg = Image.open(groundTruthImgFileName)
        groundTruthNp = np.array(groundTruthImg)
    except:
        printError("Unable to load " + groundTruthImgFileName)

    # Check for equal image sizes
    if (predictionImg.size[0] != groundTruthImg.size[0]):
        printError("Image widths of " + predictionImgFileName + " and " + groundTruthImgFileName + " are not equal.")
    if (predictionImg.size[1] != groundTruthImg.size[1]):
        printError("Image heights of " + predictionImgFileName + " and " + groundTruthImgFileName + " are not equal.")
    if ( len(predictionNp.shape) != 3 ):
        printError("Predicted image has wrong number of dimensions.")
    if ( predictionNp.shape[2] != 3 ):
        printError("Predicted image has wrong number of channels.")

    imgWidth  = predictionImg.size[0]
    imgHeight = predictionImg.size[1]

    # record metrics
    metric = {}
    metric["rgb"] = metr.MultipleMetric(
        metrics=[
          metr.DistanceMetric(p=1, vec_length=3),
          metr.PSNRMetric(),
          metr.SSIMMetric(),
          metr.LPIPSMetric(),
        ]
      )

    #Normalize to 0 1 
    predictionNp= np.expand_dims(predictionNp, axis=0)/255.
    groundTruthNp = np.expand_dims(groundTruthNp, axis=0)/255.
    metric["rgb"].add(predictionNp, groundTruthNp)
    ret = metric["rgb"].get()

    return ret

# The main method
def main():
    global args
    argv = sys.argv[1:]

    predictionImgList = []
    groundTruthImgList = []
    predictionSemanticList = []
    groundTruthSemanticList = []

    # we support the no-argument way only as the groundTruthImgList should contain paths to both semantic and confidence maps
    if len(argv) == 0:

        # use the ground truth search string specified above
        groundTruthImgList = getGroundTruth(args.groundTruthListFile)
        if not groundTruthImgList:
            printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(args.groundTruthSearch))
        # get the corresponding prediction for each ground truth imag
        for gt in groundTruthImgList:
            predictionImgList.append( getPrediction(args,gt) )

        # evaluate novel view semantic synthesis
        if args.evalSemantic==1:
            # use the ground truth search string specified above
            groundTruthSemanticList = getGroundTruth(args.groundTruthListFile,return_semantic=True)
            if not groundTruthSemanticList:
                printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(args.groundTruthSearch))
            # get the corresponding prediction for each ground truth imag
            for gt,_ in groundTruthSemanticList:
                predictionSemanticList.append( getPrediction(args,gt,return_semantic=True) )

            # evaluate
            success = evaluateImgListsSemantic(predictionSemanticList, groundTruthSemanticList, args)

            # save some images for visualization
            success = writeSemanticResultImages(predictionImgList, groundTruthSemanticList, args)

        # evaluate novel view appearance synthesis
        else:
            evaluateImgLists(predictionImgList, groundTruthImgList, args)

            # save some images for visualization
            success = writeResultImages( predictionImgList, groundTruthImgList, args )

    return

# call the main method
if __name__ == "__main__":
    main()
