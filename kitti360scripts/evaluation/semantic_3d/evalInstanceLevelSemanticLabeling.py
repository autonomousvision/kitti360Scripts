#!/usr/bin/python
#
# The evaluation script for instance-level semantic labeling.
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
#
# Please check the description of the "getPrediction" method below
# and set the required environment variables as needed, such that
# this script can locate your results.
# If the default implementation of the method works, then it's most likely
# that our evaluation server will be able to process your results as well.
#
# To run this script, make sure that your results contain text files
# (one for each test set image) with the content:
#   labelIDPrediction1 confidencePrediction1
#   labelIDPrediction2 confidencePrediction2
#   labelIDPrediction3 confidencePrediction3
#   ...
#
# - The label IDs "labelIDPrediction" specify the class of that mask,
# encoded as defined in labels.py. Note that the regular ID is used,
# not the train ID.
# - The field "confidencePrediction" is a float value that assigns a
# confidence score to the mask.
#
# Note that this tool creates a file named "gtInstances.json" during its
# first run. This file helps to speed up computation and should be deleted
# whenever anything changes in the ground truth annotations or anything
# goes wrong.

# python imports
from __future__ import print_function, absolute_import, division
import os, sys
import fnmatch
from copy import deepcopy
import numpy as np

# KITTI-360 imports
from kitti360scripts.helpers.csHelpers import *
from kitti360scripts.helpers.labels import labels, id2label, trainId2label
from kitti360scripts.helpers.annotation import Annotation3DInstance 
from kitti360scripts.helpers.ply import read_ply


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
#   {seq:0>4}_{start_frame:0>10}_{end_frame:0>10}*.txt
# or
#   {seq:0>4}_{start_frame:0>10}_{end_frame:0>10}*.npy
# for a ground truth filename
#   2013_05_28_drive_{seq:0>4}_sync/static/{start_frame:0>10}_{end_frame:0>10}.ply
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


    csWindow = getWindowInfo(groundTruthFile)
    txtFilePattern = "{:04d}_{:010d}_{:010d}*.txt".format( csWindow.sequenceNb, csWindow.firstFrameNb , csWindow.lastFrameNb )
    npyFilePattern = "{:04d}_{:010d}_{:010d}*.npy".format( csWindow.sequenceNb, csWindow.firstFrameNb , csWindow.lastFrameNb )

    predictionTxtFile = None
    for root, filenames in args.predictionWalk:
        for filename in fnmatch.filter(filenames, txtFilePattern):
            if not predictionTxtFile:
                predictionTxtFile = os.path.join(root, filename)
            else:
                printError("Found multiple .txt predictions for ground truth {}".format(groundTruthFile))

    predictionNpyFile = None
    for root, filenames in args.predictionWalk:
        for filename in fnmatch.filter(filenames, npyFilePattern):
            if not predictionNpyFile:
                predictionNpyFile = os.path.join(root, filename)
            else:
                printError("Found multiple .npy predictions for ground truth {}".format(groundTruthFile))

    if not predictionTxtFile:
        print("Found no .txt prediction for ground truth {}".format(groundTruthFile))

    if not predictionNpyFile:
        print("Found no .npy prediction for ground truth {}".format(groundTruthFile))

    return (predictionTxtFile, predictionNpyFile)


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
            groundTruthFile = os.path.join(rootPath, line)
            if not os.path.isfile(groundTruthFile):
                printError("Could not open %s. Please read the instructions of this method." % groundTruthFile)
            groundTruthFiles.append(groundTruthFile)
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
    args.exportFile = "{}/resultInstanceLevelSemanticLabeling.json".format(export_dir)
else:
    args.exportFile = os.path.join(args.kitti360Path, "evaluationResults", "resultInstanceLevelSemanticLabeling.json")

# Parameters that should be modified by user
args.groundTruthListFile = os.path.join(args.kitti360Path, 'data_3d_semantics', 'train', '2013_05_28_drive_val.txt')

# classes for evaluation 
args.validClassLabels = ['building', 'car']
args.validClassIds    = [name2label[l].trainId for l in args.validClassLabels] 
args.id2label = {}
args.label2id = {}
for i in range(len(args.validClassIds)):
    args.label2id[args.validClassLabels[i]] = args.validClassIds[i]
    args.id2label[args.validClassIds[i]] = args.validClassLabels[i]

# overlaps for evaluation
args.overlaps           = np.append(np.arange(0.5,0.95,0.05), 0.25)
# minimum region size for evaluation [points]
args.minRegionSizes     = np.array( [ 100 ] )
# distance thresholds [m]
args.distanceThs        = np.array( [  float('inf') ] )
# distance confidences
args.distanceConfs      = np.array( [ -float('inf') ] )

args.gtInstancesFile    = os.path.join(os.path.dirname(os.path.realpath(__file__)),'gtInstances.json')
args.JSONOutput         = True
args.quiet              = False
args.csv                = False
args.colorized          = True
args.instLabels         = []

# store some parameters for finding predictions in the args variable
# the values are filled when the method getPrediction is first called
args.predictionPath = None
args.predictionWalk = None


# Determine the labels that have instances
def setInstanceLabels(args):
    args.instLabels = []
    for label in args.validClassLabels:
        args.instLabels.append(label)

# Read prediction info
# imgFile, predId, confidence
def readPredInfo(predInfoFileNames,args):
    predInfo = {}
    predInfoFileName, predMaskFileName = predInfoFileNames
    if (not os.path.isfile(predInfoFileName)):
        printError("Infofile '{}' for the predictions not found.".format(predInfoFileName))
    with open(predInfoFileName, 'r') as f:
        for i, line in enumerate(f):
            splittedLine         = line.split(" ")
            if len(splittedLine) != 2:
                printError( "Invalid prediction file. Expected content: labelIDPrediction1 confidencePrediction1" )

            imageInfo            = {}
            imageInfo["labelId"] = int(float(splittedLine[0]))
            imageInfo["conf"]    = float(splittedLine[1])
            # Convert id to trainId
            imageInfo["labelId"] = id2label[imageInfo["labelId"]].trainId
            predInfo[i]   = imageInfo
    # Load prediction mask in a single numpy file
    if (not os.path.isfile(predMaskFileName)):
        printError("Mask file '{}' for the predictions not found.".format(predMaskFileName))

    predMask = np.load(predMaskFileName)

    if (predMask.max() != len(predInfo.keys())):
        printError("Number of masks does not match with number of instances!")

    for i, imageInfo in predInfo.items():
        imageInfo.update({"mask": predMask==(i+1)})

    return predInfo

def id2trainIdImg(img):
    label = np.ones(img.shape) * -1
    ids = np.unique(img)
    for cid in ids:
        labelId = id2label[cid].trainId
        label[img == cid] = labelId
    return label.astype(np.uint8)

# Routine to read ground truth point cloud
def readGTImage(groundTruthImgFileName):
    cachedGroundTruthNpFile = os.path.splitext(groundTruthImgFileName)[0] + '_instance3d.npz'
    if not os.path.isfile(cachedGroundTruthNpFile):
        groundTruthPcd = read_ply(groundTruthImgFileName)
        groundTruthInstance = groundTruthPcd['instance'] % 1000
        groundTruthSemantic = groundTruthPcd['semantic'] 
        groundTruthSemantic = id2trainIdImg(groundTruthSemantic)
        groundTruthInstance = groundTruthSemantic * 1000 + groundTruthInstance
        groundTruthInstance[groundTruthSemantic==255] = 0
        groundTruthConf = groundTruthPcd['confidence'].astype(np.float64)
        np.savez(cachedGroundTruthNpFile, **{'groundTruthInstance':groundTruthInstance, 'groundTruthConf': groundTruthConf.astype(np.float32)})
    else:
        cachedGroundTruthNp = np.load(cachedGroundTruthNpFile )
        groundTruthInstance = cachedGroundTruthNp['groundTruthInstance']
        groundTruthConf = cachedGroundTruthNp['groundTruthConf'].astype(np.float64)
    return groundTruthInstance, groundTruthConf

def getGTInstances(gtIds, gtConf, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(gtIds)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Annotation3DInstance(gtIds, gtConf, id)
        if inst.labelId in class_ids:
            instances[id2label[inst.labelId]].append(inst.to_dict())
    return instances

def evaluateMatches(matches, args):
    overlaps = args.overlaps
    min_region_sizes = [args.minRegionSizes[0]]
    dist_threshes = [args.distanceThs[0]]
    dist_confs = [args.distanceConfs[0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(args.validClassLabels), len(overlaps)), np.float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['prediction']:
                    for label_name in args.validClassLabels:
                        for p in matches[m]['prediction'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
            for li, label_name in enumerate(args.validClassLabels):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    predInstances = matches[m]['prediction'][label_name]
                    gtInstances = matches[m]['groundTruth'][label_name]
                    # filter groups in ground truth
                    gtInstances = [gt for gt in gtInstances if
                                    gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and 
                                    gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gtInstances:
                        has_gt = True
                    if predInstances:
                        has_pred = True

                    cur_true = np.ones(len(gtInstances))
                    cur_score = np.ones(len(gtInstances)) * (-float("inf"))
                    cur_match = np.zeros(len(gtInstances), dtype=np.bool)
                    # collect matches
                    for (gti, gt) in enumerate(gtInstances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (
                            gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in predInstances:
                        found_gt = False
                        for gt in pred['matchedGt']:
                            overlap = float(gt['intersection']) / (
                            gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['voidIntersection']
                            for gt in pred['matchedGt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt['dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    if(len(y_true_sorted_cumsum) == 0):
                        num_true_examples = 0
                    else:
                        num_true_examples = y_true_sorted_cumsum[-1]
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap


def computeAverages(aps, args):
    dInf = 0
    o50   = np.where(np.isclose(args.overlaps,0.5))
    o25   = np.where(np.isclose(args.overlaps,0.25))
    oAllBut25  = np.where(np.logical_not(np.isclose(args.overlaps,0.25)))
    avg_dict = {}
    #avg_dict['allAp']     = np.nanmean(aps[ dInf,:,:  ])
    avg_dict['allAp']     = np.nanmean(aps[ dInf,:,oAllBut25])
    avg_dict['allAp50%'] = np.nanmean(aps[ dInf,:,o50])
    avg_dict['allAp25%'] = np.nanmean(aps[ dInf,:,o25])
    avg_dict["classes"]  = {}
    for (li,label_name) in enumerate(args.validClassLabels):
        avg_dict["classes"][label_name]             = {}
        #avg_dict["classes"][label_name]["ap"]       = np.average(aps[ dInf,li,  :])
        avg_dict["classes"][label_name]["ap"]       = np.average(aps[ dInf,li,oAllBut25])
        avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ dInf,li,o50])
        avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ dInf,li,o25])
    return avg_dict

# match ground truth instances with predicted instances
def matchGtWithPreds(predictionList,groundTruthList,args):
    matches = {}
    if not args.quiet:
        print("Matching {} pairs of images...".format(len(predictionList)))

    count = 0
    for (pred,gt) in zip(predictionList,groundTruthList):
        # key for dicts
        dictKey = os.path.abspath(gt)

        # Read input files
        predInfo = readPredInfo(pred,args)

        # Try to assign all predictions
        test_scene_name = os.path.basename(pred[0])
        (curGtInstances,curPredInstances) = assignGt2Preds(test_scene_name, predInfo, gt)

        # append to global dict
        matches[ dictKey ] = {}
        matches[ dictKey ]["groundTruth"] = curGtInstances
        matches[ dictKey ]["prediction"]  = curPredInstances

        count += 1
        if not args.quiet:
            print("\rImages Processed: {}".format(count), end=' ')
            sys.stdout.flush()

    if not args.quiet:
        print("")

    return matches

# For a point cloud, assign all predicted instances to ground truth instances
def assignGt2Preds(scene_name, predInfo, gt_file):
    try:
        gtIds, gtConf = readGTImage(gt_file)
    except Exception as e:
        printError('unable to load ' + gt_file + ': ' + str(e))

    # get gt instances
    gtInstances = getGTInstances(gtIds, gtConf, args.validClassIds, args.validClassLabels, args.id2label)

    # associate
    gt2pred = gtInstances.copy()
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in args.validClassLabels:
        pred2gt[label] = []
    num_predInstances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gtIds//1000, args.validClassIds))

    # Loop through all prediction masks
    nMask = len(predInfo.keys())
    for i in range(nMask):
        labelId = int(predInfo[i]['labelId'])
        conf = predInfo[i]['conf']
        if not labelId in args.id2label:
            continue
        label_name = args.id2label[labelId]
        # read the mask
        pred_mask = predInfo[i]['mask']   # (N), long
        if len(pred_mask) != len(gtIds):
            printError('wrong number of lines in mask#%d: ' % (i)  + '(%d) vs #mesh vertices (%d)' % (len(pred_mask), len(gtIds)))
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        # confidence-weighted number of points
        num = np.sum(gtConf[pred_mask])
        if num < args.minRegionSizes[0]:
            continue  # skip if empty

        predInstance = {}
        predInstance['filename'] = '{}_{:03d}'.format(scene_name, num_predInstances)
        predInstance['pred_id'] = num_predInstances
        predInstance['labelId'] = labelId
        predInstance['vert_count'] = num
        predInstance['confidence'] = conf
        # confidence-weighted void intersection
        predInstance['voidIntersection'] = np.sum(gtConf[np.logical_and(bool_void, pred_mask)])

        # A list of all overlapping ground truth instances
        matchedGt = []

        # go through all gt instances with matching label
        # Loop through all ground truth instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            # confidence-weighted intersection
            intersection = np.sum(gtConf[np.logical_and(gtIds == gt_inst['instance_id'], pred_mask)])
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = predInstance.copy()
                gt_copy['intersection']   = intersection
                pred_copy['intersection'] = intersection
                matchedGt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        predInstance['matchedGt'] = matchedGt
        num_predInstances += 1
        pred2gt[label_name].append(predInstance)

    return gt2pred, pred2gt

def printResults(avgDict, args):
    sep     = (","         if args.csv       else "")
    col1    = (":"         if not args.csv   else "")
    noCol   = (colors.ENDC if args.colorized else "")
    bold    = (colors.BOLD if args.colorized else "")
    lineLen = 64

    print("")
    if not args.csv:
        print("#"*lineLen)
    line  = bold
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    line += noCol
    print(line)
    if not args.csv:
        print("#"*lineLen)

    for (lI,labelName) in enumerate(args.instLabels):
        apAvg  = avgDict["classes"][labelName]["ap"]
        ap50o  = avgDict["classes"][labelName]["ap50%"]
        ap25o  = avgDict["classes"][labelName]["ap25%"]

        line  = "{:<15}".format(labelName) + sep + col1
        line += getColorEntry(apAvg , args) + sep + "{:>15.3f}".format(apAvg ) + sep
        line += getColorEntry(ap50o , args) + sep + "{:>15.3f}".format(ap50o ) + sep
        line += getColorEntry(ap25o , args) + sep + "{:>15.3f}".format(ap25o ) + sep
        line += noCol
        print(line)

    allApAvg  = avgDict["allAp"]
    allAp50o  = avgDict["allAp50%"]
    allAp25o  = avgDict["allAp25%"]

    if not args.csv:
            print("-"*lineLen)
    line  = "{:<15}".format("average") + sep + col1
    line += getColorEntry(allApAvg , args) + sep + "{:>15.3f}".format(allApAvg)  + sep
    line += getColorEntry(allAp50o , args) + sep + "{:>15.3f}".format(allAp50o)  + sep
    line += getColorEntry(allAp25o , args) + sep + "{:>15.3f}".format(allAp25o)  + sep
    line += noCol
    print(line)
    print("")

def prepareJSONDataForResults(avgDict, aps, args):
    JSONData = {}
    JSONData["averages"] = avgDict
    JSONData["overlaps"] = args.overlaps.tolist()
    JSONData["minRegionSizes"]      = args.minRegionSizes.tolist()
    JSONData["distanceThresholds"]  = args.distanceThs.tolist()
    JSONData["minStereoDensities"]  = args.distanceConfs.tolist()
    JSONData["instLabels"] = args.instLabels
    JSONData["resultApMatrix"] = aps.tolist()

    return JSONData

# Work through image list
def evaluateImgLists(predictionList, groundTruthList, args):
    # determine labels of interest
    setInstanceLabels(args)
    # get dictionary of all ground truth instances
    #gtInstances = getGtInstances(groundTruthList,args)
    # match predictions and ground truth
    matches = matchGtWithPreds(predictionList,groundTruthList,args)
    writeDict2JSON(matches,"matches.json")
    # evaluate matches
    apScores = evaluateMatches(matches, args)
    # averages
    avgDict = computeAverages(apScores,args)
    # result dict
    resDict = prepareJSONDataForResults(avgDict, apScores, args)
    if args.JSONOutput:
        # create output folder if necessary
        path = os.path.dirname(args.exportFile)
        ensurePath(path)
        # Write APs to JSON
        writeDict2JSON(resDict, args.exportFile)

    if not args.quiet:
        # Print results
        printResults(avgDict, args)

    return resDict

# The main method
def main():
    global args
    argv = sys.argv[1:]

    predictionImgList = []
    groundTruthImgList = []

    # the image lists can either be provided as arguments
    if (len(argv) > 3):
        for arg in argv:
            if ("gt" in arg or "groundtruth" in arg):
                groundTruthImgList.append(arg)
            elif ("pred" in arg):
                predictionImgList.append(arg)
    # however the no-argument way is prefered
    elif len(argv) == 0:
        # use the ground truth search string specified above
        groundTruthImgList = getGroundTruth(args.groundTruthListFile)
        if not groundTruthImgList:
            printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(args.groundTruthListFile))
        # get the corresponding prediction for each ground truth imag
        for gt in groundTruthImgList:
            predictionImgList.append( getPrediction(args,gt) )

    # evaluate
    evaluateImgLists(predictionImgList, groundTruthImgList, args)

    return

# call the main method
if __name__ == "__main__":
    main()
