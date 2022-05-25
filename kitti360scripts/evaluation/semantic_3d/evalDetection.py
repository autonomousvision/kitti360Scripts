#!/usr/bin/python
#
# The evaluation script for pixel-level semantic labeling.
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
# 
# It is adapted from the Object Detection Evaluation code by 
# 
#    Charles R. Qi
# 
#    Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py
#
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
    
import os
import sys
import numpy as np
from scipy.spatial import ConvexHull
import fnmatch

# KITTI-360 imports
from kitti360scripts.helpers.csHelpers import *
from kitti360scripts.helpers.labels import id2label, trainId2label

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
            seq = int(line.split(' ')[0])
            startFrame = int(line.split(' ')[1])
            endFrame = int(line.split(' ')[2])
            groundTruthFile = os.path.join(rootPath, '%04d_%010d_%010d.npy' % (seq, startFrame, endFrame))
            if os.path.isfile(os.path.join(groundTruthFile)):
                groundTruthFiles.append(groundTruthFile)
            else:
                printError("Could not find %s. Please read the instructions of this method." % groundTruthFile)
    return groundTruthFiles

# The current implementation expects the results to be in a certain root folder.
# This folder is one of the following with decreasing priority:
#   - environment variable KITTI360_RESULTS
#   - environment variable KITTI360_DATASET/results
#   - ../../results/"
#
# Within the root folder, a matching prediction file is recursively searched.
# A file matches, if the filename follows the pattern
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

    filePattern = "{}*.npy".format( os.path.splitext(os.path.basename(groundTruthFile))[0] )

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


def polygonClip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def polyArea(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convexHullIntersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygonClip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3dVolume(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def rotz(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                    [s,  c,  0],
                    [0, 0,  1]])

def box3dIou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,1]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,1]) for i in range(3,-1,-1)] 
    area1 = polyArea(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = polyArea(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, interArea = convexHullIntersection(rect1, rect2)
    interArea = min(min(area1, area2), interArea)
    iou_2d = interArea/(area1+area2-interArea)
    zmax = min(corners1[0,2], corners2[0,2])
    zmin = max(corners1[4,2], corners2[4,2])
    interVol = interArea * max(0.0, zmax-zmin)
    vol1 = box3dVolume(corners1)
    vol2 = box3dVolume(corners2)
    iou = interVol / (vol1 + vol2 - interVol)
    return iou, iou_2d

def vocAP(rec, prec, use_07_metric=False):
    """ ap = vocAP(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get3dIoU(bb1,bb2):
    iou3d, iou2d = box3dIou(bb1,bb2)
    return iou3d

def evalDetectionClass(pred, gt, ovthresh=0.25, use_07_metric=False):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for bbox_ in bbox:
        #     ax.plot(*bbox_.T,'.')
        # ax.set_zlim3d([0,200])
        # plt.show()
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box,score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d,...].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get3dIoU(bb, BBGT[j,...])
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = vocAP(rec, prec, use_07_metric)

    return rec, prec, ap

def evalDetectionClassWrapper(arguments):
    pred, gt, ovthresh, use_07_metric = arguments
    rec, prec, ap = evalDetectionClass(pred, gt, ovthresh, use_07_metric)
    return (rec, prec, ap)

def evalDetection(pred_all, gt_all, ovthresh=0.25, use_07_metric=False):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {} # map {classname: pred}
    gt = {} # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox,score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    for classname in gt.keys():
        print('Computing AP for class: ', classname)
        rec[classname], prec[classname], ap[classname] = evalDetectionClass(pred[classname], gt[classname], ovthresh, use_07_metric)
        print(classname, ap[classname])
    
    return rec, prec, ap 

def evalDetectionMultiprocessing(pred_all, gt_all, ovthresh=0.25, use_07_metric=False):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {} # map {classname: pred}
    gt = {} # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox,score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    ret_values = []
    for classname in gt.keys():
        if classname in pred:
            ret_values.append( evalDetectionClassWrapper((pred[classname], gt[classname], ovthresh, use_07_metric)) )
           
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
    
    return rec, prec, ap 

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
    args.exportFile = "{}/resultBoundingBoxDetection.json".format(export_dir)
else:
    args.exportFile = os.path.join(args.kitti360Path, "evaluationResults", "resultBoundingBoxDetection.json")

# Parameters that should be modified by user
args.groundTruthListFile = os.path.join(args.kitti360Path, 'data_3d_semantics', 'train', '2013_05_28_drive_val.txt')

# Remaining params
args.JSONOutput         = True
args.quiet              = False
args.evaluateClasses    = ['building', 'car']

# store some parameters for finding predictions in the args variable
# the values are filled when the method getPrediction is first called
args.predictionPath = None
args.predictionWalk = None

args.apIouThresholds = [0.25, 0.5]

class APCalculator(object):
    ''' Calculating Average Precision '''
    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()
        
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = evalDetectionMultiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh)
        ret_dict = {} 
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision'%(clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall'%(clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall'%(clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {} # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {} # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0

def get3dBox(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = rotz(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def param2Bbox(params):
    center = params[0:3]
    size = params[3:6]
    headingAngle = params[6]
    classname = id2label[int(params[7])].name
    bboxVertices = get3dBox(size, headingAngle, center)
    return bboxVertices, classname

# Load pairs of prediction and ground truth bounding boxes.
def loadPair(predictionImgFileName, groundTruthImgFileName, args):
    bboxCenter = np.load(groundTruthImgFileName.replace('.npy', '_center.npy'))
    try:
        bboxParams = np.load(groundTruthImgFileName)
        groundTruthBboxes = []
        for bboxParam in bboxParams:
            bboxVertices, classname = param2Bbox(bboxParam)
            bboxVertices += bboxCenter
            if classname not in args.evaluateClasses:
                continue
            groundTruthBboxes.append((classname, bboxVertices))
    except:
        printError("Unable to load " + groundTruthImgFileName)
    try:
        bboxParams = np.load(predictionImgFileName)
        predictionBboxes = []
        for bboxParam in bboxParams:
            bboxVertices, classname = param2Bbox(bboxParam)
            confidence = bboxParam[-1]
            if classname not in args.evaluateClasses:
                continue
            predictionBboxes.append((classname, bboxVertices, confidence))
    except:
        printError("Unable to load " + predictionImgFileName)
    
    return groundTruthBboxes, predictionBboxes


# Evaluate image lists pairwise.
def evaluateImgLists(predictionImgList, groundTruthImgList, args, vis_every=5):
    if len(predictionImgList) != len(groundTruthImgList):
        printError("List of images for prediction and groundtruth are not of equal size.")

    ap_calculator_list = [APCalculator(iou_thresh) \
        for iou_thresh in args.apIouThresholds]

    if not args.quiet:
        print("Evaluating {} pairs of images...".format(len(predictionImgList)))

    # Evaluate all pairs of images and save them into a matrix
    for i in range(len(predictionImgList)):
        predictionImgFileName = predictionImgList[i]
        groundTruthImgFileName = groundTruthImgList[i]
        groundTruthBboxes, predictionBboxes = loadPair(predictionImgFileName, groundTruthImgFileName, args)

        for _, ap_calculator in enumerate(ap_calculator_list):
            ap_calculator.gt_map_cls[i] = groundTruthBboxes
            ap_calculator.pred_map_cls[i] = predictionBboxes

        if not args.quiet:
            print("\rImages Processed: {}".format(i+1), end=' ')
            sys.stdout.flush()
    if not args.quiet:
        print("\n")

    # Evaluate average precision
    allMetrics = {}
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-'*10, 'iou_thresh: %f'%(args.apIouThresholds[i]), '-'*10)
        metrics_dict = ap_calculator.compute_metrics()
        allMetrics[args.apIouThresholds[i]] = metrics_dict
        for key in metrics_dict:
            print('eval %s: %f'%(key, metrics_dict[key]))

    writeJSONFile( allMetrics, args)
    return True

def writeJSONFile(wholeData, args):
    path = os.path.dirname(args.exportFile)
    ensurePath(path)
    writeDict2JSON(wholeData, args.exportFile)

# The main method
def main():
    global args
    argv = sys.argv[1:]

    predictionImgList = []
    groundTruthImgList = []

    # the image lists can either be provided as arguments
    if (len(argv) >= 1):
        args.predictionPath = os.path.join(os.path.abspath("results"), argv[0], "data")
        args.exportFile = os.path.join(os.path.abspath("results"), argv[0], "resultBoundingBoxDetection.json")

    
    # use the ground truth search string specified above
    groundTruthImgList = getGroundTruth(args.groundTruthListFile)

    if not groundTruthImgList:
        printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(args.groundTruthSearch))
    # get the corresponding prediction for each ground truth imag
    for gt in groundTruthImgList:
        predictionImgList.append( getPrediction(args,gt) )

    # evaluate
    success = evaluateImgLists(predictionImgList, groundTruthImgList, args)

    return

# call the main method
if __name__ == "__main__":
    main()

