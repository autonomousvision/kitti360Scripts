#!/usr/bin/python
# -*- coding: utf-8 -*-


#################
## Import modules
#################

from __future__ import print_function, absolute_import, division
# get command line parameters
import sys
# walk directories
import glob
# access to OS functionality
import os
# call processes
import subprocess
# copy things
import copy
# numpy
import numpy as np
# matplotlib for colormaps
try:
    import matplotlib.colors
    import matplotlib.cm
    from PIL import PILLOW_VERSION
    from PIL import Image
except:
    pass

# the label tool was originally written for python 2 and pyqt4
# in order to enable compatibility with python 3, we need
# to fix the pyqt api to the old version that is default in py2
import sip
apis = ['QDate', 'QDateTime', 'QString', 'QTextStream', 'QTime', 'QUrl', 'QVariant']
for a in apis:
    sip.setapi(a, 1)

# import pyqt for everything graphical
from PyQt5 import QtCore, QtGui, QtWidgets

#################
## Helper classes
#################

# annotation helper
from kitti360scripts.helpers.annotation  import Annotation2D, Annotation2DInstance, Annotation3D
from kitti360scripts.helpers.labels     import name2label, id2label, assureSingleInstanceName
from kitti360scripts.helpers.project import CameraPerspective as Camera

#################
## Main GUI class
#################

# The main class which is a QtGui -> Main Window
class Kitti360Viewer(QtWidgets.QMainWindow):

    #############################
    ## Construction / Destruction
    #############################

    # Constructor
    def __init__(self, screenWidth=1080):
        # Construct base class
        super(Kitti360Viewer, self).__init__()

        # This is the configuration.

        # The sequence of the image we currently working on
        self.currentSequence   = ""
        # The filename of the image we currently working on
        self.currentFile       = ""
        # The filename of the labels we currently working on
        self.currentLabelFile  = ""
        # The path of the images of the currently loaded sequence 
        self.sequence          = ""
        # The name of the currently loaded sequence 
        self.sequenceName      = ""
        # Ground truth type
        self.gtType            = "semantic" 
        # The path of the labels. In this folder we expect a folder for each sequence 
        # Within these sequence folders we expect the label with a filename matching
        # the images, except for the extension
        self.labelPath         = ""
        # The transparency of the labels over the image
        self.transp            = 0.5
        # The zoom toggle
        self.zoom              = False
        # The zoom factor
        self.zoomFactor        = 1.5
        # The size of the zoom window. Currently there is no setter or getter for that
        self.zoomSize          = 400 #px
        # The width of the screen
        self.screenWidth       = screenWidth

        # The width that we actually use to show the image
        self.w                 = 0
        # The height that we actually use to show the image
        self.h                 = 0
        # The horizontal offset where we start drawing within the widget
        self.xoff              = 0
        # The vertical offset where we start drawing withing the widget
        self.yoff              = 0
        # A gap that we  leave around the image as little border
        self.bordergap         = 20
        # The scale that was used, ie
        # self.w = self.scale * self.image.width()
        # self.h = self.scale * self.image.height()
        self.scale             = 1.0
        # Filenames of all images in current sequence
        self.images            = []
        self.imagesSequence    = []
        # Filenames of all image labels in current sequence 
        self.label_images      = []
        # Image extension
        self.imageExt          = ".png"
        # Ground truth extension
        self.gtExt             = "_gt*.json"
        # Current image as QImage
        self.image             = QtGui.QImage()
        # Index of the current image within the sequence folder
        self.idx               = 0
        # All annotated objects in current image, i.e. list of csPoly or csBbox
        self.annotation2D      = []
        # All annotated 3D points in current image, i.e. dictionary of 3D points
        self.annotationSparse3D= None
        # All annotated 3D objects in current sequence, i.e. dictionary of csBbox3D
        self.annotation3D      = None 
        # The current object the mouse points to. It's index in self.labels
        self.mouseObj          = -1
        # The current object the mouse points to. It's index in self.labels
        self.mouseSemanticId   = -1
        self.mouseInstanceId   = -1
        # The current object the mouse seletect via clicking.
        self.mousePressObj     = -1
        # The object that is highlighted and its label. An object instance
        self.highlightObj      = None
        self.highlightObjSparse= None
        self.highlightObjLabel = None
        # The position of the mouse
        self.mousePosOrig      = None
        # The position of the mouse scaled to label coordinates
        self.mousePosScaled    = None
        # If the mouse is outside of the image
        self.mouseOutsideImage = True
        # The position of the mouse upon enabling the zoom window
        self.mousePosOnZoom    = None
        # A list of toolbar actions that need an image
        self.actImage          = []
        # A list of toolbar actions that need an image that is not the first
        self.actImageNotFirst  = []
        # A list of toolbar actions that need an image that is not the last
        self.actImageNotLast   = []
        # Toggle status of the play icon
        self.playState         = False
        # Enable disparity visu in general
        self.enableDisparity   = True
        # Show disparities instead of labels
        self.showDisparity     = False
        # Show point cloud or not
        self.showSparse        = False
        # 
        self.camera            = None 
        #
        self.cameraId          = 0
        # Generate colormap
        try:
            norm = matplotlib.colors.Normalize(vmin=3,vmax=100)
            cmap = matplotlib.cm.plasma
            self.colormap = matplotlib.cm.ScalarMappable( norm=norm , cmap=cmap )
        except:
            self.enableDisparity = False
        # check if pillow was imported, otherwise no disparity visu possible
        if not 'PILLOW_VERSION' in globals():
            self.enableDisparity = False

        # Default label
        self.defaultLabel = 'static'
        if self.defaultLabel not in name2label:
            print('The {0} label is missing in the internal label definitions.'.format(self.defaultLabel))
            return
        # Last selected label
        self.lastLabel = self.defaultLabel

        # Setup the GUI
        self.initUI()

        # If we already know a sequence from the saved config -> load it
        self.loadSequence()
        self.imageChanged()

    # Destructor
    def __del__(self):
        return

    # Construct everything GUI related. Called by constructor
    def initUI(self):
        # Create a toolbar
        self.toolbar = self.addToolBar('Tools')

        # Add the tool buttons
        iconDir = os.path.join( os.path.dirname(__file__) , 'icons' )

        # Loading a new sequence 
        loadAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'open.png' )), '&Tools', self)
        loadAction.setShortcuts(['o'])
        self.setTip( loadAction, 'Open sequence' )
        loadAction.triggered.connect( self.getSequenceFromUser )
        self.toolbar.addAction(loadAction)

        # Open previous image
        backAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'back.png')), '&Tools', self)
        backAction.setShortcut('left')
        backAction.setStatusTip('Previous image')
        backAction.triggered.connect( self.prevImage )
        self.toolbar.addAction(backAction)
        self.actImageNotFirst.append(backAction)

        # Open next image
        nextAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'next.png')), '&Tools', self)
        nextAction.setShortcut('right')
        self.setTip( nextAction, 'Next image' )
        nextAction.triggered.connect( self.nextImage )
        self.toolbar.addAction(nextAction)
        self.actImageNotLast.append(nextAction)

        # Play
        playAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'play.png')), '&Tools', self)
        playAction.setShortcut(' ')
        playAction.setCheckable(True)
        playAction.setChecked(False)
        self.setTip( playAction, 'Play all images' )
        playAction.triggered.connect( self.playImages )
        self.toolbar.addAction(playAction)
        self.actImageNotLast.append(playAction)
        self.playAction = playAction

        # Choose between semantic and instance labels
        selLabelAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'switch.png')), '&Tools', self)
        selLabelAction.setShortcut(' ')
        self.setTip( selLabelAction, 'Select label type' )
        selLabelAction.triggered.connect( self.selectLabel )
        self.toolbar.addAction(selLabelAction)
        self.actImageNotLast.append(selLabelAction)
        self.selLabelAction = selLabelAction

        # Select image
        selImageAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'shuffle.png' )), '&Tools', self)
        selImageAction.setShortcut('i')
        self.setTip( selImageAction, 'Select image' )
        selImageAction.triggered.connect( self.selectImage )
        self.toolbar.addAction(selImageAction)
        self.actImage.append(selImageAction)

        # Enable/disable zoom. Toggle button
        zoomAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'zoom.png' )), '&Tools', self)
        zoomAction.setShortcuts(['z'])
        zoomAction.setCheckable(True)
        zoomAction.setChecked(self.zoom)
        self.setTip( zoomAction, 'Enable/disable permanent zoom' )
        zoomAction.toggled.connect( self.zoomToggle )
        self.toolbar.addAction(zoomAction)
        self.actImage.append(zoomAction)

        # Decrease transparency
        minusAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'minus.png' )), '&Tools', self)
        minusAction.setShortcut('-')
        self.setTip( minusAction, 'Decrease transparency' )
        minusAction.triggered.connect( self.minus )
        self.toolbar.addAction(minusAction)

        # Increase transparency
        plusAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'plus.png' )), '&Tools', self)
        plusAction.setShortcut('+')
        self.setTip( plusAction, 'Increase transparency' )
        plusAction.triggered.connect( self.plus )
        self.toolbar.addAction(plusAction)

        # Display path to current image in message bar
        displayFilepathAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'filepath.png' )), '&Tools', self)
        displayFilepathAction.setShortcut('f')
        self.setTip( displayFilepathAction, 'Show path to current image' )
        displayFilepathAction.triggered.connect( self.displayFilepath )
        self.toolbar.addAction(displayFilepathAction)

        # Display help message
        helpAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'help19.png' )), '&Tools', self)
        helpAction.setShortcut('h')
        self.setTip( helpAction, 'Help' )
        helpAction.triggered.connect( self.displayHelpMessage )
        self.toolbar.addAction(helpAction)

        # Close the application
        exitAction = QtWidgets.QAction(QtGui.QIcon( os.path.join( iconDir , 'exit.png' )), '&Tools', self)
        exitAction.setShortcuts(['Esc'])
        self.setTip( exitAction, 'Exit' )
        exitAction.triggered.connect( self.close )
        self.toolbar.addAction(exitAction)

        # The default text for the status bar
        self.defaultStatusbar = 'Ready'
        # Create a statusbar. Init with default
        self.statusBar().showMessage( self.defaultStatusbar )

        # Enable mouse move events
        self.setMouseTracking(True)
        self.toolbar.setMouseTracking(True)
        # Set window size
        self.resize(self.screenWidth, int(self.screenWidth/1408*376 + 100))
        # Set a title
        self.applicationTitle = 'KITTI-360 Viewer v1.0'
        self.setWindowTitle(self.applicationTitle)
        self.displayHelpMessage()
        self.getSequenceFromUser()
        # And show the application
        self.show()

    #############################
    ## Toolbar call-backs
    #############################

    # Switch to previous image in file list
    # Load the image
    # Load its labels
    # Update the mouse selection
    # View
    def prevImage(self):
        if not self.images:
            return
        if self.idx > 0:
            self.idx -= 1
            self.imageChanged()
        else:
            message = "Already at the first image"
            self.statusBar().showMessage(message)
        return

    # Switch to next image in file list
    # Load the image
    # Load its labels
    # Update the mouse selection
    # View
    def nextImage(self):
        if not self.images:
            return
        if self.idx < len(self.images)-1:
            self.idx += 1
            self.imageChanged()
        elif self.playState:
            self.playState = False
            self.playAction.setChecked(False)
        else:
            message = "Already at the last image"
            self.statusBar().showMessage(message)
        if self.playState:
            QtCore.QTimer.singleShot(0, self.nextImage)
        return

    # Play images, i.e. auto-switch to next image
    def playImages(self, status):
        self.playState = status
        if self.playState:
            QtCore.QTimer.singleShot(0, self.nextImage)


    # Switch to a selected image of the file list
    # Ask the user for an image
    # Load the image
    # Load its labels
    # Update the mouse selection
    # View
    def selectImage(self):
        if not self.images:
            return

        dlgTitle = "Select image to load"
        self.statusBar().showMessage(dlgTitle)
        #items = QtCore.QStringList( [ os.path.basename(i) for i in self.images ] )
        items = [ os.path.basename(i) for i in self.images ]
        (item, ok) = QtWidgets.QInputDialog.getItem(self, dlgTitle, "Image", items, self.idx, False)
        if (ok and item):
            idx = items.index(item)
            if idx != self.idx:
                self.idx = idx
                self.imageChanged()
        else:
            # Restore the message
            self.statusBar().showMessage( self.defaultStatusbar )

    # Switch between instance and semantic visualization
    def selectLabel(self):
        if self.gtType == "instance":
            self.gtType = "semantic"
        else:
            self.gtType = "instance"
        self.update()


    # Toggle zoom
    def zoomToggle(self, status):
        self.zoom = status
        if status :
            self.mousePosOnZoom = self.mousePosOrig
        self.update()

    # Toggle disparity visu
    def dispToggle(self, status):
        self.showDisparity = status
        self.imageChanged()


    # Increase label transparency
    def minus(self):
        self.transp = max(self.transp-0.1,0.0)
        self.update()


    def displayFilepath(self):
        self.statusBar().showMessage("Current image: {0}".format( self.currentFile ))
        self.update()

    def displayHelpMessage(self):

        message = self.applicationTitle + "\n\n"
        message += "INSTRUCTIONS\n"
        message += " - select a sequence from drop-down menu\n"
        message += " - browse images and labels using\n"
        message += "   the toolbar buttons or the controls below\n"
        message += "\n"
        message += "CONTROLS\n"
        message += " - select sequence [o]\n"
        message += " - highlight objects [move mouse]\n"
        message += " - select instance object [left-click mouse]\n"
        message += " - release selected object [right-click mouse]\n"
        message += " - next image [left arrow]\n"
        message += " - previous image [right arrow]\n"
        message += " - toggle autoplay [space]\n"
        message += " - increase/decrease label transparency\n"
        message += "   [ctrl+mousewheel] or [+ / -]\n"

        message += " - open zoom window [z]\n"
        message += "       zoom in/out [mousewheel]\n"
        message += "       enlarge/shrink zoom window [shift+mousewheel]\n"
        message += " - select a specific image [i]\n"
        message += " - show path to image below [f]\n"
        message += " - exit viewer [esc]\n"

        QtWidgets.QMessageBox.about(self, "HELP!", message)
        self.update()

    def displaySelectHelpMessage(self):
        self.statusBar().showMessage("Use left click to select an instance".format( self.highlightObjLabel ))
        self.update()

    def displaySelectedInstance(self):
        self.statusBar().showMessage("Selected instance '{0}', use right click for deselection".format( self.highlightObjLabel ))
        self.update()

    # Decrease label transparency
    def plus(self):
        self.transp = min(self.transp+0.1,1.0)
        self.update()

    # Close the application
    def closeEvent(self,event):
         event.accept()


    #############################
    ## Custom events
    #############################

    def imageChanged(self):
        # Load the first image
        self.loadImage()
        # Load its labels if available
        self.loadLabels()
        # Update the object the mouse points to
        self.updateMouseObject()
        # Update the GUI
        self.update()

    #############################
    ## File I/O
    #############################

    # Load the currently selected sequence if possible
    def loadSequence(self):
        # Search for all *.pngs to get the image list
        self.images = []
        if os.path.isdir(self.sequence):
            # Get images contain specific obj
            if self.mousePressObj>=0:
                basenames = self.annotation2DInstance(self.mouseSemanticId, self.mouseInstanceId)
                self.images = [os.path.join(self.sequence, bn) for bn in basenames]
                self.images = [fn for fn in self.images if os.path.isfile(fn)]
                self.images.sort()

            # Get all images if no obj is selected
            else:
                if not self.imagesSequence:
                    self.images = glob.glob( os.path.join( self.sequence , '*' + self.imageExt ) )
                    self.images.sort()
                    # filter out images without labels
                    self.label_images = glob.glob(os.path.join(self.labelPath, "instance", "*.png"))
                    basenames = [os.path.basename(lb) for lb in self.label_images]
                    self.images = [fn for fn in self.images if os.path.basename(fn) in basenames]
                    self.imagesSequence = self.images
                else:
                    self.images = self.imagesSequence

                print("Loaded %d images" % len(self.images))
            
            if self.currentFile in self.images:
                self.idx = self.images.index(self.currentFile)
            else:
                self.idx = 0


    # Load the currently selected image
    # Does only load if not previously loaded
    # Does not refresh the GUI
    def loadImage(self):
        success = False
        message = self.defaultStatusbar
        if self.images:
            filename = self.images[self.idx]
            filename = os.path.normpath( filename )
            if not self.image.isNull() and filename == self.currentFile:
                success = True
            else:
                self.image = QtGui.QImage(filename)
                if self.image.isNull():
                    message = "Failed to read image: {0}".format( filename )
                else:
                    message = "Read image: {0}".format( filename )
                    self.currentFile = filename
                    success = True

        # Update toolbar actions that need an image
        for act in self.actImage:
            act.setEnabled(success)
        for act in self.actImageNotFirst:
            act.setEnabled(success and self.idx > 0)
        for act in self.actImageNotLast:
            act.setEnabled(success and self.idx < len(self.images)-1)

        self.statusBar().showMessage(message)

    # Load the labels from file
    # Only loads if they exist
    # Otherwise the filename is stored and that's it
    def loadLabels(self):
        filename = self.getLabelFilename()
        if not filename:
            self.clearAnnotation()
            return

        # If we have everything and the filename did not change, then we are good
        if self.annotation2D and filename == self.currentLabelFile:
            return

        # Clear the current labels first
        self.clearAnnotation()

        try:
            self.annotation2D = Annotation2D()
            self.annotation2D.loadInstance(filename, self.gtType)

        except IOError as e:
            # This is the error if the file does not exist
            message = "Error parsing labels in {0}. Message: {1}".format( filename, e.strerror )
            self.statusBar().showMessage(message)

        # Remember the filename loaded
        self.currentLabelFile = filename

        # Remeber the status bar message to restore it later
        restoreMessage = self.statusBar().currentMessage()

        # Restore the message
        self.statusBar().showMessage( restoreMessage )


    #############################
    ## Drawing
    #############################

    # This method is called when redrawing everything
    # Can be manually triggered by self.update()
    # Note that there must not be any other self.update within this method
    # or any methods that are called within
    def paintEvent(self, event):
        # Create a QPainter that can perform draw actions within a widget or image
        qp = QtGui.QPainter()
        # Begin drawing in the application widget
        qp.begin(self)
        # Update scale
        self.updateScale(qp)
        # Determine the object ID to highlight
        self.getHighlightedObject(qp)
        # Draw the image first
        self.drawImage(qp)

        if self.enableDisparity and self.showDisparity:
            # Draw the disparities on top
            overlay = self.drawDisp(qp)
        else:
            # Draw the labels on top
            overlay = self.drawLabels(qp)
            # Draw the label name next to the mouse
            self.drawLabelAtMouse(qp)

        # Draw the zoom
        self.drawZoom(qp, overlay)

        # Thats all drawing
        qp.end()

        # Forward the paint event
        QtWidgets.QMainWindow.paintEvent(self,event)

    # Update the scaling
    def updateScale(self, qp):
        if not self.image.width() or not self.image.height():
            return
        # Horizontal offset
        self.xoff  = self.bordergap
        # Vertical offset
        self.yoff  = self.toolbar.height()+self.bordergap
        # We want to make sure to keep the image aspect ratio and to make it fit within the widget
        # Without keeping the aspect ratio, each side of the image is scaled (multiplied) with
        sx = float(qp.device().width()  - 2*self.xoff) / self.image.width()
        sy = float(qp.device().height() - 2*self.yoff) / self.image.height()
        # To keep the aspect ratio while making sure it fits, we use the minimum of both scales
        # Remember the scale for later
        self.scale = min( sx , sy )
        # These are then the actual dimensions used
        self.w     = self.scale * self.image.width()
        self.h     = self.scale * self.image.height()

    # Determine the highlighted object for drawing
    def getHighlightedObject(self, qp):
        # This variable we want to fill
        #self.highlightObj = None

        # Without labels we cannot do so
        if not self.annotation2D:
            return

        # If available its the selected object
        highlightObjId = -1
        # If not available but the polygon is empty or closed, its the mouse object
        if highlightObjId < 0 and not self.mouseOutsideImage:
            highlightObjId = self.mouseObj
        

        ## Get the semantic and instance id of the object that is highlighted
        if highlightObjId==0: 
            self.highlightObjLabel = '%s' % (id2label[self.mouseSemanticId].name)
        else:
            self.highlightObjLabel = '%s,%d' % (id2label[self.mouseSemanticId].name, highlightObjId)

    # Draw the image in the given QPainter qp
    def drawImage(self, qp):
        # Return if no image available
        if self.image.isNull():
            return

        # Save the painters current setting to a stack
        qp.save()
        # Draw the image
        qp.drawImage(QtCore.QRect( self.xoff, self.yoff, self.w, self.h ), self.image)
        # Restore the saved setting from the stack
        qp.restore()

    # Draw the projected 3D bounding boxes
    def getLines(self, obj):
        lines = []
        for line in obj.lines:
            if obj.vertices_depth[line[0]]<0 and obj.vertices_depth[line[1]]<0:
                continue
            elif obj.vertices_depth[line[0]]<0 or obj.vertices_depth[line[1]]<0:
                if self.currentFile:
                    frame = int(os.path.splitext(os.path.basename( self.currentFile ))[0])
                    v = [obj.vertices[line[0]]*x + obj.vertices[line[1]]*(1-x) for x in np.arange(0,1,0.1)]
                    uv, d = self.camera.project_vertices(np.asarray(v), frame)
                    d[d<0] = 1e+6
                    vidx = line[0] if obj.vertices_depth[line[0]] < 0 else line[1]
                    obj.vertices_proj[0][vidx] = uv[0][np.argmin(d)]
                    obj.vertices_proj[1][vidx] = uv[1][np.argmin(d)]
                else:
                    continue

            lines.append( QtCore.QLineF(obj.vertices_proj[0][line[0]],
                                  obj.vertices_proj[1][line[0]],
                                  obj.vertices_proj[0][line[1]],
                                  obj.vertices_proj[1][line[1]] ) )
        return lines

    # Load the semantic/instance image
    def getLabelImg(self):
        if self.image.isNull() or self.w == 0 or self.h == 0:
            return
        if not self.annotation2D:
            return

        if self.gtType == 'instance':
            if self.annotation2D.instanceImg is None:
                self.annotation2D.loadInstance(self.getLabelFilename(), self.gtType)
            overlay = self.getQImage(self.annotation2D.instanceImg)
        elif self.gtType == 'semantic':
            if self.annotation2D.semanticImg is None:
                self.annotation2D.loadInstance(self.getLabelFilename(), self.gtType)
            overlay = self.getQImage(self.annotation2D.semanticImg)
        return overlay
        

    # Draw the labels in the given QPainter qp
    # optionally provide a list of labels to ignore
    def drawLabels(self, qp, ignore = []):
        if self.image.isNull() or self.w == 0 or self.h == 0:
            return
        if not self.annotation2D:
            return

        overlay = self.getLabelImg()

        # Create a new QPainter that draws in the overlay image
        qp2 = QtGui.QPainter()
        qp2.begin(overlay)

        # The color of the outlines
        qp2.setPen(QtGui.QColor('white'))

        if self.highlightObj:
            lines = self.getLines(self.highlightObj)
            name = self.highlightObj.name

            # Default drawing
            # Color from color table, solid brush
            col   = QtGui.QColor( *name2label[name].color     )
            brush = QtGui.QBrush( col, QtCore.Qt.SolidPattern )
            qp2.setBrush(brush)
            for line in lines:
                qp2.drawLine(line)


        if self.highlightObjSparse:
            for pts in self.highlightObjSparse:
                qp2.drawPoint(pts[0], pts[1])

        # End the drawing of the overlay
        qp2.end()
        # Save QPainter settings to stack
        qp.save()
        # Define transparency
        qp.setOpacity(self.transp)
        # Draw the overlay image
        #overlay = self.getLabelImg()
        qp.drawImage(QtCore.QRect( self.xoff, self.yoff, self.w, self.h ), overlay)
        # Restore settings
        qp.restore()

        return overlay 


    # Draw the label name next to the mouse
    def drawLabelAtMouse(self, qp):
        # Nothing to do with mouse outside of image 
        if self.mouseOutsideImage:
            return
        # Nothing to without a mouse position
        if not self.mousePosOrig:
            return

        # Save QPainter settings to stack
        qp.save()

        # That is the mouse positiong
        mouse = self.mousePosOrig

        # Will show zoom
        showZoom = self.zoom and not self.image.isNull() and self.w and self.h

        # The text that is written next to the mouse
        #mouseText = self.highlightObj.label
        mouseText = self.highlightObjLabel

        # Where to write the text
        # Depends on the zoom (additional offset to mouse to make space for zoom?)
        # The location in the image (if we are at the top we want to write below of the mouse)
        off = 36
        if showZoom:
            off += self.zoomSize/2
        if mouse.y()-off > self.toolbar.height():
            top = mouse.y()-off
            btm = mouse.y()
            vAlign = QtCore.Qt.AlignTop
        else:
            # The height of the cursor
            if not showZoom:
                off += 20
            top = mouse.y()
            btm = mouse.y()+off
            vAlign = QtCore.Qt.AlignBottom

        # Here we can draw
        rect = QtCore.QRect()
        rect.setTopLeft(QtCore.QPoint(mouse.x()-200,top))
        rect.setBottomRight(QtCore.QPoint(mouse.x()+200,btm))

        # The color
        qp.setPen(QtGui.QColor('white'))
        # The font to use
        font = QtGui.QFont("Helvetica",20,QtGui.QFont.Bold)
        qp.setFont(font)
        # Non-transparent
        qp.setOpacity(1)
        # Draw the text, horizontally centered
        qp.drawText(rect,QtCore.Qt.AlignHCenter|vAlign,mouseText)
        # Restore settings
        qp.restore()

    # Draw the zoom
    def drawZoom(self,qp,overlay):
        # Zoom disabled?
        if not self.zoom:
            return
        # No image
        if self.image.isNull() or not self.w or not self.h:
            return
        # No mouse
        if not self.mousePosOrig:
            return

        # Abbrevation for the zoom window size
        zoomSize = self.zoomSize
        # Abbrevation for the mouse position
        mouse = self.mousePosOrig

        # The pixel that is the zoom center
        pix = self.mousePosScaled
        # The size of the part of the image that is drawn in the zoom window
        selSize = zoomSize / ( self.zoomFactor * self.zoomFactor )
        # The selection window for the image
        sel  = QtCore.QRectF(pix.x()  -selSize/2 ,pix.y()  -selSize/2 ,selSize,selSize  )
        # The selection window for the widget
        view = QtCore.QRectF(mouse.x()-zoomSize/2,mouse.y()-zoomSize/2,zoomSize,zoomSize)
        if overlay :
            overlay_scaled = overlay.scaled(self.image.width(), self.image.height())
        else :
            overlay_scaled = QtGui.QImage( self.image.width(), self.image.height(), QtGui.QImage.Format_ARGB32_Premultiplied )

        # Show the zoom image
        qp.save()
        qp.drawImage(view,self.image,sel)
        qp.setOpacity(self.transp)
        qp.drawImage(view,overlay_scaled,sel)
        qp.restore()


    #############################
    ## Mouse/keyboard events
    #############################

    # Mouse moved
    # Need to save the mouse position
    # Need to drag a polygon point
    # Need to update the mouse selected object
    def mouseMoveEvent(self,event):


        if self.image.isNull() or self.w == 0 or self.h == 0:
            return

        mousePosOrig = QtCore.QPointF( event.x() , event.y() )
        mousePosScaled = QtCore.QPointF( float(mousePosOrig.x() - self.xoff) / self.scale , float(mousePosOrig.y() - self.yoff) / self.scale )
        mouseOutsideImage = not self.image.rect().contains( mousePosScaled.toPoint() )

        mousePosScaled.setX( max( mousePosScaled.x() , 0. ) )
        mousePosScaled.setY( max( mousePosScaled.y() , 0. ) )
        mousePosScaled.setX( min( mousePosScaled.x() , self.image.rect().right() ) )
        mousePosScaled.setY( min( mousePosScaled.y() , self.image.rect().bottom() ) )

        if not self.image.rect().contains( mousePosScaled.toPoint() ):
            print(self.image.rect())
            print(mousePosScaled.toPoint())
            self.mousePosScaled = None
            self.mousePosOrig = None
            self.updateMouseObject()
            self.update()
            return

        self.mousePosScaled    = mousePosScaled
        self.mousePosOrig      = mousePosOrig
        self.mouseOutsideImage = mouseOutsideImage

        # Redraw
        self.updateMouseObject()
        self.update()

    # Mouse Pressed
    # Left button to select one instance
    # Right button to release the selected instance
    def mousePressEvent(self,event):
        if event.button() == QtCore.Qt.RightButton:
            self.mousePressObj = -1
            self.loadSequence()
            # Restore the message
            self.statusBar().showMessage( self.defaultStatusbar )
        elif event.button() == QtCore.Qt.LeftButton and self.mouseObj<=0:
            return
        else:
            self.mousePressObj = self.mouseObj
            self.loadSequence()
            self.displaySelectedInstance()

    # Mouse left the widget
    def leaveEvent(self, event):
        self.mousePosOrig = None
        self.mousePosScaled = None
        self.mouseOutsideImage = True


    # Mouse wheel scrolled
    def wheelEvent(self, event):
        ctrlPressed = event.modifiers() & QtCore.Qt.ControlModifier

        deltaDegree = event.angleDelta() / 8 # Rotation in degree
        deltaSteps  = deltaDegree / 15 # Usually one step on the mouse is 15 degrees

        if ctrlPressed:
            self.transp = max(min(self.transp+(deltaSteps*0.1),1.0),0.0)
            self.update()
        else:
            if self.zoom:
                # If shift is pressed, change zoom window size
                if event.modifiers() and QtCore.Qt.Key_Shift:
                    self.zoomSize += deltaSteps * 10
                    self.zoomSize = max( self.zoomSize, 10   )
                    self.zoomSize = min( self.zoomSize, 1000 )
                # Change zoom factor
                else:
                    self.zoomFactor += deltaSteps * 0.05
                    self.zoomFactor = max( self.zoomFactor, 0.1 )
                    self.zoomFactor = min( self.zoomFactor, 10 )
                self.update()


    #############################
    ## Little helper methods
    #############################

    # Helper method that sets tooltip and statustip
    # Provide an QAction and the tip text
    # This text is appended with a hotkeys and then assigned
    def setTip( self, action, tip ):
        tip += " (Hotkeys: '" + "', '".join([str(s.toString()) for s in action.shortcuts()]) + "')"
        action.setStatusTip(tip)
        action.setToolTip(tip)

    # Update the object that is selected by the current mouse curser
    def updateMouseObject(self):

        if self.mousePosScaled is None:
            return

        if self.annotation2D is None:
            return

        # get current frame
        filename = os.path.basename( self.currentFile )
        frame = int(os.path.splitext(filename)[0])
        
        # Update according to mouse obj only when mouse is not pressed
        if self.mousePressObj == -1:
            self.mouseObj   = -1
            pix = self.mousePosScaled.toPoint()
            self.mouseObj = self.annotation2D.instanceId[pix.y(), pix.x()]
            self.mouseInstanceId = self.annotation2D.instanceId[pix.y(), pix.x()]
            self.mouseSemanticId = self.annotation2D.semanticId[pix.y(), pix.x()]

        # get 3D annotation
        obj = self.annotation3D(self.mouseSemanticId, self.mouseInstanceId, frame)
        # with instances
        if obj:
            # project to the current frame
            self.camera(obj, frame)

            self.highlightObj = obj

            # load sparse 3D points
            objSparse = None
            if self.showSparse:
                objSparse = self.annotationSparse3D(frame, self.mouseSemanticId, self.mouseInstanceId)
            
            # get 3D sparse points
            if objSparse:
                self.highlightObjSparse = objSparse
            else:
                self.highlightObjSparse = None
        # without instances
        else:
            self.highlightObj = None
            self.highlightObjSparse = None

        return 

    # Clear the current labels
    def clearAnnotation(self):
        self.annotation2D = None
        self.currentLabelFile = ""

    def getSequenceFromUser(self):
        # Reset the status bar to this message when leaving
        restoreMessage = self.statusBar().currentMessage()

        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')

        imagePath = os.path.join(kitti360Path, 'data_2d_raw')
        label2DPath = os.path.join(kitti360Path, 'data_2d_semantics', 'train')
        label3DBboxPath = os.path.join(kitti360Path, 'data_3d_bboxes')

        #kittiGtPath = os.path.join(kittiGtRoot, 'segmentation/dynamic/instance_sequence')
        availableSequences = [seq for seq in os.listdir(label2DPath) if os.path.isdir(os.path.join(label2DPath, seq))]
        availableSequences = sorted(availableSequences)

        # List of possible labels
        items = availableSequences

        # Specify title
        dlgTitle = "Select new sequence"
        message = dlgTitle
        question = dlgTitle
        message = "Select sequence for viewing"
        question = "Which sequence would you like to view?"
        self.statusBar().showMessage(message)

        if items:
            # Create and wait for dialog
            (item, ok) = QtWidgets.QInputDialog.getItem(self, dlgTitle, question,
                                                    items, 0, False)

            # Restore message
            self.statusBar().showMessage(restoreMessage)

            if ok and item:
                sequence = item
                self.currentSequence = sequence
                self.sequence = os.path.normpath(os.path.join(imagePath, sequence, "image_%02d" % self.cameraId, "data_rect"))
                self.labelPath = os.path.normpath(os.path.join(label2DPath, sequence))
                self.annotation2DInstance = Annotation2DInstance(os.path.join(label2DPath, sequence))
                self.annotation3D = Annotation3D(label3DBboxPath, sequence)

                self.camera = Camera(root_dir=kitti360Path, seq=sequence)
                self.imagesSequence = []
                self.loadSequence()
                self.imageChanged()

        else:

            warning = ""
            warning += "The data was not found. Please:\n\n"
            warning += " - make sure the scripts folder is in the KITTI-360 root folder\n"
            warning += "or\n"
            warning += " - set KITTI360_DATASET to the KITTI-360 root folder\n"
            warning += "       e.g. 'export KITTI360_DATASET=<root_path>'\n"

            reply = QtGui.QMessageBox.information(self, "ERROR!", warning,
                                                  QtGui.QMessageBox.Ok)
            if reply == QtGui.QMessageBox.Ok:
                sys.exit()

        return

    # Determine if the given candidate for a label path makes sense
    def isLabelPathValid(self, labelPath):
        return os.path.isdir(labelPath)

    # Get the filename where to load labels
    # Returns empty string if not possible
    def getLabelFilename(self, currentFile=None):
        if not currentFile:
            currentFile = self.currentFile

        # And we need to have a directory where labels should be searched
        if not self.labelPath:
            return ""
        # Without the name of the current images, there is also nothing we can do
        if not currentFile:
            return ""
        # Check if the label directory is valid.
        if not self.isLabelPathValid(self.labelPath):
            return ""

        # Generate the filename of the label file
        filename = os.path.basename(currentFile)
        search = [lb for lb in self.label_images if filename in lb]
        # #filename = filename.replace(self.imageExt, self.gtExt)
        # filename = os.path.join(self.labelPath, self.currentSequence + "*", filename)
        # search   = glob.glob(filename)
        if not search:
            return ""
        filename = os.path.normpath(search[0])
        return filename

    # Disable the popup menu on right click
    def createPopupMenu(self):
        pass


    @staticmethod
    def getQImage(image):
        assert (np.max(image) <= 255)
        image8 = image.astype(np.uint8)
        height, width, colors = image8.shape
        bytesPerLine = 3 * width
    
        image = QtGui.QImage(image8.data, width, height, bytesPerLine,
                           QtGui.QImage.Format_RGB888)
    
        return image

def main():

    app = QtWidgets.QApplication(sys.argv)
    screenRes = app.desktop().screenGeometry()
    tool = Kitti360Viewer(screenRes.width())
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
