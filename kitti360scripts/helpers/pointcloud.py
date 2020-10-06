class PointCloud:

    def __init__(self, ptsDir, camera=0):
        self.ptsDir = ptsDir
        self.camera = camera 

        self.w = 1408
        self.h = 376

    def loadPointCloud(self, frame):

	labelDir = '%s/annotation_%010d_%d_m.dat' % (self.ptsDir, frame, self.camera)
	print 'Processing %010d' %(f)

	if not (os.path.isfile(labelDir)):
                print '    annotation file doesnt exist'
		continue
	img = Image.new('RGB', [w, h])
	imgDL = Image.new('RGB', [w, h])

	labels = tuple(open(labelDir, 'r'))
	unarySparse = []
	sparseLabel2D = []
	
	offset = 6
	tupleNum = 3
	for l in labels:
		s = ([float(n) for n in l.split()])
		if (s[offset] != 0):
			yy = int(math.floor(s[0]/w))
			xx = int(s[0] % w)
			numCandidate = (len(s)-offset)/tupleNum
			# only feed for training when there is only one point
			candidate = []
			for i in range(numCandidate):
                            # skip the undefined label
                            if int(s[i*tupleNum+1+offset])-1 != priority.shape[0]: 
				candidate.append(int(s[i*tupleNum+1+offset])-1)
                        if not candidate:
                            continue
			index = random.sample(candidate, 1)
                        if numCandidate>1:
                                subprior = priority[np.ix_(candidate, candidate)]
                                if np.std(subprior.flatten()) != 0:
                                        midx = np.argmax(np.sum(subprior, axis=1) )
                                        index[0] = candidate[midx]
			img.putpixel((xx, yy), (colorMap[index[0]+1][0], colorMap[index[0]+1][1], colorMap[index[0]+1][2]))
                        # take it for training only when there is no ambuity 
                        if np.std(candidate)==0:
			    imgDL.putpixel((xx, yy), (colorMap[candidate[0]+1][0], colorMap[candidate[0]+1][1], colorMap[candidate[0]+1][2]))


if __name__=='__main__':
    pcd = PointCloud('')
