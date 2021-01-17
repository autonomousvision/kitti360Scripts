#include "nanoflann.hpp"
#include <iostream>
#include <vector>

using namespace std;
using namespace nanoflann;

template <class T>
struct Point
{
	T  x,y,z;
	Point(){}
	Point(T mx, T my, T mz) {
		x = mx; y = my; z = mz;
	}
};

template <class T>
struct PointCloud
{
	std::vector<Point<T> >  pts;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
	inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t /*size*/) const
	{
		const T d0=p1[0]-pts[idx_p2].x;
		const T d1=p1[1]-pts[idx_p2].y;
		const T d2=p1[2]-pts[idx_p2].z;
		return d0*d0+d1*d1+d2*d2;
	}

	// TODO

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline T kdtree_get_pt(const size_t idx, int dim) const
	{
		if (dim==0) return pts[idx].x;
		else if (dim==1) return pts[idx].y;
		else return pts[idx].z;
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

	/*PointCloud<T>& operator=(const PointCloud<T> &pcd)
	{
		for (int i = 0; i < pcd.pts.size(); i++) {
			this->pts.push_back(pcd.pts[i]);
		}
		return *this;
	}*/

};

template <class T>
class KDtree
{

	PointCloud<T> cloud;

public:
	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<T, PointCloud<T> > ,
		PointCloud<T>,
		3 /* dim */
		> 
		kd_tree;


	kd_tree *tree;

	KDtree() {
		tree = new kd_tree(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(1000 /* max leaf */) );
		tree->buildIndex();
	}

	KDtree(PointCloud<T> &pcd) {
		this->cloud = pcd;
		tree = new kd_tree(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(1000 /* max leaf */) );
		tree->buildIndex();
	}

	~KDtree() {
		delete tree;
	}

	void knnSearch(const Point<T> query, int k, std::vector<size_t> &indices, std::vector<T> &dist)
	{
		T q[3] = {query.x, query.y, query.z};
		indices.resize(k);
		dist.resize(k);
		tree->knnSearch(&q[0], k, &indices[0], &dist[0]);
	}

	size_t radiusSearch(const Point<T> query, T r, std::vector<size_t> &indices, std::vector<T> &dist)
	{
		std::vector<std::pair<size_t, T> > ret_matches;
		nanoflann::SearchParams params;
		//params.sorted = false;

		T q[3] = {query.x, query.y, query.z};
		size_t nMatches = tree->radiusSearch(&q[0], r, ret_matches, params);
		
		for (size_t i = 0; i < ret_matches.size(); i++){
			indices.push_back(ret_matches[i].first);
			dist.push_back(ret_matches[i].second);
		}
		return nMatches;
	}
};
