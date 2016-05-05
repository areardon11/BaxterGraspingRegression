#!/usr/bin/env python
import numpy as np
import pickle
from tf import transformations
from IPython import embed
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs import msg
import rospy

def process_points(f, num_points=10000):
    """
    Opens point cloud data file, gets rid of
    nan's, and returns [num_points] points
    """
    with open(f, 'rb') as f:
        data = pickle.load(f)
    data = np.vstack(data)
    data = data[:,:3] #Throw away the last number, let's ask Jeff about this
 
    #Get rid of nan's   
    nan = np.isnan(data)
    idx = np.where(nan != True)[0]
    idx = np.unique(idx)
    data = data[idx]
    
    if num_points == -1:
        return data
    #Subsample points
    idx = np.random.randint(0, data.shape[0], num_points)
    return data[idx]
 
def transform_points(points, tranform):
    """
    Transforms points from kinect frame
    to Baxter's base frame
    """
    t_points = []
    trans, rot = tranform
    G = transformations.quaternion_matrix(rot)
    G[:3,3] += trans
    for p in points:
        p = np.append(p, 1) 
        t_points.append(G.dot(p)[:3])
    return np.vstack(t_points)   

def plot(data, c, ax):
    ax.scatter(data[:,0], data[:,1], data[:,2], c=c, marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

if __name__ == "__main__":
    rospy.init_node('pc_publisher')
    pcl_pub = rospy.Publisher("/point_clouds", PointCloud2, queue_size=10)

    with open('tf_dict', 'rb') as f:
        tf_dict = pickle.load(f)
        
    with open('kinect1_pc2_read', 'rb') as f:
        k1 = pickle.load(f)
        
    with open('kinect2_pc2_read', 'rb') as f:
        k2 = pickle.load(f)
            
    k1_base = tf_dict['kinect1_base']
    k2_base = tf_dict['kinect2_base']
    
    print("Processing Kinect 1 Point Cloud ...")
    p1 = process_points('kinect1_pc2_read', -1)
    tp1 = transform_points(p1, k1_base).tolist()
    print("Processing Kinect 2 Point Cloud ...")
    p2 = process_points('kinect2_pc2_read', -1)
    tp2 = transform_points(p2, k2_base).tolist()
   
    print("Contatenating lists")
    points = tp1 + tp2
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        header = msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base' 
        point_cloud = pc2.create_cloud_xyz32(header, points)
        pcl_pub.publish(point_cloud)
        rate.sleep()
