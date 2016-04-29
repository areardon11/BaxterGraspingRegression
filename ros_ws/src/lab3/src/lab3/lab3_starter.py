import numpy as np
import obj_file
import transformations

SPRAY_BOTTLE_MESH_FILENAME = 'data/spray.obj'

def contacts_to_baxter_hand_pose(contact1, contact2):
    c1 = np.array(contact1)
    c2 = np.array(contact2)

    # compute gripper center and axis
    center = 0.5 * (c1 + c2)
    y_axis = c2 - c1
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.array([y_axis[1], -y_axis[0], 0]) # the z axis will always be in the table plane for now
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)

    # convert to hand pose
    R_obj_gripper = np.array([x_axis, y_axis, z_axis]).T
    t_obj_gripper = center
    T_obj_gripper = np.eye(4)
    T_obj_gripper[:3,:3] = R_obj_gripper
    T_obj_gripper[:3,3] = t_obj_gripper
    q_obj_gripper = transformations.quaternion_from_matrix(T_obj_gripper)

    return t_obj_gripper, q_obj_gripper 

if __name__ == '__main__':
    of = obj_file.ObjFile(SPRAY_BOTTLE_MESH_FILENAME)
    mesh = of.read()

    vertices = mesh.vertices
    triangles = mesh.triangles
    normals = mesh.normals

    print 'Num vertices:', len(vertices)
    print 'Num triangles:', len(triangles)
    print 'Num normals:', len(normals)

    # 1. Generate candidate pairs of contact points

    # 2. Check for force closure

    # 3. Convert each grasp to a hand pose
    contact1 = vertices[0]
    contact2 = vertices[100]
    t_obj_gripper, q_obj_gripper = contacts_to_baxter_hand_pose(contact1, contact2)
    print 'Translation', t_obj_gripper
    print 'Rotation', q_obj_gripper

    # 4. Execute on the actual robot
