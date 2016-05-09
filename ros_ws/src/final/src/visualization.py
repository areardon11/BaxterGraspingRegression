import numpy as np
import vispy.scene
from vispy.scene import visuals
# from point_cloud_sampler import determine_grasp_points
import pickle
import sys
from IPython import embed

# pc = np.asarray(np.load('boxed'))[:,:3]
# pos = pc[~np.isnan(pc).any(axis=1)]
# print pc.shape
# print pos.shape

def view_pc(pc):
    #
    # Make a canvas and add simple view
    #
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    scatter = visuals.Markers()
    scatter.set_data(pc, edge_color=None, face_color=(1, 1, 1, .5), size=5)
    view.add(scatter)

    view.camera = 'turntable'

    axis = visuals.XYZAxis(parent=view.scene)

    if sys.flags.interactive != 1:
        vispy.app.run()


def view_contacts(pc, contacts):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    scatter = visuals.Markers()
    scatter.set_data(pc, edge_color=None, face_color=(1, 1, 1, .5), size=5)
    view.add(scatter)

    scatter2 = visuals.Markers()
    scatter2.set_data(contacts, edge_color=None, face_color=(0, 1, 0, 1), size=20)
    view.add(scatter2)

    view.camera = 'turntable'

    axis = visuals.XYZAxis(parent=view.scene)

    if sys.flags.interactive != 1:
        vispy.app.run()

# def test_pc():
#     view_pc(pos)

# def test_contacts():
#     p = determine_grasp_points(pos)
#     try:
#         p = p.reshape(2,3)
#     except AttributeError as e:
#         embed()
#     view_contacts(pos, p)

# test_pc()
