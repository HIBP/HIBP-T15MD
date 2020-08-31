from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as ax3d
from copy import deepcopy
import numpy as np
import time
try:
    import visvis as vv
    visvis_avail = True
except ImportError:
    visvis_avail = False
    print("visvis not found.")

# Define a class for wires with current-----------------------------------------
#-------------------------------------------------------------------------------
class Wire:
    '''
    represents an arbitrary 3D wire geometry
    '''
    def __init__(self, current=1, path=None, discretization_length=0.01):
        '''

        :param current: electrical current in Ampere used for field calculations
        :param path: geometry of the wire specified as path of n 3D (x,y,z) 
                     points in a numpy array with dimension n x 3
                     length unit is meter
        :param discretization_length: lenght of dL after discretization
        '''
        self.current = current
        self.path = path
        self.discretization_length = discretization_length


    @property
    def discretized_path(self):
        '''
        calculate end points of segments of discretized path
        approximate discretization lenghth is given by self.discretization_length
        elements will never be combined
        elements longer that self.dicretization_length will be divided into pieces
        :return: discretized path as m x 3 numpy array
        '''

        try:
            return self.dpath
        except AttributeError:
            pass

        self.dpath = deepcopy(self.path)
        for c in range(len(self.dpath)-2, -1, -1):
            # go backwards through all elements
            # length of element
            element = self.dpath[c+1]-self.dpath[c]
            el_len = np.linalg.norm(element)
            # number of parts that this element should be split up into
            npts = int(np.ceil(el_len / self.discretization_length)) 
            if npts > 1:
                # element too long -> create points between
                # length of new sub elements
                sel = el_len / float(npts)
                for d in range(npts-1, 0, -1):
                    self.dpath = np.insert(self.dpath, c+1, self.dpath[c] + \
                                           element / el_len * sel * d, axis=0)

        return self.dpath

    @property
    def IdL_r1(self):
        '''
        calculate discretized path elements dL and their center point r1
        :return: numpy array with I * dL vectors, numpy array of r1 vectors 
        (center point of element dL)
        '''
        npts = len(self.discretized_path)
        if npts < 2:
            print("discretized path must have at least two points")
            return

        IdL = np.array([self.discretized_path[c+1] - self.discretized_path[c] 
                        for c in range(npts-1)]) * self.current
        r1 = np.array([(self.discretized_path[c+1] + \
                        self.discretized_path[c])*0.5 for c in range(npts-1)])

        return IdL, r1


    def vv_plot_path(self, discretized=True, color='r'):
        if not visvis_avail:
            print("plot path works only with visvis module")
            return

        if discretized:
            p = self.discretized_path
        else:
            p = self.path

        vv.plot(p, ms='x', mc=color, mw='2', ls='-', mew=0)


    def mpl3d_plot_path(self, discretized=True, show=True, ax=None, plt_style='-r'):

        if ax is None:
            fig = plt.figure(None)
            ax = ax3d.Axes3D(fig)
            
        if discretized:
            p = self.discretized_path
        else:
            p = self.path    

        ax.plot(p[:, 0], p[:, 1], p[:, 2], plt_style)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if show:
            plt.show()

        return ax

    def extend_path(self, path):
        '''
        extends existing path by another one
        :param path: path to append
        '''
        if self.path is None:
            self.path = path
        else:
            # check if last point is identical to avoid zero length segments
            if np.all(self.path[-1] == path[0]):
                self.path=np.append(self.path, path[1:], axis=0)
            else:
                self.path=np.append(self.path, path, axis=0)

    def translate(self, xyz):
        '''
        move the wire in space
        :param xyz: 3 component vector that describes translation in x,y and z direction
        '''
        if self.path is not None:
            self.path += np.array(xyz)

        return self

    def rotate(self, axis=(1,0,0), deg=0):
        '''
        rotate wire around given axis by deg degrees
        :param axis: axis of rotation
        :param deg: angle
        '''
        if self.path is not None:
            n = axis
            ca = np.cos(np.radians(deg))
            sa = np.sin(np.radians(deg))
            R = np.array([[n[0]**2*(1-ca)+ca, n[0]*n[1]*(1-ca)-n[2]*sa,
                           n[0]*n[2]*(1-ca)+n[1]*sa],
                          [n[1]*n[0]*(1-ca)+n[2]*sa, n[1]**2*(1-ca)+ca,
                           n[1]*n[2]*(1-ca)-n[0]*sa],
                          [n[2]*n[0]*(1-ca)-n[1]*sa, n[2]*n[1]*(1-ca)+n[0]*sa,
                           n[2]**2*(1-ca)+ca]])
            self.path = np.dot(self.path, R.T)

        return self

    # different standard paths
    @staticmethod
    def linear_path(pt1=(0, 0, 0), pt2=(0, 0, 1)):
        return np.array([pt1, pt2]).T

    @staticmethod
    def rectangular_path(dx=0.1, dy=0.2):
        dx2 = dx/2.0; dy2 = dy/2.0
        return np.array([[dx2, dy2, 0], [dx2, -dy2, 0], [-dx2, -dy2, 0], 
                         [-dx2, dy2, 0], [dx2, dy2, 0]]).T

    @staticmethod
    def circular_path(radius=0.1, pts=20):
        return Wire.EllipticalPath(rx=radius, ry=radius, pts=pts)

    @staticmethod
    def sinus_circular_path(radius=0.1, amplitude=0.01, frequency=10, pts=100):
        t = np.linspace(0, 2 * np.pi, pts)
        return np.array([radius * np.sin(t), radius * np.cos(t), 
                         amplitude * np.cos(frequency*t)]).T

    @staticmethod
    def elliptical_path(rx=0.1, ry=0.2, pts=20):
        t = np.linspace(0, 2 * np.pi, pts)
        return np.array([rx * np.sin(t), ry * np.cos(t), 0]).T

    @staticmethod
    def solenoid_path(radius=0.1, pitch=0.01, turns=30, pts_per_turn=20):
        return Wire.EllipticalSolenoidPath(rx=radius, ry=radius, pitch=pitch, 
                                           turns=turns,
                                           pts_per_turn=pts_per_turn)

    @staticmethod
    def elliptical_solenoid_path(rx=0.1, ry=0.2, pitch=0.01,
                                 turns=30, pts_per_turn=20):
        t = np.linspace(0, 2 * np.pi * turns, pts_per_turn * turns)
        return np.array([rx * np.sin(t), ry * np.cos(t),
                         t / (2 * np.pi) * pitch]).T
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def vv_PlotWires(wires):
    ''' 
    function plots wires with visvis 
    '''
    for w in wires:
        w.vv_plot_path()

def mpl3d_PlotWires(wires, ax):
    ''' 
    function plots wires with matplotlib 
    '''
    for w in wires:
        w.mpl3d_plot_path(show=False, ax=ax)

if __name__ == '__main__':
    pass