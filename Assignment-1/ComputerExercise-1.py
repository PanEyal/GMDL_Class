import numpy as np
import matplotlib.pyplot as  plt


x2,x1 = np.mgrid[-2:2:0.01,-2:2:0.01]

def l1normSquared():
    return np.abs(x1) + np.abs(x2)

def l2normSquared():
    return x1**2 + x2**2

def QnormSquared(Q):
    return x1**2*Q[0,0]+x2**2*Q[1,1]+2*x1*x2*Q[0,1]

def axis_xy(g=None):
    # mimics Matlab's "axis xy" command;
    # making the y axis pointing upward when using imshow
    if g is None:
        g = plt.gca()
    bottom, top = g.get_ylim()
    if top<bottom:
        g.set_ylim(top,bottom)

def visualize(distances):
    plt.figure()
    less_than = (distances < 1.05).astype(int)
    greater_than = (distances > 0.95).astype(int)
    outline = less_than & greater_than
    plt.imshow(outline,cmap='gray')
    axis_xy()
    plt.axis('off')
    plt.title(r'0.95<$x^TQx<1.05$')
    # plt.subplots_adjust(wspace=0.4)
    plt.show()

#l2
Q=np.eye(2)
print ('Q_l2=\n',Q)
visualize(l2normSquared())

#l1
Q=np.eye(2)
print ('Q_l2=\n',Q)
visualize(l1normSquared())

#Q1
Q=np.array([[9,0],[0,1]])
print ('Q1=\n',Q)
visualize(QnormSquared(Q))

#Q2
Q=np.array([[9,2],[2,1]])
print ('Q2=\n',Q)
visualize(QnormSquared(Q))

#Q3
Q=np.array([[9,-2],[-2,1]])
print ('Q3=\n',Q)
visualize(QnormSquared(Q))
