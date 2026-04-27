import roboticstoolbox as rtb
from roboticstoolbox.robot import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteDH
import numpy as np
import matplotlib as plt
class kuka300(DHRobot):
    def __init__(self, robot_type = "correct"):
        if robot_type == "correct":
            links = [
                    RevoluteDH(
                                d= 0.675,                                                  # link length (Denavit-Hartenberg-Notation) 
                                a= 0.350,                                                              # link offset (Denavit-Hartenberg-Notation)
                                alpha= -np.pi/2,     # 90                                                     # link twist (Denavit-Hartenberg-Notation)
                                I= [1.742e+1, 3.175e+1, 3.493e+1, 2.455e+0, -1.234e-1, -4.47056e+0],   # inertia tensor of link with respect to center of mass I = [L_xx, L_yy, L_zz, L_xy, L_yz, L_xz]
                                r= [-352.17374e-3, 169.90937e-3, -11.42400e-3],                        # distance of ith origin to center of mass [x,y,z] in link reference frame
                                m= 402.26,                                                             # mass of link 
                                Jm= 0.00923,                                                           # actuator inertia 
                                G= 212.76,                                                              # gear ratio
                                B= 0.0021517,                                                        # actuator viscous friction coefficient (actual value/G²)
                                Tc= [0.89302, -0.89302],                                               # actuator Coulomb friction coefficient for direction [-,+] (actual value/G)
                                qlim= [-90*np.pi/180, 90*np.pi/180],                                                 # maximum backward and forward link rotation
                                # offset= -90*np.pi/180,                                                       # compensation for DH-theta value -> offset on link rotation
                                offset= 0*np.pi/180,                                                       # compensation for DH-theta value -> offset on link rotation
                                flip=True
                            ),
                        RevoluteDH(
                                d= -0.189,
                                a= 1.15,
                                alpha= 0,           # 0
                                I= [7.293e+0, 8.742e+1, 8.712e+1, -6.600e-1, -9.125e-2, 3.924e+0],
                                r= [-705.34904e-3, -3.56655e-3, 0e-3],
                                m= 332.14,
                                Jm= 0.0118,
                                G= 203.52,
                                B= 0.0184437,
                                Tc= [2.45399, -2.45399],
                                qlim= [-30*np.pi/180, 40*np.pi/180],
                                offset= -90*np.pi/180

                            ),
                        RevoluteDH(
                                d= 0.189,
                                a= 0.041,
                                alpha= -np.pi/2,        # -90
                                I= [2.317e+1, 2.315e+1, 3.43410e+0, -2.545e-1, 1.27099e+0, 1.085e+0],
                                r= [-39.8514e-3, -43.0814e-3, -183.83108e-3],
                                m= 167.89,
                                Jm= 0.0118,
                                G= 192.75,
                                B= 0.0143936,
                                Tc= [2.33463, -2.33463],
                                qlim= [-110*np.pi/180, 40*np.pi/180],
                                offset= 90*np.pi/180
                                
                                ),
                        RevoluteDH(
                                    d= -1,
                                    a= 0,
                                    alpha= np.pi/2,         # 90
                                    I= [1.324e-1, 4.509e-2, 1.361e-1, 5.608e-7, 6.530e-3, -5.01236e-7],
                                    r= [0.00055e-3, 121.91066e-3, 4.32167e-3],
                                    m= 9.69,
                                    Jm= 0.00173,
                                    G= 156,
                                    B= 0.0038455,
                                    Tc= [0.60897, -0.60897],
                                    qlim= [-180*np.pi/180, 180*np.pi/180],
                                    offset= 0*np.pi/180
                                ),
                        RevoluteDH(
                                    d= 0,
                                    a= 0,
                                    alpha= -np.pi/2,        # -90
                                    I= [7.185e-1, 5.55113e-1, 4.384e-1, 3.801e-5, 1.519e-1, 1.056e-4],
                                    r= [0.00454e-3, -49.96316e-3, -59.16827e-3],
                                    m= 49.61,
                                    Jm= 0.00173,
                                    G= 156,
                                    B= 0.0038455,
                                    Tc= [0.60897, -0.60897],
                                    qlim= [-90*np.pi/180, 90*np.pi/180],
                                    offset= 0*np.pi/180
                                ),
                        RevoluteDH(
                                    d= -0.24,
                                    a= 0,
                                    alpha= np.pi,           # 180
                                    I= [3.880e-2, 1.323e-1, 1.681e-1, 2.635e-2, 1.590e-3, -3.322e-3],
                                    r= [-66.63199e-3, 17.20624e-3, -16.63216e-3],
                                    m= 9.18,
                                    Jm= 0.00173,
                                    G= 102.17,
                                    B= 0.0050314,
                                    Tc= [0.53832, -0.53832],
                                    qlim= [-180*np.pi/180, 180*np.pi/180]
                                )
                    ]
        elif robot_type == "no_offset":
                        links = [
                    RevoluteDH(
                                d= 0.675,                                                  # link length (Denavit-Hartenberg-Notation) 
                                a= 0.350,                                                              # link offset (Denavit-Hartenberg-Notation)
                                alpha= -np.pi/2,     # 90                                                     # link twist (Denavit-Hartenberg-Notation)
                                I= [1.742e+1, 3.175e+1, 3.493e+1, 2.455e+0, -1.234e-1, -4.47056e+0],   # inertia tensor of link with respect to center of mass I = [L_xx, L_yy, L_zz, L_xy, L_yz, L_xz]
                                r= [-352.17374e-3, 169.90937e-3, -11.42400e-3],                        # distance of ith origin to center of mass [x,y,z] in link reference frame
                                m= 402.26,                                                             # mass of link 
                                Jm= 0.00923,                                                           # actuator inertia 
                                G= 212.76,                                                              # gear ratio
                                B= 0.0021517,                                                        # actuator viscous friction coefficient (actual value/G²)
                                Tc= [0.89302, -0.89302],                                               # actuator Coulomb friction coefficient for direction [-,+] (actual value/G)
                                qlim= [-90*np.pi/180, 90*np.pi/180],                                                 # maximum backward and forward link rotation
                                # offset= -90*np.pi/180,                                                       # compensation for DH-theta value -> offset on link rotation
                                offset= 0*np.pi/180,                                                       # compensation for DH-theta value -> offset on link rotation
                                flip=True
                            ),
                        RevoluteDH(
                                d= -0.189,
                                a= 1.15,
                                alpha= 0,           # 0
                                I= [7.293e+0, 8.742e+1, 8.712e+1, -6.600e-1, -9.125e-2, 3.924e+0],
                                r= [-705.34904e-3, -3.56655e-3, 0e-3],
                                m= 332.14,
                                Jm= 0.0118,
                                G= 203.52,
                                B= 0.0184437,
                                Tc= [2.45399, -2.45399],
                                qlim= [-30*np.pi/180, 40*np.pi/180],
                                # offset= -90*np.pi/180
                                offset= 0*np.pi/180
                                

                            ),
                        RevoluteDH(
                                d= 0.189,
                                a= 0.041,
                                alpha= -np.pi/2,        # -90
                                I= [2.317e+1, 2.315e+1, 3.43410e+0, -2.545e-1, 1.27099e+0, 1.085e+0],
                                r= [-39.8514e-3, -43.0814e-3, -183.83108e-3],
                                m= 167.89,
                                Jm= 0.0118,
                                G= 192.75,
                                B= 0.0143936,
                                Tc= [2.33463, -2.33463],
                                qlim= [-110*np.pi/180, 40*np.pi/180],
                                # offset= 90*np.pi/180
                                offset= 0*np.pi/180
                                
                                
                                ),
                        RevoluteDH(
                                    d= -1,
                                    a= 0,
                                    alpha= np.pi/2,         # 90
                                    I= [1.324e-1, 4.509e-2, 1.361e-1, 5.608e-7, 6.530e-3, -5.01236e-7],
                                    r= [0.00055e-3, 121.91066e-3, 4.32167e-3],
                                    m= 9.69,
                                    Jm= 0.00173,
                                    G= 156,
                                    B= 0.0038455,
                                    Tc= [0.60897, -0.60897],
                                    qlim= [-180*np.pi/180, 180*np.pi/180],
                                    offset= 0*np.pi/180
                                ),
                        RevoluteDH(
                                    d= 0,
                                    a= 0,
                                    alpha= -np.pi/2,        # -90
                                    I= [7.185e-1, 5.55113e-1, 4.384e-1, 3.801e-5, 1.519e-1, 1.056e-4],
                                    r= [0.00454e-3, -49.96316e-3, -59.16827e-3],
                                    m= 49.61,
                                    Jm= 0.00173,
                                    G= 156,
                                    B= 0.0038455,
                                    Tc= [0.60897, -0.60897],
                                    qlim= [-90*np.pi/180, 90*np.pi/180],
                                    offset= 0*np.pi/180
                                ),
                        RevoluteDH(
                                    d= -0.24,
                                    a= 0,
                                    alpha= np.pi,           # 180
                                    I= [3.880e-2, 1.323e-1, 1.681e-1, 2.635e-2, 1.590e-3, -3.322e-3],
                                    r= [-66.63199e-3, 17.20624e-3, -16.63216e-3],
                                    m= 9.18,
                                    Jm= 0.00173,
                                    G= 102.17,
                                    B= 0.0050314,
                                    Tc= [0.53832, -0.53832],
                                    qlim= [-180*np.pi/180, 180*np.pi/180]
                                )
                    ]
        elif robot_type == "no_qlim":
            pass
        super().__init__(links, name="KR300")
        # self.payload(150)
        self.gravity = [0, 0, 9.81]
        self.qz = np.array([0, 0, 0, 0, 0, 0])  # zero angles
        self.qs = np.array([0, -np.pi/2, np.pi/2, 0, 0, 0])  # start at -90° pose
        self.qr = np.array([0, -2/3*np.pi, 3/4*np.pi, 0, np.pi/4, 0])  # ready pose Z-Shape
        self.qh = np.array([0, np.pi/2, 0, 0, 0, 0])  # hanging down
        self.qd0 = np.array([0, 0, 0, 0, 0, 0])
        self.qt = np.pi / 180*np.array([0, -140, 150, 0, -120, 0])
        self.dof = 6
# t.qz = np.random.rand(dof) 
# print(t.qz,t)
# t.qr = np.array([0, -2/3*np.pi, 3/4*np.pi, 0, np.pi/4, 0])  # ready pose Z-Shape
# t.qh = np.array([0, np.pi/2, 0, 0, 0, 0])  # hanging down
# t.qt = np.pi / 180
# acc = t.accel(t.qz,np.random.rand(dof) , np.random.rand(dof) )
# print(t,acc, t.gravity)
# print(t.inertia(t.qz))