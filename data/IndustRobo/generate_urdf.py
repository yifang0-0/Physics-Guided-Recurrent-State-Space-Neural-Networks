import math
def generate_urdf():
    # Define the parameters for each link and joint
    # link_params = [
    #     {'d': 0.675, 'a': 0.350, 'alpha': -math.pi/2, 'I': [1.742e+1, 3.175e+1, 3.493e+1, 2.455e+0, -1.234e-1, -4.47056e+0], 'r': [-352.17374e-3, 169.90937e-3, -11.42400e-3], 'm': 402.26, 'Jm': 0.00923, 'G': 212.76, 'B': 0.0021517, 'Tc': [0.89302, -0.89302], 'qlim': [-147*math.pi/180, 147*math.pi/180], 'offset': -90*math.pi/180, 'flip': True},  # Define parameters for L(1)
    #     {'d': -0.189, 'a': 1.15, 'alpha': 0, 'I': [7.293e+0, 8.742e+1, 8.712e+1, -6.600e-1, -9.125e-2, 3.924e+0], 'r': [-705.34904e-3, -3.56655e-3, 0e-3], 'm': 332.14, 'Jm': 0.0118, 'G': 203.52, 'B': 0.0184437, 'Tc': [2.45399, -2.45399], 'qlim': [-140*math.pi/180 , -5*math.pi/180], 'offset': 0},  # Define parameters for L(2)
    #     {'d': 0.189, 'a': 0.041, 'alpha': -math.pi/2, 'I': [2.317e+1, 2.315e+1, 3.43410e+0, -2.545e-1, 1.27099e+0, 1.085e+0], 'r': [-39.8514e-3, -43.0814e-3, -183.83108e-3], 'm': 167.89, 'Jm': 0.0118, 'G': 192.75, 'B': 0.0143936, 'Tc': [2.33463, -2.33463], 'qlim': [-112 *math.pi/180,  153*math.pi/180], 'offset': 90*math.pi/180},  # Define parameters for L(3)
    #     {'d': -1, 'a': 0, 'alpha': math.pi/2, 'I': [1.324e-1, 4.509e-2, 1.361e-1, 5.608e-7, 6.530e-3, -5.01236e-7], 'r': [0.00055e-3, 121.91066e-3, 4.32167e-3], 'm': 9.69, 'Jm': 0.00173, 'G': 156, 'B': 0.0038455, 'Tc': [0.60897, -0.60897], 'qlim': [-350*math.pi/180, 350*math.pi/180], 'offset': 0},  # Define parameters for L(4)
    #     {'d': 0, 'a': 0, 'alpha': -math.pi/2, 'I': [7.185e-1, 5.55113e-1, 4.384e-1, 3.801e-5, 1.519e-1, 1.056e-4], 'r': [0.00454e-3, -49.96316e-3, -59.16827e-3], 'm': 49.61, 'Jm': 0.00173, 'G': 156, 'B': 0.0038455, 'Tc': [0.60897, -0.60897], 'qlim': [-122.5*math.pi/180,122.5*math.pi/180], 'offset': 0},  # Define parameters for L(5)
    #     {'d': -0.24, 'a': 0, 'alpha': math.pi, 'I': [3.880e-2, 1.323e-1, 1.681e-1, 2.635e-2, 1.590e-3, -3.322e-3], 'r': [-66.63199e-3, 17.20624e-3, -16.63216e-3], 'm': 9.18, 'Jm': 0.00173, 'G': 102.17, 'B': 0.0050314, 'Tc': [0.53832, -0.53832], 'qlim': [-350*math.pi/180, 350*math.pi/180]},  # Define parameters for L(6)
    # ]
    link_params = [
        {'d': 0.675, 'a': 0.350, 'alpha': -math.pi/2, 'I': [1.742e+1, 3.175e+1, 3.493e+1, 2.455e+0, -1.234e-1, -4.47056e+0], 'r': [-352.17374e-3, 169.90937e-3, -11.42400e-3], 'm': 402.26, 'Jm': 0.00923, 'G': 212.76, 'B': 0.0021517, 'Tc': [0.89302, -0.89302], 'qlim': [-147, 147], 'offset': -90, 'flip': True},  # Define parameters for L(1)
        {'d': -0.189, 'a': 1.15, 'alpha': 0, 'I': [7.293e+0, 8.742e+1, 8.712e+1, -6.600e-1, -9.125e-2, 3.924e+0], 'r': [-705.34904e-3, -3.56655e-3, 0e-3], 'm': 332.14, 'Jm': 0.0118, 'G': 203.52, 'B': 0.0184437, 'Tc': [2.45399, -2.45399], 'qlim': [-140 , -5], 'offset': 0},  # Define parameters for L(2)
        {'d': 0.189, 'a': 0.041, 'alpha': -math.pi/2, 'I': [2.317e+1, 2.315e+1, 3.43410e+0, -2.545e-1, 1.27099e+0, 1.085e+0], 'r': [-39.8514e-3, -43.0814e-3, -183.83108e-3], 'm': 167.89, 'Jm': 0.0118, 'G': 192.75, 'B': 0.0143936, 'Tc': [2.33463, -2.33463], 'qlim': [-112 ,  153], 'offset': 90},  # Define parameters for L(3)
        {'d': -1, 'a': 0, 'alpha': math.pi/2, 'I': [1.324e-1, 4.509e-2, 1.361e-1, 5.608e-7, 6.530e-3, -5.01236e-7], 'r': [0.00055e-3, 121.91066e-3, 4.32167e-3], 'm': 9.69, 'Jm': 0.00173, 'G': 156, 'B': 0.0038455, 'Tc': [0.60897, -0.60897], 'qlim': [-350, 350], 'offset': 0},  # Define parameters for L(4)
        {'d': 0, 'a': 0, 'alpha': -math.pi/2, 'I': [7.185e-1, 5.55113e-1, 4.384e-1, 3.801e-5, 1.519e-1, 1.056e-4], 'r': [0.00454e-3, -49.96316e-3, -59.16827e-3], 'm': 49.61, 'Jm': 0.00173, 'G': 156, 'B': 0.0038455, 'Tc': [0.60897, -0.60897], 'qlim': [-122.5,122.5], 'offset': 0},  # Define parameters for L(5)
        {'d': -0.24, 'a': 0, 'alpha': math.pi, 'I': [3.880e-2, 1.323e-1, 1.681e-1, 2.635e-2, 1.590e-3, -3.322e-3], 'r': [-66.63199e-3, 17.20624e-3, -16.63216e-3], 'm': 9.18, 'Jm': 0.00173, 'G': 102.17, 'B': 0.0050314, 'Tc': [0.53832, -0.53832], 'qlim': [-350, 350]},  # Define parameters for L(6)
    ]    
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    joint_values = [
        [0, 0, 0, 0, 0, 0],            # qz
        [0, -math.pi/2, math.pi/2, 0, 0, 0],  # qs
        [0, -2/3*math.pi, 3/4*math.pi, 0, math.pi/4, 0],  # qr
        [0, math.pi/2, 0, 0, 0, 0],    # qh
        [0, -140, 150, 0, -120, 0],  # qt
    ]

    # Generate the URDF file content
    urdf_content = """<?xml version="1.0"?>
<robot name="KR300">
"""
    for i, params in enumerate(link_params):
        urdf_content += f"\t<link name='link{i+1}'>\n"
        # Add parameters for each link
        for key, value in params.items():
            urdf_content += f"\t\t<{key}>{value}</{key}>\n"
        urdf_content += "\t</link>\n"
    
    for i, joint_name in enumerate(joint_names):
        urdf_content += f"\t<joint name='{joint_name}' type='revolute'>\n"
        urdf_content += f"\t\t<parent link='link{i}'/>\n"
        urdf_content += f"\t\t<child link='link{i+1}'/>\n"
        urdf_content += "\t\t<origin xyz='0 0 0' rpy='0 0 0'/>\n"
        urdf_content += "\t\t<axis xyz='0 0 1'/>\n"
        urdf_content += "\t\t<limit lower='-350' upper='350'/>\n"  # Assuming joint limits
        urdf_content += "\t</joint>\n"

    # Add joint configurations for each pose
    for i, pose in enumerate(joint_values):
        urdf_content += f"\t<group_state name='pose_{i}'>\n"
        for j, value in enumerate(pose):
            urdf_content += f"\t\t<joint name='joint{j+1}' value='{value}'/>\n"
        urdf_content += "\t</group_state>\n"

    urdf_content += "</robot>"
    return urdf_content

# Save the URDF file
urdf_content = generate_urdf()
with open("robot_model.urdf", "w") as f:
    f.write(urdf_content)