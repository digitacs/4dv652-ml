import numpy as np

# Mirror
# 'a' means all axis else 'x', 'y' or 'z'
# if Append calculated data will be appended to the input data
def mirror(data, axis, append=False):
    try:
        
        if axis == 'a':
            target_labels = [col for col in data.columns]
        else:
            axis = "_" + axis
            target_labels = [col for col in data.columns if axis in col]

        aug_data_mirror = data.copy()

        for t in target_labels:
            temp = -aug_data_mirror[t]
            aug_data_mirror = aug_data_mirror.assign(**{t: temp.values})

        if append:
            return data.append(aug_data_mirror,ignore_index=True)

        return aug_data_mirror
    
    except IOError as e:
        print(e)
        return None

# Multiplier
# if multiplier > 1 strecth and if multiplier < 1 compress
# if Append calculated data will be appended to input data
def augMultiplier(data, multiplier, append=False):
    
    try:
        aug_data_offset = data.copy()
        aug_data_offset *= multiplier
        if append:
            return data.append(aug_data_offset,ignore_index=True)
        
        return aug_data_offset
    
    except IOError as e:
        print(e)
        return None

# Rotate
def rotatePoint(angle, x,y):
    a = np.radians(angle)
    cosa = np.cos(a)
    sina = np.sin(a)
    return x*cosa - y*sina, x*sina + y*cosa 

def rotate(data, angle, posenet=False):
    
    try:
        aug_data = data.copy()
        length = len(aug_data)
        if posenet:
            for index, row in aug_data.iterrows():
                head_x, head_y = rotatePoint(angle,row['head_x'],row['head_y'])

                left_shoulder_x, left_shoulder_y = rotatePoint(angle,row['left_shoulder_x'],row['left_shoulder_y'])
                right_shoulder_x, right_shoulder_y = rotatePoint(angle,row['right_shoulder_x'],row['right_shoulder_y'])
                left_elbow_x, left_elbow_y = rotatePoint(angle,row['left_elbow_x'],row['left_elbow_y'])
                right_elbow_x, right_elbow_y = rotatePoint(angle,row['right_elbow_x'],row['right_elbow_y'])
                left_wrist_x, left_wrist_y = rotatePoint(angle,row['left_wrist_x'],row['left_wrist_y'])
                right_wrist_x, right_wrist_y = rotatePoint(angle,row['right_wrist_x'],row['right_wrist_y'])
                left_hip_x, left_hip_y = rotatePoint(angle,row['left_hip_x'],row['left_hip_y'])
                right_hip_x, right_hip_y = rotatePoint(angle,row['right_hip_x'],row['right_hip_y'])
                left_knee_x, left_knee_y = rotatePoint(angle,row['left_knee_x'],row['left_knee_y'])
                right_knee_x, right_knee_y = rotatePoint(angle,row['right_knee_x'],row['right_knee_y'])
                left_ankle_x, left_ankle_y = rotatePoint(angle,row['left_ankle_x'],row['left_ankle_y'])
                right_ankle_x, right_ankle_y = rotatePoint(angle,row['right_ankle_x'],row['right_ankle_y']) 
                aug_data = aug_data.append({'head_x':head_x,'head_y':head_y,
                                            'left_shoulder_x':left_shoulder_x,'left_shoulder_y':left_shoulder_y,
                                            'right_shoulder_x':right_shoulder_x,'right_shoulder_y':right_shoulder_y,
                                            'left_elbow_x':left_elbow_x,'left_elbow_y':left_elbow_y,
                                            'right_elbow_x':right_elbow_x,'right_elbow_y':right_elbow_y,
                                            'left_wrist_x':left_wrist_x,'left_wrist_y':left_wrist_y,
                                            'right_wrist_x':right_wrist_x,'right_wrist_y':right_wrist_y,
                                            'left_hip_x':left_hip_x,'left_hip_y':left_hip_y,
                                            'right_hip_x':right_hip_x,'right_hip_y':right_hip_y,
                                            'left_knee_x':left_knee_x,'left_knee_y':left_knee_y,
                                            'right_knee_x':right_knee_x,'right_knee_y':right_knee_y,
                                            'left_ankle_x':left_ankle_x,'left_ankle_y':left_ankle_y,
                                            'right_ankle_x':right_ankle_x,'right_ankle_y':right_ankle_y},ignore_index=True)

        else:
            for index, row in aug_data.iterrows():
                head_x, head_y = rotatePoint(angle,row[' head_x'],row['head_y'])
                left_shoulder_x, left_shoulder_y = rotatePoint(angle,row['left_shoulder_x'],row['left_shoulder_y'])
                right_shoulder_x, right_shoulder_y = rotatePoint(angle,row['right_shoulder_x'],row['right_shoulder_y'])
                left_elbow_x, left_elbow_y = rotatePoint(angle,row['left_elbow_x'],row['left_elbow_y'])
                right_elbow_x, right_elbow_y = rotatePoint(angle,row['right_elbow_x'],row['right_elbow_y'])
                left_hand_x, left_hand_y = rotatePoint(angle,row['left_hand_x'],row['left_hand_y'])
                right_hand_x, right_hand_y = rotatePoint(angle,row['right_hand_x'],row['right_hand_y'])
                left_hip_x, left_hip_y = rotatePoint(angle,row['left_hip_x'],row['left_hip_y'])
                right_hip_x, right_hip_y = rotatePoint(angle,row['right_hip_x'],row['right_hip_y'])
                left_knee_x, left_knee_y = rotatePoint(angle,row['left_knee_x'],row['left_knee_y'])
                right_knee_x, right_knee_y = rotatePoint(angle,row['right_knee_x'],row['right_knee_y'])
                left_foot_x, left_foot_y = rotatePoint(angle,row['left_foot_x'],row['left_foot_y'])
                right_foot_x, right_foot_y = rotatePoint(angle,row['right_foot_x'],row['right_foot_y']) 
                aug_data = aug_data.append({' head_x':head_x,'head_y':head_y,
                                            'left_shoulder_x':left_shoulder_x,'left_shoulder_y':left_shoulder_y,
                                            'right_shoulder_x':right_shoulder_x,'right_shoulder_y':right_shoulder_y,
                                            'left_elbow_x':left_elbow_x,'left_elbow_y':left_elbow_y,
                                            'right_elbow_x':right_elbow_x,'right_elbow_y':right_elbow_y,
                                            'left_hand_x':left_hand_x,'left_hand_y':left_hand_y,
                                            'right_hand_x':right_hand_x,'right_hand_y':right_hand_y,
                                            'left_hip_x':left_hip_x,'left_hip_y':left_hip_y,
                                            'right_hip_x':right_hip_x,'right_hip_y':right_hip_y,
                                            'left_knee_x':left_knee_x,'left_knee_y':left_knee_y,
                                            'right_knee_x':right_knee_x,'right_knee_y':right_knee_y,
                                            'left_foot_x':left_foot_x,'left_foot_y':left_foot_y,
                                            'right_foot_x':right_foot_x,'right_foot_y':right_foot_y},ignore_index=True)
        
        return aug_data
    
    except IOError as e:
        print(e)
        return None

# Test
import matplotlib.pyplot as plt

def drawP(data,posenet=False):
    try:
        aug_data = data.copy()
        if posenet:
            for index, row in aug_data.iterrows():                
                fig=plt.figure() 
                ax=fig.add_subplot(111,projection='3d')

                # for rotate the axes and update.
                for angle in range(0,360): 
                    ax.view_init(90,angle)

                plt.plot([row['leftShoulder_x'],row['leftElbow_x']],[row['leftShoulder_y'],row['leftElbow_y']])
                
                plt.plot([row['leftElbow_x'],row['leftWrist_x']],[row['leftElbow_y'],row['leftWrist_y']])
                
                plt.plot([row['leftShoulder_x'],row['leftHip_x']],[row['leftShoulder_y'],row['leftHip_y']])
                
                plt.plot([row['leftHip_x'],row['leftKnee_x']],[row['leftHip_y'],row['leftKnee_y']])
                
                plt.plot([row['leftKnee_x'],row['leftAnkle_x']],[row['leftKnee_y'],row['leftAnkle_y']])
                
                plt.plot([row['leftShoulder_x'],row['rightShoulder_x']],[row['leftShoulder_y'],row['rightShoulder_y']])
                
                plt.plot([row['rightElbow_x'],row['rightWrist_x']],[row['rightElbow_y'],row['rightWrist_y']])
                
                plt.plot([row['rightShoulder_x'],row['rightHip_x']],[row['rightShoulder_y'],row['rightHip_y']])
                
                plt.plot([row['rightHip_x'],row['rightKnee_x']],[row['rightHip_y'],row['rightKnee_y']])
                
                plt.plot([row['rightShoulder_x'],row['rightElbow_x']],[row['rightShoulder_y'],row['rightElbow_y']])
                
                plt.plot([row['rightKnee_x'],row['rightAnkle_x']],[row['rightKnee_y'],row['rightAnkle_y']])

        else:
            for index, row in aug_data.iterrows():
                
                fig=plt.figure() 
                ax=fig.add_subplot(111,projection='3d')

                # for rotate the axes and update.
                for angle in range(0,360): 
                    ax.view_init(90,angle)

                plt.plot([row['leftShoulder_x'],row['leftElbow_x']],[row['leftShoulder_y'],row['leftElbow_y']])
                
                plt.plot([row['leftElbow_x'],row['leftWrist_x']],[row['leftElbow_y'],row['leftWrist_y']])
                
                plt.plot([row['leftShoulder_x'],row['leftHip_x']],[row['leftShoulder_y'],row['leftHip_y']])
                
                plt.plot([row['leftHip_x'],row['leftKnee_x']],[row['leftHip_y'],row['leftKnee_y']])
                
                plt.plot([row['leftKnee_x'],row['leftAnkle_x']],[row['leftKnee_y'],row['leftAnkle_y']])
                
                plt.plot([row['leftShoulder_x'],row['rightShoulder_x']],[row['leftShoulder_y'],row['rightShoulder_y']])
                
                plt.plot([row['rightElbow_x'],row['rightWrist_x']],[row['rightElbow_y'],row['rightWrist_y']])
                
                plt.plot([row['rightShoulder_x'],row['rightHip_x']],[row['rightShoulder_y'],row['rightHip_y']])
                
                plt.plot([row['rightHip_x'],row['rightKnee_x']],[row['rightHip_y'],row['rightKnee_y']])
                
                plt.plot([row['rightShoulder_x'],row['rightElbow_x']],[row['rightShoulder_y'],row['rightElbow_y']])
                
                plt.plot([row['rightKnee_x'],row['rightAnkle_x']],[row['rightKnee_y'],row['rightAnkle_y']])
                    
                plt.show()
    except IOError as e:
        print(e)

# Angle
def multiDimenDist(point1,point2):
    #find the difference between the two points, its really the same as below
    deltaVals = [point2[dimension]-point1[dimension] for dimension in range(len(point1))]
    runningSquared = 0
    #because the pythagarom theorm works for any dimension we can just use that
    for coOrd in deltaVals:
        runningSquared += coOrd**2
    return runningSquared**(1/2)
def findVec(point1,point2,unitSphere = False):
    #setting unitSphere to True will make the vector scaled down to a sphere with a radius one, instead of it's orginal length
    finalVector = [0 for coOrd in point1]
    for dimension, coOrd in enumerate(point1):
        #finding total differnce for that co-ordinate(x,y,z...)
        deltaCoOrd = point2[dimension]-coOrd
        #adding total difference
        finalVector[dimension] = deltaCoOrd
    if unitSphere:
        totalDist = multiDimenDist(point1,point2)
        unitVector =[]
        for dimen in finalVector:
            unitVector.append( dimen/totalDist)
        return unitVector
    else:
        return finalVector

def angleP(data,posenet=False):
    try:
        aug_data = data.copy()
        if posenet:
            for index, row in aug_data.iterrows():
                #LEFT
                vector_11 = findVec([row['leftShoulder_x'],row['leftShoulder_y']], [row['leftElbow_x'],row['leftElbow_y']])
                vector_12 = findVec([row['leftShoulder_x'],row['leftShoulder_y']], [row['leftHip_x'],row['leftHip_y']])

                unit_vector_11 = vector_11 / np.linalg.norm(vector_11)
                unit_vector_12 = vector_12 / np.linalg.norm(vector_12)
                angle_left_1 = np.arccos(np.dot(unit_vector_11, unit_vector_12))

                #print(np.degrees(angle_left_1))
                
                vector_13 = findVec([row['leftElbow_x'],row['leftElbow_y']], [row['leftShoulder_x'],row['leftShoulder_y']])
                vector_14 = findVec([row['leftElbow_x'],row['leftElbow_y']], [row['leftWrist_x'],row['leftWrist_y']])

                unit_vector_13 = vector_13 / np.linalg.norm(vector_13)
                unit_vector_14 = vector_14 / np.linalg.norm(vector_14)
                angle_left_2 = np.arccos(np.dot(unit_vector_13, unit_vector_14))

                #print(np.degrees(angle_left_2))
                
                #RIGHT
                vector_21 = findVec([row['rightShoulder_x'],row['rightShoulder_y']], [row['rightElbow_x'],row['rightElbow_y']])
                vector_22 = findVec([row['rightShoulder_x'],row['rightShoulder_y']], [row['rightHip_x'],row['rightHip_y']])

                unit_vector_21 = vector_21 / np.linalg.norm(vector_21)
                unit_vector_22 = vector_22 / np.linalg.norm(vector_22)
                angle_right_1 = np.arccos(np.dot(unit_vector_21, unit_vector_22))

                #print(np.degrees(angle_right_1))
                
                vector_23 = findVec([row['rightElbow_x'],row['rightElbow_y']], [row['rightShoulder_x'],row['rightShoulder_y']])
                vector_24 = findVec([row['rightElbow_x'],row['rightElbow_y']], [row['rightWrist_x'],row['rightWrist_y']])

                unit_vector_23 = vector_23 / np.linalg.norm(vector_23)
                unit_vector_24 = vector_24 / np.linalg.norm(vector_24)
                angle_right_2 = np.arccos(np.dot(unit_vector_23, unit_vector_24))

                #print(np.degrees(angle_right_2))
                
                aug_data.loc[index,'angle_left_1'] = angle_left_1
                aug_data.loc[index,'angle_left_2'] = angle_left_2
                aug_data.loc[index,'angle_right_1'] = angle_right_1
                aug_data.loc[index,'angle_right_2'] = angle_right_2
        else:
            for index, row in aug_data.iterrows():
                #LEFT
                vector_11 = findVec([row['left_shoulder_x'],row['left_shoulder_y']], [row['left_elbow_x'],row['left_elbow_y']])
                vector_12 = findVec([row['left_shoulder_x'],row['left_shoulder_y']], [row['left_hip_x'],row['left_hip_y']])

                unit_vector_11 = vector_11 / np.linalg.norm(vector_11)
                unit_vector_12 = vector_12 / np.linalg.norm(vector_12)
                angle_left_1 = np.arccos(np.dot(unit_vector_11, unit_vector_12))

                #print(np.degrees(angle_left_1))
                
                vector_13 = findVec([row['left_elbow_x'],row['left_elbow_y']], [row['left_shoulder_x'],row['left_shoulder_y']])
                vector_14 = findVec([row['left_elbow_x'],row['left_elbow_y']], [row['left_wrist_x'],row['left_wrist_y']])

                unit_vector_13 = vector_13 / np.linalg.norm(vector_13)
                unit_vector_14 = vector_14 / np.linalg.norm(vector_14)
                angle_left_2 = np.arccos(np.dot(unit_vector_13, unit_vector_14))

                #print(np.degrees(angle_left_2))
                
                #RIGHT
                vector_21 = findVec([row['right_shoulder_x'],row['right_shoulder_y']], [row['right_elbow_x'],row['right_elbow_y']])
                vector_22 = findVec([row['right_shoulder_x'],row['right_shoulder_y']], [row['right_hip_x'],row['right_hip_y']])

                unit_vector_21 = vector_21 / np.linalg.norm(vector_21)
                unit_vector_22 = vector_22 / np.linalg.norm(vector_22)
                angle_right_1 = np.arccos(np.dot(unit_vector_21, unit_vector_22))

                #print(np.degrees(angle_right_1))
                
                vector_23 = findVec([row['right_elbow_x'],row['right_elbow_y']], [row['right_shoulder_x'],row['right_shoulder_y']])
                vector_24 = findVec([row['right_elbow_x'],row['right_elbow_y']], [row['right_wrist_x'],row['right_wrist_y']])

                unit_vector_23 = vector_23 / np.linalg.norm(vector_23)
                unit_vector_24 = vector_24 / np.linalg.norm(vector_24)
                angle_right_2 = np.arccos(np.dot(unit_vector_23, unit_vector_24))

                #print(np.degrees(angle_right_2))
                
                aug_data.loc[index,'angle_left_1'] = angle_left_1
                aug_data.loc[index,'angle_left_2'] = angle_left_2
                aug_data.loc[index,'angle_right_1'] = angle_right_1
                aug_data.loc[index,'angle_right_2'] = angle_right_2
        
        return aug_data
    except IOError as e:
        print(e)