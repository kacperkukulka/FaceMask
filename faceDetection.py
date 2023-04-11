import cv2
import mediapipe as mp
import math
import numpy as np
import trimesh
import pyrender

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def vec_length(a, b):
    x = a[0]-b[0]
    y = a[1]-b[1]
    return math.sqrt(x*x + y*y)

def rotations(pNose, pLeft, pRight, isDraw=False, image=None):
    xx = vec_length(pRight,pNose)
    yy = vec_length(pLeft,pNose)
    zz = vec_length(pLeft,pRight)
    pp = (xx*xx + zz*zz - yy*yy) / (2*zz)
    hh = math.sqrt(xx*xx - pp*pp)
    a = (pLeft[0] - pRight[0])/(pLeft[1] - pRight[1])
    b = pLeft[0] - a*pLeft[1]
    yrot = yRotation(pNose, zz, pp, isDraw, image)
    xrot = xRotation(pNose, pLeft, pRight, hh, a, b, isDraw, image)
    return (xrot,yrot)

def yRotation(pNose, zz, pp, isDraw=False, image=None):
    pos = -1
    if pp/zz < 0.5:
        pp = zz-pp
        pos = 1
    angleY = pp/zz
    angleY -= 0.5
    angleY = -30.0*angleY*angleY + 105.0*angleY
    angleY *= pos
    if isDraw:
        pNose = np.floor(pNose).astype(int)
        cv2.line(image,(pNose[1],pNose[0]),(pNose[1]- math.floor(angleY*2),pNose[0]),(255,0,0), 2)
    return angleY

def xRotation(pNose, pLeft, pRight, hh, a, b, isDraw = False, image = None):
    pos = 1
    #print(str(pNose[0]) + " " + str(a*pNose[1] + b))
    if pNose[0] > a*pNose[1] + b:
        pos = -1
    hh = hh * pos
    hh = hh - 37
    a = -45.0/77.0
    b = -1035.0/77.0
    if isDraw:
        pNose = np.floor(pNose).astype(int)
        cv2.line(image,(pNose[1],pNose[0]),(pNose[1],pNose[0] - math.floor(hh)),(0,0,255), 2)
    return (a*hh + b)

def findEdges(landmarks):
    minx, miny, maxx, maxy = 20000, 20000, -1, -1

    for i in landmarks:
        point = [image.shape[0] * i.y , image.shape[1] * i.x]
        if point[0]> maxx:
            maxx = point[0]
        if point[0]< minx:
            minx= point[0]
        if point[1]> maxy:
            maxy = point[1]
        if point[1]< miny:
            miny = point[1]
    lenx = maxx - minx
    leny = maxy - miny
    lenAll = max(lenx,leny)
    if lenx > leny:
        more = (lenx - leny)/2.0
        pStart = (math.floor(miny-more),math.floor(minx))
    else:
        more = (leny - lenx)/2.0 
        pStart = (math.floor(miny), math.floor(minx - more))

    return (pStart, lenAll)

def applyRotations(mesh, light, camera_pose, yrot, xrot):
    roty = trimesh.transformations.rotation_matrix(math.pi*(yrot/180.0),[0,1,0],[0,0,0])
    rotx = trimesh.transformations.rotation_matrix(math.pi*(xrot/180.0),[1,0,0],[0,0,0])
    mesh.apply_transform(rotx)
    mesh.apply_transform(roty)
    scene = pyrender.Scene.from_trimesh_scene(mesh)
    scene.add(cam, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    color, _ = r.render(scene)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  
    color = cv2.flip(color,1)  
    lenAllBig = math.floor(lenAll*3)
    lenAllOff = math.floor(lenAll)
    color = cv2.resize(color,(lenAllBig,lenAllBig))

    # undo rotations
    roty = trimesh.transformations.rotation_matrix(math.pi*(yrot/180.0),[0,-1,0],[0,0,0])
    rotx = trimesh.transformations.rotation_matrix(math.pi*(xrot/180.0),[-1,0,0],[0,0,0])
    mesh.apply_transform(roty)
    mesh.apply_transform(rotx) 
    return (color, lenAllOff, lenAllBig)

def addPhotos(color, image, lenAllOff, offset, pStart):
    colorGrey = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    _, colorGrey = cv2.threshold(colorGrey,254,255,cv2.THRESH_BINARY_INV)
    color = cv2.bitwise_and(cv2.cvtColor(colorGrey, cv2.COLOR_GRAY2BGR),color)
    colorGrey = cv2.bitwise_not(colorGrey)
    colorGrey = cv2.cvtColor(colorGrey, cv2.COLOR_GRAY2BGR)
    try:
        odX, doX = pStart[1]-lenAllOff-offset, pStart[1]-lenAllOff-offset+lenAllBig
        odXmodel, doXmodel = 0, color.shape[0]
        odY, doY = pStart[0]-lenAllOff, pStart[0]-lenAllOff+lenAllBig
        odYmodel, doYmodel = 0, color.shape[1]
        
        offsetModelXod = 0
        offsetModelXdo = 0
        offsetModelYod = 0
        offsetModelYdo = 0

        if odX < 0:
            offsetModelXod = abs(odX)
            odX = 0
        if doX > image.shape[0]:
            offsetModelXdo = doX - image.shape[0]
            doX = image.shape[0]
        if odY < 0:
            offsetModelYod = abs(odY)
            odY = 0
        if doY > image.shape[1]:
            offsetModelYdo = doY - image.shape[1]
            doY = image.shape[1]

        image[odX:doX,odY:doY] = cv2.bitwise_and(image[odX:doX,odY:doY],colorGrey[odXmodel+offsetModelXod:doXmodel-offsetModelXdo,odYmodel+offsetModelYod:doYmodel-offsetModelYdo])
        image[odX:doX,odY:doY] = cv2.bitwise_or(image[odX:doX,odY:doY],color[odXmodel+offsetModelXod:doXmodel-offsetModelXdo,odYmodel+offsetModelYod:doYmodel-offsetModelYdo])       
    except:
        None
##main code

modelShow = True
axisShow = False
num_of_people = 2

cam = pyrender.camera.PerspectiveCamera(yfov=np.pi / 6.0)
s = np.sqrt(2)/2

camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.3],
    [0, 0, 0, 1],
])

light = pyrender.SpotLight(color=np.ones(3), intensity=1.0,
    innerConeAngle=np.pi/16.0,
    outerConeAngle=np.pi/6.0)

mesh = trimesh.load('RubikCube.obj')
r = pyrender.OffscreenRenderer(viewport_height=640, viewport_width=640, point_size=1.0)


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=num_of_people,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        width = image.shape[0] * 2
        height = image.shape[1] * 2

        image = cv2.resize(image,(height,width),interpolation = cv2.INTER_AREA)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image = cv2.flip(image, 1)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pNose = [image.shape[0] * face_landmarks.landmark[6].y , image.shape[1] * face_landmarks.landmark[6].x]
                pLeft = [image.shape[0] * face_landmarks.landmark[454].y , image.shape[1] * face_landmarks.landmark[454].x]
                pRight = [image.shape[0] * face_landmarks.landmark[234].y , image.shape[1] * face_landmarks.landmark[234].x]
                
                minx, miny, maxx, maxy = 20000, 20000, -1, -1

                pStart, lenAll = findEdges(face_landmarks.landmark)

                xrot, yrot = rotations(pNose,pLeft,pRight, axisShow,image)
                if modelShow:
                    offset = 60
                    color, lenAllOff, lenAllBig = applyRotations(mesh, light, camera_pose, yrot, xrot)
                    addPhotos(color, image, lenAllOff, offset, pStart)

        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image,1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()