import cv2
import numpy as np
import math

MAX_FRAME = 1000
MIN_NUM_FEAT = 2000

def getAbsoluteScale(frame_id, sequence_id, z_cal):
    x, y, z = 0, 0, 0
    x_prev, y_prev, z_prev = 0, 0, 0

    with open("/home/mehmet/Datasets/KITTI_VO/00.txt") as file:
        for i, line in enumerate(file):
            if i <= frame_id:
                data = line.split()
                z_prev = z
                x_prev = x
                y_prev = y
                z = float(data[11])
                y = float(data[7])
                x = float(data[3])

    return math.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)

def featureDetection(img, points):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)
    points.extend(kp)

def featureTracking(img1, img2, points1, points2, status):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    for i, match in enumerate(matches):
        if match.distance < 50:
            points2[i] = kp2[match.trainIdx].pt
            status[i] = 1

    points2 = [cv2.KeyPoint(point[0], point[1], 1) for point in points2 if status[points2.index(point)] == 1]

def main():
    img_1_c = cv2.imread("/home/mehmet/Datasets/KITTI_VO/00/image_2/000000.png")
    img_2_c = cv2.imread("/home/mehmet/Datasets/KITTI_VO/00/image_2/000001.png")

    if img_1_c is None or img_2_c is None:
        print("Error reading images")
        return

    img_1 = cv2.cvtColor(img_1_c, cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(img_2_c, cv2.COLOR_BGR2GRAY)

    points1 = []
    featureDetection(img_1, points1)

    points2 = [None] * len(points1)
    status = [0] * len(points1)
    featureTracking(img_1, img_2, points1, points2, status)

    focal = 718.8560
    pp = (607.1928, 185.2157)

    E, mask = cv2.findEssentialMat(np.array(points2), np.array(points1), focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, np.array(points2), np.array(points1), focal=focal, pp=pp, mask=mask)

    prevImage = img_2
    prevFeatures = points2

    traj = np.zeros((600, 600, 3), dtype=np.uint8)

    R_f = R.copy()
    t_f = t.copy()

    for numFrame in range(2, MAX_FRAME):
        filename = f"/home/avisingh/Datasets/KITTI_VO/00/image_2/{numFrame:06d}.png"
        currImage_c = cv2.imread(filename)
        currImage = cv2.cvtColor(currImage_c, cv2.COLOR_BGR2GRAY)

        status = [0] * len(prevFeatures)
        featureTracking(prevImage, currImage, prevFeatures, points2, status)

        E, mask = cv2.findEssentialMat(np.array(points2), np.array(prevFeatures), focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, np.array(points2), np.array(prevFeatures), focal=focal, pp=pp, mask=mask)

        scale = getAbsoluteScale(numFrame, 0, t[2])

        if scale > 0.1 and t[2] > t[0] and t[2] > t[1]:
            t_f += scale * np.dot(R_f, t)
            R_f = np.dot(R, R_f)

        if len(prevFeatures) < MIN_NUM_FEAT:
            featureDetection(prevImage, prevFeatures)
            points2 = [None] * len(prevFeatures)
            status = [0] * len(prevFeatures)
            featureTracking(prevImage, currImage, prevFeatures, points2, status)

        prevImage = currImage.copy()
        prevFeatures = points2.copy()

        x = int(t_f[0]) + 300
        y = int(t_f[2]) + 100
        cv2.circle(traj, (x, y), 1, (255, 0, 0), 2)

        cv2.rectangle(traj, (10, 30), (550, 50), (0, 0, 0), cv2.FILLED)
        text = f"Coordinates: x = {t_f[0]:.2f}m y = {t_f[1]:.2f}m z = {t_f[2]:.2f}m"
        cv2.putText(traj, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        cv2.imshow("Road facing camera", currImage_c)
        cv2.imshow("Trajectory", traj)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

