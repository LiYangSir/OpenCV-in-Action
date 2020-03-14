import cv2
import numpy as np


class Stitch:

    def stitch(self, images, ratio=0.75, repro_thresh=4.0, show_matches=False):
        (image_left, image_right) = images
        (kp_right, features_right) = self.detect_and_describe(image_right)
        (kp_left, features_left) = self.detect_and_describe(image_left)

        M = self.match_keypoints(kp_left, kp_right, features_left, features_right, ratio, repro_thresh)

        if M is None:
            return None

        matches, H, status = M

        result = cv2.warpPerspective(image_right, H, (image_right.shape[1] + image_left.shape[1], image_right.shape[0]))

        result[0:image_right.shape[0], 0:image_right.shape[1]] = image_left

        self.cv_show(result)

        return result

    def detect_and_describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        descriptor = cv2.xfeatures2d.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(image, None)

        kps = np.float32([kp.pt for kp in kps])

        return kps, features

    def match_keypoints(self, kp_left, kp_right, features_left, features_right, ratio, repro_thresh):

        matcher = cv2.BFMatcher()

        raw_matches = matcher.knnMatch(features_left, features_right, 2)

        matches = []

        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                print(m[0].trainIdx, m[0].queryIdx, m[1].trainIdx, m[1].queryIdx)
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            pts_right = np.float32([kp_left[i] for (_, i) in matches])
            pts_left = np.float32([kp_right[i] for (i, _) in matches])

            (H, status) = cv2.findHomography(pts_left, pts_right, cv2.RANSAC, repro_thresh)

            return matches, H, status
        else:
            return None

    def cv_show(self, src):
        cv2.imshow("", src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()