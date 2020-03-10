# import imutils import contours
import numpy as np
import argparse
import cv2
from . import myutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="path to input image")
ap.add_argument('-t', '--template', required=True, help='path to template OCR-A image')

args = vars(ap.parse_args())


def cv_show(source):
    cv2.imshow("", source)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread(args['template'])

ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# 只检测外圈轮廓， 只保留终点坐标
refCnts, hierachy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)

print(len(refCnts))
refCnts, round_rect = myutils.sort_contours(refCnts)

digits = {}
for (i, c) in enumerate(round_rect):
    (x, y, w, h) = c
    roi = ref[y:y+h, x:x+w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
image = cv2.imread(args['image'])
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

gradX = cv2.morphologyEx(top_hat, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

thresh_cnt, hierachy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = thresh_cnt
cur_image = image.copy()
cv2.drawContours(cur_image, cnts, -1, (0, 0, 255), 3)

locs = []
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if 2.5 < ar < 4.0:
        if 40 < w < 55 and 10 < h < 20:
            locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x: x[0])
output = []

for (i, (gx, gy, gw, gh)) in enumerate(locs):
    groupOutput = []
    group = gray[gy - 5: gy + gh, gx - 5: gx + gw + 5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = myutils.sort_contours(digitCnts, method="left-to-right")[0]
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        # 计算匹配得分
        scores = []

        # 在模板中计算每一个得分
        for (digit, digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))
    print(groupOutput)
    cv2.rectangle(image, (gx - 5, gy - 5),
                  (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gx, gy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # 得到结果
    output.extend(groupOutput)

print(output)
print("Credit Card # : {}".format(" ".join(output)))
cv_show(image)