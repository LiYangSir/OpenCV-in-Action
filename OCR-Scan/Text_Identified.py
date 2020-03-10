# 下载地址：https://digi.bib.uni-mannheim.de/tesseract/
# 配置环境变量path中添加安装路径，例如E:\***\Tesseract-OCR
# tesseract -v进行验证是否安装成功
# tesseract XXX.png 得到结果
# pip install pytesseract
# tesseract_cmd 修改为绝对路径即可
from PIL import Image
import pytesseract
import cv2
import os

preprocess = 'blur'  # thresh

image = cv2.imread('output/src.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

if preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename))

# 打印输出
print(text)
os.remove(filename)
