# 导入所需模块
import cv2
from matplotlib import pyplot as plt
import os


# 显示图片
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# plt显示彩色图片
def plt_show0(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()


# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z',
            '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁',
            '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']

# 读取一个文件夹下的所有图片，输入参数是文件名
def read_directory(directory_name):
    referImg_list = []
    for filename in os.listdir(directory_name):
        # print(filename)  # 仅仅是为了测试
        # img = cv2.imread(directory_name + "/" + filename)
        referImg_list.append(directory_name + "/" + filename)

    return referImg_list

# 匹配中文
c_words = []
for i in range(34,64):
    c_word = read_directory('./refer1/'+ template[i])
    c_words.append(c_word)
c_words[1]

# 读取一个车牌字符
img = cv2.imread('./words/test4_1.png')
plt_show0(img)

# 灰度处理，二值化
# 高斯去噪
image = cv2.GaussianBlur(img, (3, 3), 0)
# 灰度处理
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt_show(gray_image)

# 自适应阈值处理
ret, image_ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
plt_show(image_)

import numpy as np

best_score = []
for c_word in c_words:
    score = []
    for word in c_word:
        #         print(word)
        # fromfile()函数读回数据时需要用户指定元素类型，并对数组的形状进行适当的修改
        template_img = cv2.imdecode(np.fromfile(word, dtype=np.uint8), 1)
        #         template_img = cv2.imread(word)
        #         print(template_img)
        #         cv_show('template_img',template_img)
        #         template_img = np.float32(template_img)
        #         plt_show0(template_img)
        #         print(word)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
        ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)

        height, width = template_img.shape
        image = image_.copy()
        image = cv2.resize(image, (width, height))
        result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF)
        score.append(result[0][0])
    best_score.append(max(score))

print(best_score)
print(max(best_score))
print(best_score.index(max(best_score)))
print(template[34])
print(template[34 + 27])

'''score = []
best_template = None
best_refer = None
referImg_list = read_directory(img)
for refer in referImg_list:
        #refer = refer[0]
        #print(refer)
    template = cv2.imread(refer)

    template = cv2.GaussianBlur(template, (3, 3), 0)
    # 灰度处理
    template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    # 自适应阈值处理
    ret, template = cv2.threshold(template, 0, 255, cv2.THRESH_OTSU)

    height, width = template.shape
    image = image_.copy()
    image = cv2.resize(image, (width, height))  # 和模板一致
    #     plt_show(image)
    # TM_SQDIFF TM_CCOEFF
    # TM_SQDIFF：计算平方不同，计算出来的值越小，越相关
    # TM_CCORR：计算相关性，计算出来的值越大，越相关
    # TM_CCOEFF：计算相关系数，计算出来的值越大，越相关
    # TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
    # TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
    # TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    #     print(score)
    if score != [] and result[0][0] > max(score):
        best_template = None
        best_refer = None
        best_template = template
        best_refer = refer
    score.append(result[0][0])

print(score)
plt_show(best_template)
print(max(score))
print(best_refer)
'''
# 读取一个车牌字符
img = cv2.imread('./words/test1_5.png')
plt_show0(img)

# 灰度处理，二值化
# 高斯去噪
image = cv2.GaussianBlur(img, (3, 3), 0)
# 灰度处理
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt_show(gray_image)

# 自适应阈值处理
ret, image_ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
plt_show(image_)

# 字母模板列表
c_words = []
for i in range(10,34):
    c_word = read_directory('./refer1/'+ template[i])
    c_words.append(c_word)
c_words

import numpy as np
best_score = []
for c_word in c_words:
    score = []
    for word in c_word:
        # fromfile()函数读回数据时需要用户指定元素类型，并对数组的形状进行适当的修改
        template_img=cv2.imdecode(np.fromfile(word,dtype=np.uint8),1)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
        ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
        height, width = template_img.shape
        image = image_.copy()
        image = cv2.resize(image, (width, height))
        result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF)
        score.append(result[0][0])
    best_score.append(max(score))

print(best_score)
print(max(best_score))
print(best_score.index(max(best_score)))
print(template[10])
print(template[10+16])

# 读取一个车牌字符
img = cv2.imread('./words/test1_4.png')
plt_show0(img)

# 灰度处理，二值化
# 高斯去噪
image = cv2.GaussianBlur(img, (3, 3), 0)
# 灰度处理
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt_show(gray_image)

# 自适应阈值处理
ret, image_ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
plt_show(image_)

# 字母数字模板列表
c_words = []
for i in range(0,34):
    c_word = read_directory('./refer1/'+ template[i])
    c_words.append(c_word)

import numpy as np
best_score = []
for c_word in c_words:
    score = []
    for word in c_word:
        # fromfile()函数读回数据时需要用户指定元素类型，并对数组的形状进行适当的修改
        template_img=cv2.imdecode(np.fromfile(word,dtype=np.uint8),1)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
        ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
        height, width = template_img.shape
        image = image_.copy()
        image = cv2.resize(image, (width, height))
        result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF)
        score.append(result[0][0])
    best_score.append(max(score))

print(best_score)
print(max(best_score))
print(best_score.index(max(best_score)))
print(template[0])
print(template[0+4])