import numpy as np
import operator


def createDataSet():
    # 19组二维特征
    features = np.array(
        [[700, 500], [750, 460], [1000, 230], [800, 420], [690, 500], [730, 490], [60, 1200], [80, 1100], [1260, 200],
         [240, 1200], [200, 1090], [1200, 200], [900, 320], [97, 1170], [118, 1108], [1203, 200], [1180, 88],
         [1111, 99], [67, 1222]])
    # 19组特征的标签
    labels = ['多云', '多云', '晴', '多云', '多云', '多云', '雨', '雨', '晴', '雨', '雨', '晴', '晴', '雨', '雨', '晴',
              '晴', '晴', '雨']
    return features, labels


features, labels = createDataSet()
print(features, '\n', labels)


def classify0(inX, dataSet, labels, k):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后取绝对值
    sqDiffMat = abs(diffMat)
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = sqDistances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


features, labels = createDataSet()
test = [800, 400]
test_label = '多云'
for k in range(1, 20):
    test_class = classify0(test, features, labels, k)
    print("预测类别：", test_class, "真实类别：", test_label, "k值:", k)
