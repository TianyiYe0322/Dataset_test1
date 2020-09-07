import os
from PIL import Image
from array import *
from random import shuffle
import sys

def changeFile(fileName):
    imageData = array('B')
    FileList = []
    Im = Image.open(fileName)
    pixel = Im.load()
    width,height = Im.size
    for x in range(0, width):
        for y in range(0, height):
            imageData.append(pixel[y,x])
    header = array('B')
    header.extend([0,0,8,1,0,0])
    header.append(int('0x' + '00', 16))#这两行是魔数的设置，格式要求
    header.append(int('0x' + '01', 16))#关于魔数的一些东西以后会补上
    if max([width,height]) <= 256:
        header.extend([0,0,0,width,0,0,0,height])
    else:
        raise ValueError('Image exceeds maximum size: 256x256 pixels')
    header[3] = 3
    imageData = header + imageData
    outPutFile = open(fileName.split('.')[0] + '.idx3-ubyte', 'wb')
    imageData.tofile(outPutFile)
    outPutFile.close()

if __name__ == '__main__':
    fileName = sys.argv[1]
    changeFile(fileName)