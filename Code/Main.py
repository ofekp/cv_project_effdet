import time
from runMe import *
from busProjectTest import runTest
import sys
import os

class my_time:

    def tic(self):
        self.t = time.time()

    def toc(self):
        self.elapsed = float(time.time()) - float(self.t)
        s = "elapsed time is %0.3f seconds" % self.elapsed
        print(s)
        return self.elapsed

def main():
    i = 0
    saveDir = None
    while(i < len(sys.argv)):
        arg = sys.argv[i]
        print(arg)
        if('-myAnns' in arg):
            myAnnFileName = sys.argv[i + 1]
            myAnnFileName = os.path.join(os.getcwd(), myAnnFileName)
        if ('-anns' in arg):
            annFileNameGT = sys.argv[i + 1]
            annFileNameGT = os.path.join(os.getcwd(), annFileNameGT)
        if('-buses' in arg):
            busDir = sys.argv[i + 1]
            busDir = os.path.join(os.getcwd(), busDir)
        if ('-saveDir' in arg):
            saveDir = sys.argv[i + 1]
            saveDir = os.path.join(os.getcwd(), saveDir)
            if(not os.path.exists(saveDir)):
                os.mkdir(saveDir)
        if('-h' in arg):
            print('This script compares the ground truth (GT) and the estimated detections\nUsage: \n-myAnns <estimated annotations file>')
            print('-anns <real annotations file> \n-buses <directory of the training images>')
            print('-saveDir <output directory> use this option if you want to save the images with the annotations')
            print('example:\npython Main.py -myAnns myAnns.txt -anns anotationsTrain.txt -buses busesDir -saveDir output')
            print('-h - show this message and exit')
            sys.exit()
        i += 1

    if(i == 1):
        print('This script compares the ground truth (GT) and the estimated detections\nUsage: \n-myAnns <estimated annotations file>')
        print('-anns <real annotations file> \n-buses <directory of the training images>')
        print('-saveDir <output directory> use this option if you want to save the images with the annotations')
        print('example:\npython Main.py -myAnns myAnns.txt -anns anotationsTrain.txt -buses busesDir -saveDir output')
        print('-h - show this message and exit')
        sys.exit()

    t = my_time()

    t.tic()

    run(myAnnFileName, busDir)

    elapsed = t.toc()

    runTest(annFileNameGT, myAnnFileName, busDir, saveDir, elapsed)

if __name__ == '__main__':
    main()



