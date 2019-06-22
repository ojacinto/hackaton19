import os
import re


def iterate_files(rootDir="./train"):
    for dirName, subdirList, fileList in os.walk(rootDir):
        image, xml = "", ""
        for fname in fileList:
            # Processing name's file
            mo = re.search(".jpg$", fname)
            if mo:
               image = fname
               aux = fname.split('.')
               xml = aux[0]+".xml"
               mo = False
               print (""+ image, xml)

read_files()