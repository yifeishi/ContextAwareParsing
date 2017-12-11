#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import getopt

input_file = ''
output_file = ''

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'scn2scn.py -i <input_file> -o <output_file>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'scn2scn.py -i <input_file> -o <output_file>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
    print 'inputfile', input_file
    print 'outputfile', output_file

    xMax = -99999.0
    yMax = -99999.0
    zMax = -99999.0
    xMin = 99999.0
    yMin = 99999.0
    zMin = 99999.0
    file = open(input_file)
    while 1:
        line = file.readline()
        if not line:
            break
 #       print line
        L = line.split(' ')
        if L[0] == 'v':
            if xMax < float(L[1]):
                xMax = float(L[1])
            if yMax < float(L[2]):
                yMax = float(L[2])
            if zMax < float(L[3]):
                zMax = float(L[3])
            if xMin > float(L[1]):
                xMin = float(L[1])
            if yMin > float(L[2]):
                yMin = float(L[2])
            if zMin > float(L[3]):
                zMin = float(L[3])
    xLen = xMax-xMin
    yLen = yMax-yMin
    zLen = zMax-zMin
    maxLen = max(xLen,yLen,zLen)
    xCen = (xMax+xMin)/2
    yCen = (yMax+yMin)/2
    zCen = (zMax+zMin)/2
    print 'info: ',xLen,yLen,zLen,xCen,yCen,zCen,maxLen


    file = open(input_file)
    outfile = open(output_file,'w')
    while 1:
        line = file.readline()
        if not line:
            break
        L = line.split(' ')
        if L[0] == 'v':
            outfile.write(L[0])
            outfile.write(' ')
            outfile.write(str((float(L[1])-xCen)/maxLen))
            outfile.write(' ')
            outfile.write(str((float(L[2])-yCen)/maxLen))
            outfile.write(' ')
            outfile.write(str((float(L[3])-zCen)/maxLen))
            outfile.write('\n')
        else:
            outfile.write(line)

if __name__ == "__main__":
    main(sys.argv[1:])

