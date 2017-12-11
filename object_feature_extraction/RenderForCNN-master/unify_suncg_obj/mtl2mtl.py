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
        print 'mtl2mtl.py -i <input_file> -o <output_file>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'mtl2mtl.py -i <input_file> -o <output_file>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
    print 'inputfile', input_file
    print 'outputfile', output_file

    os.system('cp %s %s' % (input_file, output_file))

if __name__ == "__main__":
    main(sys.argv[1:])

