#!/usr/bin/python
from PIL import Image
import os
import numpy
resize = (20,20)

read_path = "Img/Sample%03d/img%03d-%03d.png"
write_path = "Downscaled/Sample%03d/img%03d-%03d.png"
dataset = numpy.empty([1,417])
dataset_grayscale = numpy.empty([1,417])
num_alphabets = 17

output_map = numpy.identity(num_alphabets)
numpy.set_printoptions(suppress=True)

for letter in range(1,17):
        for sample in range(1,26):
                #print(img_path%(letter,letter,sample))
                image = Image.open(read_path%(letter,letter,sample))
                image = image.resize(resize, Image.ANTIALIAS)
                try:
                        os.stat("Downscaled/Sample%03d"%letter)
                except:
                        os.mkdir("Downscaled/Sample%03d"%letter)
                image.save(write_path%(letter,letter,sample),"PNG")
                pixels = image.load()
                line = numpy.empty([0,0])
                line_grayscale = numpy.empty([0,0])
                for i in range(0,20):
                        for j in range(0,20):
                                grayscale_value = 0.299*pixels[i,j][0] + 0.587*pixels[i,j][1] + 0.114*pixels[i,j][2]

                                if grayscale_value > 200:
                                    line = numpy.append(line,1)
                                else:
                                    line = numpy.append(line,0)
                                line_grayscale = numpy.append(line_grayscale,grayscale_value)
                line = numpy.append(line,output_map[letter])
                line_grayscale = numpy.append(line_grayscale,output_map[letter])
                #print dataset
                #print line
                dataset = numpy.vstack([dataset,line])
                dataset_grayscale = numpy.vstack([dataset_grayscale,line_grayscale])
                print "Completed a sample"
numpy.savetxt("kannada.csv",dataset,delimiter=",",fmt="%5.2f")
numpy.savetxt("kannada_grayscale.csv",dataset_grayscale,delimiter=",",fmt="%5.2f")
