#### Author of this code: James Kirk
#### Contact: jameskirk@live.co.uk

import numpy as np

def parseInput(file):
    """Function to parse the input parameters from an input file, e.g. fitting_input.txt.

    Input: file to be read

    Returns: dictionary of parameter names:parameter values

    """
    try:
        blob = np.loadtxt(file,dtype=str,delimiter='\n')
    except:
        blob = reader(file)

    input_dict = {}

    for line in blob:
        ignore_comments = line.split("#")[0]
        k,v = ignore_comments.split('=')
        input_dict[k.strip()] = v.strip().replace("\n","").replace("\t","")

    for k,v in zip(input_dict.keys(),input_dict.values()):
        if v == '':
            input_dict[k] = None

    return input_dict


def reader(file):
    """A function that can also read in the inputs, if np.loadtxt doesn't work with the specified delimiter"""
    with open(file) as f:
        lines = f.readlines()
    blob = []
    for i in lines:
        if i[0] == "#" or i[0] == "\n":
            pass
        else:
            blob.append(i)
    return np.array(blob)
