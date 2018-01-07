#!/usr/bin/python
import sys

# Calculate Cohen's kappa
# Kappa statistics for measuring inter-rater agreement
# References:
# [1] https://onlinecourses.science.psu.edu/stat509/node/162
# [2] http://www.real-statistics.com/reliability/cohens-kappa/
# [3] https://en.wikipedia.org/wiki/Cohen's_kappa
# [4] http://virtualhost.cs.columbia.edu/~julia/courses/CS6998/Interrater_agreement.Kappa_statistic.pdf

#
# Description
# ------------------------------
# - Given raw data containing rows, each row contains
# labels judged by two raters (categorical variables)
# Return the kappa statistics
# - The program can accept the raw data in the form of contigency table

# Return a matrix (use dictionary to store matrix)
# Given the data file in the format
# rater1 rater2 num_cases
# E.g.,
# A A 21
def read_matrix(datafile):
    matrix1 = {}
    lbdict = {}
    num_labels = 0

    f = open(datafile, 'rU')
    for line in f:
        line = line.strip()
        if line == '': continue
        fields = line.split()
        lbdict[fields[0]] = 1
        lbdict[fields[1]] = 1
        matrix1[fields[0], fields[1]] = int(fields[2])
        
    f.close()

    labels = sorted(lbdict.keys())
    num_labels = len(labels)

    matrix = {}
    for i in range(num_labels):
        for j in range(num_labels):
            if  matrix1.has_key( (labels[i], labels[j]) ):
                matrix[i, j] = matrix1[labels[i], labels[j]]
            else:
                matrix[i, j] = 0

    return (num_labels, matrix)

def read_raw_data(datafile):
    matrix1 = {}
    lbdict = {}
    num_labels = 0
    matrix = {}
    
    f = open(datafile, 'rU')
    
    for line in f:
        line = line.strip()
        if line == '': continue
        
        fields = line.split()
        lbdict[fields[0]] = 1
        lbdict[fields[1]] = 1
        
        if matrix1.has_key( (fields[0], fields[1]) ):
            matrix1[fields[0], fields[1]] += 1
        else:
            matrix1[fields[0], fields[1]] = 1
            
    f.close()

    labels = sorted(lbdict.keys())
    num_labels = len(labels)

    matrix = {}
    for i in range(num_labels):
        for j in range(num_labels):
            if  matrix1.has_key( (labels[i], labels[j]) ):
                matrix[i, j] = matrix1[labels[i], labels[j]]
            else:
                matrix[i, j] = 0

    return (num_labels, matrix)
    

def cohen_kappa(num_labels, matrix):
    n = 0
    pa = 0.0
    rows = []
    cols = []
    for i in range(num_labels):
        rows.append(0)
        cols.append(0)
        
    for i in range(num_labels):
        pa += matrix[i, i]
        for j in range(num_labels):

            n += matrix[i, j]
            rows[i] += matrix[i, j]
            cols[j] += matrix[i, j]

    pa = pa/n
    pe = 0.0
    for i in range(num_labels):
        pe += rows[i] * cols[i]

    pe = pe/(n ** 2)

    return (pa - pe)/(1 - pe)

def usage():
    print 'usage: [--raw | --matrix] datafile'
    print ' --raw    use raw data'
    print ' --matrix use matrix (contingency table)'
    sys.exit(1)

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        usage()

    opt = args[0]
    datafile = args[1]

    if opt != '--raw' and opt != '--matrix':
        usage()

    matrix = {}
    num_labels = 0
    if opt == '--matrix':
        (num_labels, matrix) = read_matrix(datafile)
    else:
        (num_labels, matrix) = read_raw_data(datafile)

    kappa = cohen_kappa(num_labels, matrix)

    print 'Cohen\'s Kappa Coefficient: %s' % kappa
        

if __name__ == '__main__':
    main()



