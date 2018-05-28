#!/usr/bin/python
import sys

# Implementation of Fleiss Kappa
# References:
# [1] http://www.real-statistics.com/reliability/fleiss-kappa/
# [2] https://en.wikipedia.org/wiki/Fleiss'_kappa

# Given a matrix file in the format
#   Columns are categories
#   Rows are subjects
#   Each cell is filled with the number of raters who agreed that a certain subject belongs to a certain category
def read_matrix(filename):
    m = 0
    n = 0
    k = 0
    matrix = {}
    
    f = open(filename, 'rU')

    for line in f:
        line = line.strip()
        if line == '': continue
        cols = line.split()
        cols = map(int, cols)
        size = len(cols)
        val_sum = sum(cols)
        if m == 0:
            k = size
            n = val_sum
        else:
            if size != k:
                print('number of categories mismatched: %s:  %s ' % (k, line))
                sys.exit(1)
            if val_sum != n:
                print('number of raters mismatched: %s:  %s' % (n, line))

        for i in range(k):
            matrix[m, i] = cols[i]

        m += 1    

    f.close()
    
    return (m, n, k, matrix)

# Return fleiss_kappa score
# Given
#   m  number of subjects
#   n  number of raters
#   k  number of categories
#   matrix  matrix in which cell is the number of raters who agreed that a certain subject belongs to a certain category
def fleiss_kappa(m, n, k, matrix):
    p_col = []
    p_row = []

    for j in range(k):
        v = 0
        for i in range(m):
            v += matrix[i, j]
        v = 1.0 * v / (m*n)
        p_col.append(v)

    for i in range(m):
        v = 0
        for j in range(k):
            v += matrix[i, j] * (matrix[i, j] - 1)
        v = 1.0 * v / (n * (n-1))
        p_row.append(v)

    p = sum(p_row) / m
    pe = sum( map(lambda x: x*x, p_col) )

    k = (p - pe) / ( 1 - pe )

    return k
    
# Read data about judges of raters
# Each line contains judges of raters and delimited by tab characters
# Raters who do not rate will have '' values
def read_raw_data(filename):
    m = 0
    n = 0
    k = 0
    matrix = {}

    matrix1 = {}
    lbdict = {}

    f = open(filename, 'rU')
    for line in f:
        line = line.strip()
        if line == '': continue
        
        fields = line.split()
        n = len(fields)
        
        for lb in fields:
            if lb != '':
                lbdict[lb] = 1
                if matrix1.__contains__( (m, lb) ):
                    matrix1[m, lb] += 1
                else:
                    matrix1[m, lb] = 1

        m += 1
            
    f.close()

    labels = sorted(lbdict.keys())
    k      = len(labels)

    for i in range(m):
        for j in range(k):
            if  matrix1.__contains__( (i, labels[j]) ):
                matrix[i, j] = matrix1[i, labels[j]]
            else:
                matrix[i, j] = 0
                
    return (m, n, k, matrix)
    
def usage():
    print('usage: [--raw | --matrix] datafile')
    print(' --raw    use raw data')
    print(' --matrix use matrix input')
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
    m = 0
    n = 0
    k = 0
    if opt == '--matrix':
        (m, n, k, matrix) = read_matrix(datafile)
    else:
        (m, n, k, matrix) = read_raw_data(datafile)

    print('# Number of raters: %s' % n)
    print('# Number of subjects: %s' % m)
    print('# Number of categories: %s' % k)
    # print '# Matrix: %s' % repr(matrix)
    print()

    kappa = fleiss_kappa(m, n, k, matrix)
    print('Fleiss kappa: %s' % kappa)

if __name__ == '__main__':
    main()
