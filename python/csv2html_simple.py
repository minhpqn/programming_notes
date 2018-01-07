#!/usr/bin/python

# csv2html.py
# CSV to HTML Converter

import csv
import sys

table_indent_num = 2
tr_indent_num = 4
td_indent_num = 6
white_space = " "

def main():
    csv_reader = csv.reader(open(sys.argv[1]), delimiter='\t')
    table_indent = white_space * table_indent_num
    tr_indent = white_space * tr_indent_num
    td_indent = white_space * td_indent_num

    print '<meta charset="utf-8">'
    print table_indent + "<table border='1'>"
    for i, row in enumerate(csv_reader):
        print tr_indent + "<tr>"

        # Uncomment the following two lines if you don't
        # want a column for indexes
        if i == 0:
            print td_indent + "<th>#</th>"
        else: print td_indent + "<td>" + str(i) + "</td>"

        # Assume that the first line is a header
        for column in row:
            if i == 0:
                print td_indent + "<th>" + column + "</th>"
            else: print td_indent + "<td>" + column + "</td>"

        print tr_indent + "</tr>"

    print table_indent + "</table>"

if __name__ == "__main__":
    argn = len(sys.argv)
    if argn != 2:
        print "Usage: python csv2html.py <CSV file>"
        exit(1)
    main()
        
