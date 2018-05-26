#!/usr/local/bin/python2.7
import sys
import q4
import q5

if __name__ == "__main__":
    if sys.argv[1] == "q4":
        rare_maker = q4.Rare_Maker(sys.argv[2], sys.argv[3])
        rare_maker.getCount()
        rare_maker.replace()
    elif sys.argv[1] == "q5" or sys.argv[1] == "q6":
        cky = q5.CKY(sys.argv[2], sys.argv[3], sys.argv[4])
        cky.get_parameters()
        cky.dp_and_write_output()








