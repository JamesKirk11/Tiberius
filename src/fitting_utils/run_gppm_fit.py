import sys
import os
import Tiberius

Tiberius_path = "/".join(sys.argv[0].split("/")[:-1])
starting_bin = int(sys.argv[1])
stopping_bin = int(sys.argv[2])

for i in range(starting_bin,stopping_bin):
	os.system("python %s/gppm_fit.py %d"%(Tiberius_path,i))

os.system("python %s/plot_output.py -s -st -cp"%Tiberius_path)
