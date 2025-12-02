import sys
import os
import Tiberius
from global_utils import parseInput

Tiberius_path = "/".join(sys.argv[0].split("/")[:-1])
starting_bin = int(sys.argv[1])
stopping_bin = int(sys.argv[2])

input_dict = parseInput('fitting_input.txt')

if bool(int(input_dict['generate_LDCs'])):
	os.system("python %s/generate_LDCs.py %d"%(Tiberius_path,i))

for i in range(starting_bin,stopping_bin):
	os.system("python %s/light_curve_fit.py %d"%(Tiberius_path,i))

os.system("python %s/plot_output.py -s -st -cp"%Tiberius_path)
os.system("python %s/model_table_generator.py"%Tiberius_path)
try:
	os.system("mkdir tables")
	os.system("mkdir plots")
	os.system("mkdir pickled_objects")
except:
	pass
os.system("mv *.pickle pickled_objects/")
os.system("mv *.txt tables/")
os.system("mv *.png plots/")
os.system("mv *.pdf plots/")
