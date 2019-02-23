import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import argparse
import json

parser = argparse.ArgumentParser(description='Make pseudo data')
parser.add_argument('-o', dest='out', type=str, required=True, help='Output file')
parser.add_argument('--xmin', dest='xmin', type=float, default=0, help='Min range')
parser.add_argument('--xmax', dest='xmax', type=float, default=1, help='Max range')
parser.add_argument('--bins', dest='bins', type=int, default=50, help='Number of bins')
parser.add_argument('--norm', dest='norm', type=int, default=10000, help='Number of bins')
parser.add_argument('--exp', dest='exp', type=float, default=-1, help='Exponential argument')
parser.add_argument('--toys', dest='toys', type=int, default=1, help='Number of toys to be saved')
parser.add_argument('--addsignal', dest='sig', type=int, default=0, help='Add signal with this normalization')
parser.add_argument('--sigmean', dest='sig_mean', type=float, default=0.5, help='Gaussian signal mean')
parser.add_argument('--sigwidth', dest='sig_width', type=float, default=0.05, help='Gaussian signal sigma')

args = parser.parse_args()

def gauss(x, p):
    mu, sigma = p
    return np.exp(-(x-mu)**2/(2.*sigma**2))

print('Making x axis')
x = np.linspace(args.xmin, args.xmax, args.bins)
bin_size = float(args.xmax - args.xmin)/float(2*args.bins )

print('Making truth exponential')
my_exp_truth = (args.norm*np.exp( args.exp*x )).tolist()
my_mod_truth = my_exp_truth[:]

if args.sig > 0:
    print('Making truth signal')
    my_sig_truth = gauss(x, (args.sig_mean, args.sig_width) )
    my_sig_truth = (args.sig*my_sig_truth/np.sum(my_sig_truth)).tolist()
    print('Signal integral', np.sum(my_sig_truth) )
    my_mod_truth = [ i+j for i,j in zip(my_mod_truth,my_sig_truth ) ]

my_toys = []
my_errs = []

print('Making toys')
for nt in range(args.toys):
    if nt%50 == 0: print("Making toy", nt)
    my_toys.append( np.random.poisson( my_mod_truth ).tolist() )
    my_errs.append( np.sqrt( my_toys[nt] ).tolist() )

print("Plotting...")
plt.figure()
plt.plot(x, my_mod_truth, 'b', label='Model', zorder=1)
plt.plot(x, my_exp_truth, 'g-.', label='Background model', zorder=3)
if args.sig > 0: plt.plot(x, my_sig_truth, 'r--', label='Signal model', zorder=5)
for it,nt in enumerate(my_toys):
    plt.errorbar( x, nt, xerr=bin_size, yerr=my_errs[it], color='k', fmt = 'none', marker='o', zorder=10, label='Pseudo-data')
    break
plt.legend(frameon=False)
plt.xlabel("Observable")
plt.ylabel("Observations")
plt.savefig(args.out + '.pdf')

to_json = {}
to_json['bkg_model'] = my_exp_truth
if args.sig > 0: to_json['sig_model'] = my_sig_truth
to_json['toys'] = my_toys

print('Saving data to json file')
json_file = open(args.out+'.json', 'w')
json.dump(to_json, json_file)
json_file.close()

print('Done!')
