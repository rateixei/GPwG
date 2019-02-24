import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import argparse
import json
import george
from george import kernels
from matplotlib import gridspec
import utils as ut
import sys

parser = argparse.ArgumentParser(description='Gaussian process fit')
parser.add_argument('-i', dest='inf', type=str, required=True, help='Input data file')
parser.add_argument('-o', dest='out', type=str, required=True, help='Output file')

args = parser.parse_args()

print('Reading data')
data = json.load( open( args.inf, 'r') )

ytruth = np.array(data['bkg_model'])
x = np.array(data['x'])
y_toys = data['toys']
if 'sig_model' in data:
    sig_model = np.array(data['sig_model'])

x_pred = np.linspace(min(x), max(x), 500)
y = np.array(y_toys[0])
yerr = np.sqrt(y)

print('Defining GP')
# kernel = np.var(y) * kernels.ExpSquaredKernel(0.5)
kernel = np.var(y) * kernels.Matern52Kernel(0.5)
# kernel = np.var(y) * kernels.ExpKernel(10)
gp = george.GP(kernel)
gp.compute(x, yerr)

sig_mus = np.linspace(-5, 6, 500)
my_llrs = []
my_max_llr = -1000000
max_llr_result = 0
fitted_y = y
fitted_mu = -10

for mu in sig_mus:
    y_minus_sig = y - mu*sig_model
    llr, res = ut.gp_fit(gp, x, y_minus_sig, yerr, x_pred,0)
    my_llrs.append(llr)
    if llr > my_max_llr:
        my_max_llr = llr
        max_llr_result = res
        fitted_y = y_minus_sig
        fitted_mu = mu

    # print(mu, llr)
print('Min -log likelihood mu =', fitted_mu)
ut.llr_plot(my_llrs, sig_mus, args.out)
# sys.exit()
gp.set_parameter_vector(max_llr_result.x)

print('Predicting w/ minimum NLL model')
pred, pred_var = gp.predict(fitted_y, x_pred, return_var=True)

print('Making plot...')
ut.make_gp_plot(x, y, yerr, ytruth, x_pred, pred, pred_var, fitted_mu*sig_model, args.out)

print('Done!')
