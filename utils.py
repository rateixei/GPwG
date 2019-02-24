import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import pchip
from scipy.optimize import curve_fit
from scipy.optimize import minimize


def gp_fit(gp, x, y, yerr, x_pred, verbose=1):

    if verbose:  print('Predicting...')
    pred, pred_var = gp.predict(y, x_pred, return_var=True)

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    if verbose: print("Initial ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))

    if verbose: print('Minimizing...')
    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    if verbose: print(result)

    gp.set_parameter_vector(result.x)
    if verbose: print("\nFinal ln-likelihood: {0:.2f}".format( gp.log_likelihood(y) ))

    return gp.log_likelihood(y), result

def llr_plot(llr, mus, out):
    my_llr = -2*(np.array(llr)-max(llr))
    plt.figure(figsize=(8,6))
    plt.errorbar(mus, my_llr, fmt = '.k', marker='o', markersize=5)
    plt.xlabel('Signal strength')
    plt.ylabel('-2*Log-likelihood ratio')
    plt.savefig(out+'_llr.pdf')

def gaussian(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def make_gp_plot(x, y, yerr, ytruth, x_pred, pred, pred_var, my_sig, out):

    interpolated_pred = pchip(x_pred, pred)
    interpolated_pred_err = pchip(x_pred, pred_var)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex='row',
                               gridspec_kw={'height_ratios': [3, 1, 1], 'hspace': 0.025},
                               figsize=(8, 6) )
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax0.tick_params(axis='x', direction='in')
    ax1.tick_params(axis='x', direction='in')
    ax2.tick_params(axis='x', direction='in')
    ax0.tick_params(axis='y', direction='in')
    ax1.tick_params(axis='y', direction='in')
    ax2.tick_params(axis='y', direction='in')

    ax0.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                    color="k", alpha=0.2, label='GP Std. deviation')
    ax0.plot(x_pred, pred, "k", lw=1.5, alpha=0.5, label='GP mean')
    ax0.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0, label='Toy data')
    ax0.plot(x, ytruth, "g-", label='Truth')
    ax0.legend(frameon=False)
    ax0.set_ylabel("Observations")

    # ax1.fill_between(x_pred, 1 - np.sqrt(pred_var)/pred, 1+np.sqrt(pred_var)/pred, color="k", alpha=0.2  )
    # ax1.errorbar( x, y/interpolated_pred(x), yerr=yerr/interpolated_pred(x), fmt=".k", capsize=0 )
    # ax1.set_ylabel('Ratio')

    ax1.fill_between(x_pred, 0 - np.sqrt(pred_var), np.sqrt(pred_var), color="k", alpha=0.2  )
    ax1.plot(x, my_sig, "r-", label='Fitted signal')
    ax1.errorbar( x, y - interpolated_pred(x), yerr=yerr, fmt=".k", capsize=0 )
    ax1.set_ylabel(r'$y - GP$')
    ax1.legend(frameon=False)

    uncert_tot = np.sqrt( interpolated_pred_err(x) + yerr**2 )
    pulls = (y - interpolated_pred(x))/uncert_tot
    ax2.errorbar( x, pulls, xerr=(x[1]-x[0])/2., fmt='r-' )
    ax2.plot([min(x), max(x)], [-1,-1], color='gray', linestyle='dashed')
    ax2.plot([min(x), max(x)], [1,1], color='gray', linestyle='dashed')
    ax2.set_ylabel(r'$\frac{y - GP}{\sqrt{\sigma_{y}^{2} + \sigma_{GP}^{2} }}$')
    ax2.set_xlabel("Observable")

    fig.savefig(out + '.pdf')

    plt.figure(figsize=(8,6))
    ns, bins, _ = plt.hist(pulls, bins=20, range=(-3,3), histtype='step')
    p0 = [1., 0., 1.]
    bin_cents = 0.5*(bins[:-1] + bins[1:])
    coeff, var_matrix = curve_fit(gaussian, bin_cents, ns, p0=p0)
    gauss_x = np.linspace(-3,3, 500)
    hist_fit = gaussian(gauss_x, *coeff)
    perr = np.sqrt(np.diag(var_matrix))
    my_legend = r'$\mu = ${:.3f} $\pm$ {:.3f}'.format(coeff[1], perr[1])
    my_legend += '\n'
    my_legend += r'$\sigma = ${:.3f} $\pm$ {:.3f}'.format(coeff[2], perr[2])
    plt.plot(gauss_x, hist_fit, color='red', linestyle='dashed', label=my_legend)
    plt.xlabel(r'$\frac{y - GP}{\sqrt{\sigma_{y}^{2} + \sigma_{GP}^{2} }}$')
    plt.ylabel('Bins')
    plt.legend(frameon=False)

    plt.savefig(out+'_pull.pdf')
