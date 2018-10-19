import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import ProbPlot
import CAPMmodel


def plot_diagnostics(asset)
    model = run_model(asset)
    results = model.fit()
    plt.style.use('ggplot')
    plt.scatter(X,y,c='b')
    plt.plot(X,results.predict(),'r--')
    plt.ylabel('Bitcoin % Change')
    plt.savefig('BTCpc.png')
    plot_acf(y) #deprecated since use
    plt.title('Bitcoin % Change Autocorrelation')
    plt.savefig('BTCacfpc.png')
    plt.style.use('seaborn') # pretty matplotlib plots
    plt.rc('font', size=14)
    plt.rc('figure', titlesize=18)
    plt.rc('axes', labelsize=15)
    plt.rc('axes', titlesize=18)
    # fitted values (need a constant term for intercept)
    model_fitted_y = results.fittedvalues
    # model residuals
    model_residuals = results.resid
    # normalized residuals
    model_norm_residuals = results.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = results.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = results.get_influence().cooks_distance[0]
    plot_lm_1 = plt.figure(1)
    plot_lm_1.set_figheight(8)
    plot_lm_1.set_figwidth(12)
    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'Bit', data=clean_monthly_returns,
                            lowess=True,
                            scatter_kws={'alpha': 0.5},
                            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_lm_1.axes[0].set_title('Bitcoin')
    plot_lm_1.axes[0].set_xlabel('Fitted Values')
    plot_lm_1.axes[0].set_ylabel('Residuals')
    # annotations
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_7 = abs_resid[:7]
    for i in abs_resid_top_7.index:
        plot_lm_1.axes[0].annotate(i,
                                xy=(model_fitted_y[i],
                                    model_residuals[i]));
    plt.savefig('BTCfitres.png')
    QQ = ProbPlot(model_norm_residuals)
    plot_qq_1 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
    plot_qq_1.set_figheight(8)
    plot_qq_1.set_figwidth(12)
    plot_qq_1.axes[0].set_title('Normal Q-Q')
    plot_qq_1.axes[0].set_xlabel('Theoretical Quantiles')
    plot_qq_1.axes[0].set_ylabel('Standardized Residuals');
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_4 = abs_norm_resid[:4]
    for r, i in enumerate(abs_norm_resid_top_4):
        plot_qq_1.axes[0].annotate(i,
                                xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                    model_norm_residuals[i]));
    plt.savefig('BTCqq.png')

    plot_cd_1 = plt.figure(4)
    plot_cd_1.set_figheight(8)
    plot_cd_1.set_figwidth(12)
    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_cd_1.axes[0].set_xlim(0, 0.35)
    plot_cd_1.axes[0].set_ylim(-3, 5)
    plot_cd_1.axes[0].set_title('Residuals vs Leverage')
    plot_cd_1.axes[0].set_xlabel('Leverage')
    plot_cd_1.axes[0].set_ylabel('Standardized Residuals')
    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
    for i in leverage_top_3:
        plot_cd_1.axes[0].annotate(i,
                                xy=(model_leverage[i],
                                    model_norm_residuals[i]))
    plt.savefig('BTClev.png')

    return 
