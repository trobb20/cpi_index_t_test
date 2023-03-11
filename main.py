import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

score_tag = 'CPI score 2022'
country_tag = 'ISO3'
n_sources_tag = 'Number of sources'


def scores_sort_by_sources(df, cutoff=7):
    high_sources = df[df[n_sources_tag] > cutoff]
    low_sources = df[df[n_sources_tag] <= cutoff]

    high_sources_scores = high_sources[score_tag]
    low_sources_scores = low_sources[score_tag]

    return high_sources_scores, low_sources_scores


def t_test_scores_by_sources(high_sources, low_sources):
    statistic, pvalue = stats.ttest_ind(high_sources, low_sources, equal_var=False)
    #                                               Welch's T-Test assumes different variances
    return statistic, pvalue


def main():
    df = pd.read_excel(io='cpi_results_2022.xlsx')

    alpha = 0.05

    cutoffs = []
    T = []
    P = []

    for cutoff in range(min(df[n_sources_tag]), max(df[n_sources_tag])+1):
        print(f'Trying cutoff: {cutoff} sources.')

        high_sources_scores, low_sources_scores = scores_sort_by_sources(df, cutoff)

        t, p = t_test_scores_by_sources(high_sources_scores, low_sources_scores)

        success = alpha > p

        print(f'Result of t-test was: statistic: {t} p: {p}.')
        print(f'Test significant? {success}')

        if success:
            high_mean = np.mean(high_sources_scores)
            low_mean = np.mean(low_sources_scores)
            diff = round(high_mean-low_mean, 2)

            plt.figure()
            plt.title(f'Significant difference between high / low sources cutoff={cutoff} sources. \n'
                      f'Mean difference: {diff}')
            sns.histplot(high_sources_scores, kde=True)
            sns.histplot(low_sources_scores, kde=True)
            plt.axvline(high_mean, color='tab:blue', ls='dashed')
            plt.axvline(low_mean, color='tab:orange', ls='dashed')
            plt.legend([f'High sources > {cutoff}', f'Low sources <= {cutoff}'])
            plt.xlabel('CPI Score')
            plt.ylabel('Count')

        cutoffs.append(cutoff)
        T.append(t)
        P.append(p)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Cutoff Between High / Low Sources')
    ax1.set_ylabel('T-Value', color=color)
    ax1.plot(cutoffs, T, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('P-Value', color=color)  # we already handled the x-label with ax1
    ax2.plot(cutoffs, P, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(alpha, ls='dashed')

    plt.title('T-Test Results vs Cutoff on High/Low Sources')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == '__main__':
    main()
