import timeit

import numpy as np
import jax.numpy as jnp
import jax.random as random
import pandas as pd
import seaborn as sns
from scipy.stats import norm

import gwasprs

class time_batch_matrix_calculation:

    def __init__(self, n_samples=200, n_SNPs=20000, n_batches=[2,4,5,10], n_repeat=5, times=10):
        self.X = gwasprs.array.simulate_genotype_matrix(random.PRNGKey(32), (n_samples, n_SNPs), impute=True, standardize=True)
        self.binarize_2d = lambda batch: list(map(lambda x: 1 if x > 0.5 else 0, batch))
        self.batch_sample = {b:int(n_samples/b) for b in n_batches}
        self.batch_snp = {b:int(n_SNPs/b) for b in n_batches}

        self.n_repeat, self.times = n_repeat, times

    def main(self, to_df=True):
        self.batch_sample_time_dist = self.flow(batch_on='sample')
        self.batch_snp_time_dist = self.flow(batch_on='snp')
        if to_df:
            self.to_df()
            return self.result_df
        else:
            return self.batch_sample_time_dist, self.batch_snp_time_dist
        
    def flow(self, batch_on):
        time_dist = {}
        for n_batch in self.batch_sample.keys():
            
            if batch_on == 'sample':
                x, beta, y = self.get_batch_sample_params(n_batch)
            elif batch_on == 'snp':
                x, beta, y = self.get_batch_snp_params(n_batch)
            self.model = gwasprs.regression.BatchedLogisticRegression(beta)

            time_dist.setdefault(
                n_batch,
                {
                    'predict_single':self.test_predict_single(x),
                    'predict_pmap':self.test_predict_pmap(x),
                    'gradient':self.test_gradient(x, y),
                    'hessian':self.test_hessian(x),
                    'loglikelihood':self.test_loglikelihood(x, y),
                    # doesn't perform because of unexpected error and tremendous time cost
                    # 'fit':self.test_fit(x, y)
                    # 'beta':self.test_beta(x, y)
                }
            )
        return time_dist

    def test_predict_single(self, x):
        return self._timeit(lambda: self.model.predict(x))

    def test_predict_pmap(self, x):
        return self._timeit(lambda: self.model.predict(x, acceleration='pmap'))

    def test_gradient(self, x, y):
        return self._timeit(lambda: self.model.gradient(x, y))

    def test_hessian(self, x):
        return self._timeit(lambda: self.model.hessian(x))

    def test_loglikelihood(self, x, y):
        return self._timeit(lambda: self.model.loglikelihood(x, y))

    def test_fit(self, x, y):
        return self._timeit(lambda: self.model.fit(x, y))

    def test_beta(self, x, y):
        return self._timeit(lambda: self.model.beta(self.model.gradient(x, y), self.model.hessian(x)))

    def get_batch_sample_params(self, n_batch):
        x = np.reshape(self.X, (n_batch, self.batch_sample[n_batch], -1))
        
        beta = jnp.zeros((n_batch, x.shape[2]))
        
        z = gwasprs.linalg.batched_mvmul(x, beta)
        pred_y = norm.cdf(z - np.mean(z) + np.random.randn(n_batch, self.batch_sample[n_batch]))
        y = np.array(list(map(self.binarize_2d, pred_y)))

        return x, beta, y

    def get_batch_snp_params(self, n_batch):
        x = np.reshape(self.X, (n_batch, -1, self.batch_snp[n_batch]))

        beta = jnp.zeros((n_batch, x.shape[2]))

        z = gwasprs.linalg.batched_mvmul(x, beta)
        pred_y = norm.cdf(z - np.mean(z) + np.random.randn(n_batch, x.shape[1]))
        y = np.array(list(map(self.binarize_2d, pred_y)))

        return x, beta, y

    def to_df(self):
        sample_df = self._concat_batch_df(self.batch_sample_time_dist)
        sample_df['batch on'] = ['sample']*len(sample_df)
        snp_df = self._concat_batch_df(self.batch_snp_time_dist)
        snp_df['batch on'] = ['snp']*len(snp_df)
        result_df = pd.concat([sample_df, snp_df])
        self.result_df = result_df.melt(id_vars=['n_batch', 'batch on'], value_vars=result_df.columns[0:5], var_name='function', value_name='time')

    def to_fig(self):
        g = sns.lmplot(
            data=self.result_df, x="n_batch", y="time", col="function", hue="batch on",
            col_wrap=2, palette="muted", ci=None,
            height=4, scatter_kws={"s": 50, "alpha": 1}, sharey=False
        )
        g.savefig('time_cost.png', dpi=300)
        return g
        
    def _concat_batch_df(self, time_dist):
        concat_df = pd.DataFrame()
        for n_batch in time_dist.keys():
            batch_df = pd.DataFrame.from_dict(time_dist[n_batch])/self.times
            batch_df['n_batch'] = [n_batch]*len(batch_df)
            concat_df = pd.concat([concat_df, batch_df], axis=0)
        return concat_df
            
    def _timeit(self, func):
        return timeit.repeat(func, repeat=self.n_repeat, number=self.times)

    
if __name__ == '__main__':
    TEST = time_batch_matrix_calculation()
    result = TEST.main()
    result.to_csv('time_batch_matrix_calculation.csv', index=None)
    TEST.to_fig()

"""
TEST = time_batch_matrix_calculation()

Usage:
    The hyperparameters in n_batches should be able to divide either the number of samples or the number of SNPs.

    # get batched params
    x, beta, y = TEST.get_batch_sample_params(n_batch)

    # get batched params
    x, beta, y = TEST.get_batch_snp_params(n_batch)

    # run time evaluation of functions
    TEST.flow(batch_on='sample')
    TEST.flow(batch_on='snp')

    # convert to dataframe
    TEST.to_df()

    # convert to figure
    TEST.to_fig()

    The final conclusion is that we should batch on the snp aixs,
    otherwise, the hessian calculation will take a lot of time.
"""