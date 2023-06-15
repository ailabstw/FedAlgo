import numpy as np
from bed_reader import open_bed

import gwasprs

# client

bfile_prefix = "/mnt/prsdata/Test/Data/DEMO_CLF/demo_hg38"
my_bed = open_bed(f"{bfile_prefix}.bed")

y = np.array(list(map(float, my_bed.pheno)))
genotype = my_bed.read(index=np.s_[:, list(range(1000))])

## pre-process

# X = gwasprs.concat((genotype, cov_values))
mask = gwasprs.get_mask(genotype)
X = gwasprs.impute_with(genotype) * mask
y = y * mask.reshape((mask.shape[0], ))

XtX1 = gwasprs.unnorm_autocovariance(X)
Xty1 = gwasprs.unnorm_covariance(X, y)


# server

XtXs = [XtX1, ]
Xtys = [Xty1, ]

server_XtX = sum(XtXs)
server_Xty = sum(Xtys)

server_model = gwasprs.LinearRegression(XtX=server_XtX, Xty=server_Xty)
beta = server_model.coef


# client

client_model = gwasprs.LinearRegression(beta=beta)
sse = client_model.sse(X, y)
obs1 = gwasprs.nonnan_count(genotype)


# server

obss = [obs1, ]

nobs = sum(obss)

dof = server_model.dof(nobs)
t_stats = server_model.t_stats(sse, server_XtX, dof)
pvals = gwasprs.t_dist_pvalue(t_stats, dof)
