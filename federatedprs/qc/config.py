VERSION = "0.0.1"

# MOUDLE
STEP_MODULE="steps"


# LOG
LOG_FORMAT = '%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d| %(message)s'
EVENT_LIST = ['trainInitDoneEvent', 'trainStartedEvent', 'trainFinishedEvent' ]


# LINALG operation
BATCH_SNP_NUM = 10000
CLIENT_CPU_CORES = 2


# weight name 
WEIGHT_NAME = "weights.ckpt"
INFO_NAME = "info.json"



# LOGISTIC REGRESSION
delta_log_likelihood_threshold = 0.0001
log_min_iter_num = 2
log_max_iter_num = 15





# QC SETTING
PLINK2="plink2"
# Variant filter
MAF_QC=0.05
GENO_QC=0.02
HWE_QC=5e-7 
# Individual filter
MIND_QC=0.02
HETER_SD=3
KINGCUTOFF=0.0442
HET_RANGE=(-0.5, 0.5)
HET_BIN=1000
# LD prune
PRUNE_WINDOW=50
PRUNE_STEP=5
PRUNE_THRESHOLD=0.2

stepEvent="stepEvent"


