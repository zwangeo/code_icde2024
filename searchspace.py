# ADAPTER_CANDIDATES = ['skip', 'sequential', 'parallel']
# ADAPTER_CANDIDATES = ['skip', 'parallel']
# JK_CANDIDATES = ['concat', 'last', 'max', 'mean', 'att', 'gpr', 'lstm', 'node_adaptive']
# POOL_CANDIDATES = ['sum', 'mean', 'max', 'set2set', 'att', 'sort', 'gmt', 'gru', 'ds']


# ADAPTER_CANDIDATES = ['skip']
# JK_CANDIDATES = ['last']
# POOL_CANDIDATES = ['mean']
ADAPTER_CANDIDATES = ['skip', 'identity_sum', 'adapter_sum']
JK_CANDIDATES = ['concat', 'last', 'mean', 'att', 'gpr', 'lstm', 'node_adaptive']
POOL_CANDIDATES = ['sum', 'set2set', 'sort', 'gru', 'ds']
