import numpy as np
from datetime import datetime as dt
from sklearn.decomposition import PCA

N = 1000 # TODO Is this correct M and N?
M = 100
COMPONENTS = 2
start_time = dt.now()
# X = np.random.rand(N,M)
X = np.array([[-1.0, -1.0], [-2.0, -1.0], [-3.0, -2.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]])
pca = PCA(n_components=COMPONENTS)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

end_time = dt.now()
elapsed=end_time-start_time

print("SKCUDA Total timeTime: %02d:%02d:%02d:%02d" % (elapsed.days, elapsed.seconds // 3600, elapsed.seconds // 60 % 60, elapsed.seconds % 60))