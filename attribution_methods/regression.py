import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def lineare_regression(lrp , lrp_epsilon, pdiff, gradxin, lrp_z, lrp_id, lrp_merged, ig, omega):

    lrp = lrp.flatten()
    lrp_epsilon = lrp_epsilon.flatten()
    pdiff = pdiff.flatten()
    gradxin = gradxin.flatten()
    lrp_z = lrp_z.flatten()
    lrp_id = lrp_id.flatten()
    lrp_merged = lrp_merged.flatten()
    ig = ig.flatten()
    omega = omega.flatten()

    df = pd.DataFrame(data=[lrp, pdiff, gradxin, lrp_id, lrp_epsilon, lrp_z, lrp_merged, ig, omega]).T
    df.columns = ['Heuristic Rule', 'Occlusion', 'GradientXInput', 'Identity Rule', 'Epsilon Rule', 'Z Rule', 'LRP Fusion', 'Integrated Gradients', 'Omega Rule']


    sns.pairplot(df, kind="reg",diag_kind="kde", corner=True)

    plt.savefig('regression.png', bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    n_embedding = 49
    embedding_dim = 4
    lrp = np.random.randn(n_embedding * embedding_dim)
    fdiff = np.random.randn(n_embedding * embedding_dim)
    pdiff = np.random.randn(n_embedding * embedding_dim)

    lineare_regression(lrp, fdiff, pdiff)