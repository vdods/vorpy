import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
import vorpy.pickle

if __name__ == '__main__':
    J_v = []
    lam_v = []

    for pickle_p in map(pathlib.Path, map(str.strip, sys.stdin.readlines())):
        data_d = vorpy.pickle.unpickle(pickle_filename=str(pickle_p), log_out=sys.stdout)
        J_v.append(data_d['J_initial'])
        lam_v.append(data_d['lam'])

    row_count   = 1
    col_count   = 1
    size        = 8
    fig,axis_vv = plt.subplots(row_count, col_count, squeeze=False, figsize=(size*col_count,size*row_count))

    axis = axis_vv[0][0]
    axis.set_title('(J,lambda)')
    axis.scatter(J_v, lam_v)

    plot_p = pathlib.Path('J_vs_lam.png')

    fig.tight_layout()
    plot_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_p), bbox_inches='tight')
    print(f'wrote to file "{plot_p}"')
    # VERY important to do this -- otherwise your memory will slowly fill up!
    # Not sure which one is actually sufficient -- apparently none of them are, YAY!
    plt.clf()
    plt.cla()
    plt.close()
    plt.close(fig)
    plt.close('all')
    del fig
    del axis_vv
