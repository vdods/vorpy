import json
#import matplotlib.pyplot as plt
import numpy as np
import pathlib
import typing
import vorpy.experimental.kh
import vorpy.experimental.plot

def generate_curve_params () -> typing.Generator[typing.Dict[str,typing.Any],None,None]:
    curve_param_string__v = [
        'order:1_class:1_obj:1.0365e-07_dt:3.000e-03_t-max:2.000e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,3.04309794915225820e-06,-5.64195669743654538e-01]]_sheet-index:0_t-min:1.8589e+01',
        'order:2_class:1_obj:2.0008e-08_dt:3.000e-03_t-max:1.679e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,1.64381570097825697e-01,-8.92952723743407617e-01]]_sheet-index:0_t-min:1.5267e+01',
        'order:3_class:1_obj:2.7061e-08_dt:3.000e-03_t-max:2.084e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,2.26447253991376635e-01,-1.01708409153050949e+00]]_sheet-index:0_t-min:1.8948e+01',
        'order:3_class:2_obj:3.3190e-08_dt:3.000e-03_t-max:3.114e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,1.13078493507692862e-01,-7.90346570563142059e-01]]_sheet-index:0_t-min:2.8296e+01',
        'order:4_class:1_obj:1.3445e-08_dt:3.000e-03_t-max:2.503e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,2.70756053341894620e-01,-1.10570169023154552e+00]]_sheet-index:0_t-min:2.2749e+01',
        'order:4_class:3_obj:6.3434e-08_dt:3.000e-03_t-max:4.723e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,8.71858103378525262e-02,-7.38561204223461276e-01]]_sheet-index:0_t-min:4.2957e+01',
        'order:5_class:1_obj:3.8086e-08_dt:3.000e-03_t-max:2.904e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,3.07367356874409137e-01,-1.17892429729657455e+00]]_sheet-index:0_t-min:2.6403e+01',
        'order:5_class:2_obj:2.0495e-07_dt:3.000e-03_t-max:3.750e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,1.98984991121259114e-01,-9.62159565790274507e-01]]_sheet-index:0_t-min:3.4095e+01',
        'order:5_class:3_obj:2.9994e-07_dt:3.000e-03_t-max:4.740e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,1.33318770152522442e-01,-8.30827123852801108e-01]]_sheet-index:0_t-min:4.3092e+01',
        'order:5_class:4_obj:7.1085e-08_dt:3.000e-03_t-max:6.464e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,7.09670653373268706e-02,-7.06123714222410048e-01]]_sheet-index:0_t-min:5.8728e+01',
        'order:6_class:1_obj:2.7460e-08_dt:3.000e-03_t-max:3.287e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,3.39420556503194581e-01,-1.24303069655414555e+00]]_sheet-index:0_t-min:2.9892e+01',
        'order:6_class:5_obj:1.1463e-07_dt:3.000e-03_t-max:8.281e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,5.97963965682354048e-02,-6.83782376684227033e-01]]_sheet-index:0_t-min:7.5231e+01',
        'order:7_class:1_obj:7.3865e-08_dt:3.000e-03_t-max:3.655e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,3.68364153692793017e-01,-1.30091789093334231e+00]]_sheet-index:0_t-min:3.3225e+01',
        'order:7_class:2_obj:2.1387e-07_dt:3.000e-03_t-max:4.591e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,2.49898446921375805e-01,-1.06398647739050789e+00]]_sheet-index:0_t-min:4.1730e+01',
        'order:7_class:3_obj:9.1267e-07_dt:3.000e-03_t-max:5.421e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,1.88499720660134040e-01,-9.41189024868024360e-01]]_sheet-index:0_t-min:4.9299e+01',
        'order:7_class:4_obj:1.1795e-06_dt:3.000e-03_t-max:6.399e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,1.42026551431478493e-01,-8.48242686410713320e-01]]_sheet-index:0_t-min:5.8209e+01',
        'order:7_class:5_obj:5.8212e-07_dt:3.000e-03_t-max:7.799e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,9.84207418189708139e-02,-7.61031067185697907e-01]]_sheet-index:0_t-min:7.0938e+01',
        'order:7_class:6_obj:1.9297e-07_dt:3.000e-03_t-max:1.014e+02_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,5.16236664858185645e-02,-6.67436916519393408e-01]]_sheet-index:0_t-min:9.2238e+01',
        'order:8_class:1_obj:7.0838e-08_dt:3.000e-03_t-max:4.005e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,3.95010849958790378e-01,-1.35421128346533703e+00]]_sheet-index:0_t-min:3.6417e+01',
        'order:8_class:3_obj:1.0248e-06_dt:3.000e-03_t-max:5.837e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,2.08718960820790667e-01,-9.81627505189337612e-01]]_sheet-index:0_t-min:5.3043e+01',
        'order:8_class:5_obj:1.7120e-06_dt:3.000e-03_t-max:7.840e+01_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,1.25738428518738427e-01,-8.15666440585233188e-01]]_sheet-index:0_t-min:7.1268e+01',
        'order:8_class:7_obj:2.2664e-07_dt:3.000e-03_t-max:1.206e+02_ic:[[1.00000000000000000e+00,0.00000000000000000e+00,0.00000000000000000e+00],[0.00000000000000000e+00,4.53992392384779467e-02,-6.54988062024712159e-01]]_sheet-index:0_t-min:1.0959e+02',
    ]

    for curve_param_string in curve_param_string__v:
        token__v = curve_param_string.split('_')
        untyped_curve_param__d = dict(token.split(':') for token in token__v)
        curve_param__d = {
            'source-string' :curve_param_string,
            'order'         :int(untyped_curve_param__d['order']),
            'class'         :int(untyped_curve_param__d['class']),
            'obj'           :float(untyped_curve_param__d['obj']),
            'dt'            :float(untyped_curve_param__d['dt']),
            't-max'         :float(untyped_curve_param__d['t-max']),
            'ic'            :np.array(json.loads(untyped_curve_param__d['ic'])),
            'sheet-index'   :int(untyped_curve_param__d['sheet-index']),
            't-min'         :float(untyped_curve_param__d['t-min']),
        }
        yield curve_param__d

def plot_Euclidean (t__v:np.ndarray, qp__t:np.ndarray, plot__p:pathlib.Path, *, decoration:bool=True, title__o:typing.Optional[str]=None) -> None:
    qp = vorpy.experimental.kh.EuclideanSymbolics.qp_coordinates()

    #plt.box(False)

    plot = vorpy.experimental.plot.Plot(row_count=1, col_count=1, size=4)

    axis = plot.axis(0, 0)
    #axis.set_title(f'(x(t), y(t))')
    if title__o is not None:
        axis.set_title(title__o)
    axis.axis('off')
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.set_aspect(1.0)
    axis.plot(qp__t[:,0,0], qp__t[:,0,1])
    lim = (min(axis.get_xlim()[0], axis.get_ylim()[0]), max(axis.get_xlim()[1], axis.get_ylim()[1]))
    axis.set_xlim(lim[0], lim[1])
    axis.set_ylim(lim[0], lim[1])

    #axis = plot.axis(0, 1)
    #axis.set_title(f'(t, z(t))')
    #axis.plot(t__v, qp__t[:,0,2])

    plot.savefig(plot__p)

def plot_periodic_orbits (base_dir__p:pathlib.Path):
    for curve_param__d in generate_curve_params():
        base_name = f'{curve_param__d["class"]}:{curve_param__d["order"]}'

        pickle_filename__p = base_dir__p / f'{base_name}.pickle'
        plot__p = base_dir__p / f'{base_name}.pdf'

        results = vorpy.experimental.kh.EuclideanNumerics.compute_trajectory(pickle_filename__p, curve_param__d['ic'], curve_param__d['t-min'], curve_param__d['sheet-index'])
        plot_Euclidean(results.t_v, results.y_t, plot__p, title__o=f'Symmetry type {curve_param__d["class"]}:{curve_param__d["order"]}')

if __name__ == '__main__':
    plot_periodic_orbits(pathlib.Path('periodic-orbits.00'))


