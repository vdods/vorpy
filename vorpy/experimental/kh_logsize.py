import itertools
import numpy as np
import pathlib
import typing
import vorpy.experimental.kh
import vorpy.experimental.plot
import vorpy.realfunction.piecewiselinear

def plot_LogSize_and_Euclidean (t__v:np.ndarray, qp_ls__t:np.ndarray, qp_r3__t:np.ndarray, plot__p:pathlib.Path) -> None:
    qp_ls = vorpy.experimental.kh.LogSizeSymbolics.qp_coordinates()
    qp_r3 = vorpy.experimental.kh.EuclideanSymbolics.qp_coordinates()

    plot = vorpy.experimental.plot.Plot(row_count=3, col_count=4, size=6)

    H__v = vorpy.apply_along_axes(vorpy.experimental.kh.LogSizeNumerics.H__fast, (1,2), (qp_ls__t,))
    J__v = vorpy.apply_along_axes(vorpy.experimental.kh.LogSizeNumerics.J__fast, (1,2), (qp_ls__t,))

    abs_H_error__v = np.abs(H__v - H__v[0])
    abs_J_error__v = np.abs(J__v - J__v[0])

    for phase_index in range(2):
        for coord_index in range(3):
            coord_name = str(qp_ls[phase_index,coord_index])

            axis = plot.axis(phase_index, coord_index)
            axis.set_title(f'(t, {coord_name}(t))')
            axis.plot(t__v, qp_ls__t[:,phase_index,coord_index])

        axis = plot.axis(2, 2*phase_index+0)
        if phase_index == 0:
            axis.set_title(f'(x(t), y(t))')
        else:
            axis.set_title(f'(p_x(t), p_y(t))')
        axis.set_aspect(1.0)
        axis.plot(qp_r3__t[:,phase_index,0], qp_r3__t[:,phase_index,1])

        axis = plot.axis(2, 2*phase_index+1)
        if phase_index == 0:
            axis.set_title(f'(t, z(t))')
        else:
            axis.set_title(f'(t, p_z(t))')
        axis.plot(t__v, qp_r3__t[:,phase_index,2])

    axis = plot.axis(0, 3)
    axis.set_title(f'(t, abs(H(t) - H(0)))\nH(0) = {H__v[0]}\nmax error = {np.max(abs_H_error__v)}')
    axis.semilogy(t__v, abs_H_error__v)

    axis = plot.axis(1, 3)
    axis.set_title(f'(t, abs(J(t) - J(0)))\nJ(0) = {J__v[0]}\nmax error = {np.max(abs_J_error__v)}')
    axis.semilogy(t__v, abs_J_error__v)

    plot.savefig(plot__p)

def plot_LogSize (t__v:np.ndarray, qp__t:np.ndarray, plot__p:pathlib.Path) -> None:
    qp = vorpy.experimental.kh.LogSizeSymbolics.qp_coordinates()

    plot = vorpy.experimental.plot.Plot(row_count=2, col_count=4, size=6)

    H__v = vorpy.apply_along_axes(vorpy.experimental.kh.LogSizeNumerics.H__fast, (1,2), (qp__t,))
    J__v = vorpy.apply_along_axes(vorpy.experimental.kh.LogSizeNumerics.J__fast, (1,2), (qp__t,))

    abs_H_error__v = np.abs(H__v - H__v[0])
    abs_J_error__v = np.abs(J__v - J__v[0])

    for phase_index in range(2):
        for coord_index in range(3):
            coord_name = str(qp[phase_index,coord_index])

            axis = plot.axis(phase_index, coord_index)
            axis.set_title(f'(t, {coord_name}(t))')
            if coord_index == 2:
                axis.axhline(0.0, color='black')
            axis.plot(t__v, qp__t[:,phase_index,coord_index])

    axis = plot.axis(0, 3)
    axis.set_title(f'(t, abs(H(t) - H(0)))\nH(0) = {H__v[0]}\nmax error = {np.max(abs_H_error__v)}')
    axis.semilogy(t__v, abs_H_error__v)

    axis = plot.axis(1, 3)
    axis.set_title(f'(t, abs(J(t) - J(0)))\nJ(0) = {J__v[0]}\nmax error = {np.max(abs_J_error__v)}')
    axis.semilogy(t__v, abs_J_error__v)

    plot.savefig(plot__p)

def plot_QuadraticCylindrical (t__v:np.ndarray, qp__t:np.ndarray, plot__p:pathlib.Path) -> None:
    qp = vorpy.experimental.kh.QuadraticCylindricalSymbolics.qp_coordinates()

    plot = vorpy.experimental.plot.Plot(row_count=2, col_count=4, size=6)

    H__v = vorpy.apply_along_axes(vorpy.experimental.kh.QuadraticCylindricalNumerics.H__fast, (1,2), (qp__t,))
    J__v = vorpy.apply_along_axes(vorpy.experimental.kh.QuadraticCylindricalNumerics.J__fast, (1,2), (qp__t,))

    abs_H_error__v = np.abs(H__v - H__v[0])
    abs_J_error__v = np.abs(J__v - J__v[0])

    for phase_index in range(2):
        for coord_index in range(3):
            coord_name = str(qp[phase_index,coord_index])

            axis = plot.axis(phase_index, coord_index)
            axis.set_title(f'(t, {coord_name}(t))')
            if coord_index == 2:
                axis.axhline(0.0, color='black')
            axis.plot(t__v, qp__t[:,phase_index,coord_index])

    axis = plot.axis(0, 3)
    axis.set_title(f'(t, abs(H(t) - H(0)))\nH(0) = {H__v[0]}\nmax error = {np.max(abs_H_error__v)}')
    axis.semilogy(t__v, abs_H_error__v)

    axis = plot.axis(1, 3)
    axis.set_title(f'(t, abs(J(t) - J(0)))\nJ(0) = {J__v[0]}\nmax error = {np.max(abs_J_error__v)}')
    axis.semilogy(t__v, abs_J_error__v)

    plot.savefig(plot__p)

def plot_Euclidean (t__v:np.ndarray, qp__t:np.ndarray, plot__p:pathlib.Path) -> None:
    qp = vorpy.experimental.kh.EuclideanSymbolics.qp_coordinates()

    plot = vorpy.experimental.plot.Plot(row_count=2, col_count=3, size=6)

    H__v = vorpy.apply_along_axes(vorpy.experimental.kh.EuclideanNumerics.H__fast, (1,2), (qp__t,))
    J__v = vorpy.apply_along_axes(vorpy.experimental.kh.EuclideanNumerics.J__fast, (1,2), (qp__t,))

    abs_H_error__v = np.abs(H__v - H__v[0])
    abs_J_error__v = np.abs(J__v - J__v[0])

    for phase_index in range(2):
        axis = plot.axis(phase_index, 0)
        if phase_index == 0:
            axis.set_title(f'(x(t), y(t))')
        else:
            axis.set_title(f'(p_x(t), p_y(t))')
        axis.set_aspect(1.0)
        axis.plot([0.0], [0.0], '.', color='black')
        axis.plot(qp__t[:,phase_index,0], qp__t[:,phase_index,1])

        axis = plot.axis(phase_index, 1)
        if phase_index == 0:
            axis.set_title(f'(t, z(t))')
        else:
            axis.set_title(f'(t, p_z(t))')
        axis.axhline(0.0, color='black')
        axis.plot(t__v, qp__t[:,phase_index,2])

    axis = plot.axis(0, 2)
    axis.set_title(f'(t, abs(H(t) - H(0)))\nH(0) = {H__v[0]}\nmax error = {np.max(abs_H_error__v)}')
    axis.semilogy(t__v, abs_H_error__v)

    axis = plot.axis(1, 2)
    axis.set_title(f'(t, abs(J(t) - J(0)))\nJ(0) = {J__v[0]}\nmax error = {np.max(abs_J_error__v)}')
    axis.semilogy(t__v, abs_J_error__v)

    plot.savefig(plot__p)

def plot_LogSize_simple (t__v:np.ndarray, qp__t:np.ndarray, plot__p:pathlib.Path) -> None:
    qp = vorpy.experimental.kh.LogSizeSymbolics.qp_coordinates()

    H_initial = vorpy.experimental.kh.LogSizeNumerics.H__fast(qp__t[0])
    J_initial = vorpy.experimental.kh.LogSizeNumerics.J__fast(qp__t[0])
    p_theta_initial = vorpy.experimental.kh.LogSizeNumerics.p_theta__fast(qp__t[0])

    plot = vorpy.experimental.plot.Plot(row_count=3, col_count=1, size=3.5)

    if np.abs(H_initial) < 1.0e-15:
        H_initial = 0
    if np.abs(J_initial) < 1.0e-15:
        J_initial = 0

    if J_initial < 0:
        J_string = 'J < 0'
    elif J_initial > 0:
        J_string = 'J > 0'
    else:
        J_string = 'J = 0'

    #plot.fig.suptitle(f'Trajectory in (s,theta,u) coordinates\nH = {H_initial}, J = {J_initial}, p_theta = {p_theta_initial}')
    plot.fig.suptitle(f'Trajectory with H = {H_initial}, {J_string}')

    for coord_index in range(3):
        coord_name = str(qp[0,coord_index])

        axis = plot.axis(coord_index, 0)
        axis.set_title(f'Graph: (t, {coord_name}(t))')
        if coord_index == 2:
            axis.axhline(0.0, color='black')
        axis.plot(t__v, qp__t[:,0,coord_index])

    ## Find zeros of u(t).
    #u__v = qp__t[:,0,2]
    #_, _, u_zero_t__v = vorpy.realfunction.piecewiselinear.oriented_zeros(u__v, t_v=t__v)

    #axis = plot.axis(0, 2)
    #for u_zero_t in u_zero_t__v:
        #axis.axvline(u_zero_t, color='red', alpha=0.5)

    # Find zeros of u(t).
    u__v = qp__t[:,0,2]
    _, _, u_zero_t__v = vorpy.realfunction.piecewiselinear.oriented_zeros(u__v, t_v=t__v)

    # Compute fundamental domains, and visualize them
    fundamental_domain_boundary__v = u_zero_t__v[::2]
    fundamental_domain__v = list(zip(fundamental_domain_boundary__v[:-1], fundamental_domain_boundary__v[1:]))
    for i,fundamental_domain in enumerate(fundamental_domain__v):
        if i % 2 == 0:
            color = 'cyan'
        else:
            color = 'green'
        for coord_index in range(3):
            axis = plot.axis(coord_index, 0)
            axis.axvspan(fundamental_domain[0], fundamental_domain[1], color=color, alpha=0.1)

    for coord_index in range(3):
        axis = plot.axis(coord_index, 0)
        for u_zero_t in u_zero_t__v:
            axis.axvline(u_zero_t, color='black', alpha=0.25)

    plot.savefig(plot__p, tight_layout_kwargs=dict(rect=[0, 0.03, 1, 0.95]))

def plot_LogSize_u_simple (t__v:np.ndarray, qp__t:np.ndarray, plot__p:pathlib.Path) -> None:
    qp = vorpy.experimental.kh.LogSizeSymbolics.qp_coordinates()

    H_initial = vorpy.experimental.kh.LogSizeNumerics.H__fast(qp__t[0])
    J_initial = vorpy.experimental.kh.LogSizeNumerics.J__fast(qp__t[0])
    p_theta_initial = vorpy.experimental.kh.LogSizeNumerics.p_theta__fast(qp__t[0])

    plot = vorpy.experimental.plot.Plot(row_count=1, col_count=1, size=10)

    if np.abs(H_initial) < 1.0e-15:
        H_initial = 0
    if np.abs(J_initial) < 1.0e-15:
        J_initial = 0
    #plot.fig.suptitle(f'Trajectory in (s,theta,u) coordinates\nH = {H_initial}, J = {J_initial}, p_theta = {p_theta_initial}')

    coord_index = 2
    coord_name = str(qp[0,coord_index])

    axis = plot.axis(0, 0)
    axis.set_title(f'Graph: (t, {coord_name}(t)); H = {H_initial}, J = {J_initial}, p_theta = {p_theta_initial}')
    #axis.set_aspect(0.5)
    if coord_index == 2:
        axis.axhline(0.0, color='black')
    axis.plot(t__v, qp__t[:,0,coord_index])

    # Find zeros of u(t).
    u__v = qp__t[:,0,2]
    _, _, u_zero_t__v = vorpy.realfunction.piecewiselinear.oriented_zeros(u__v, t_v=t__v)

    fundamental_domain_boundary__v = u_zero_t__v[::2]
    fundamental_domain__v = list(zip(fundamental_domain_boundary__v[:-1], fundamental_domain_boundary__v[1:]))
    for i,fundamental_domain in enumerate(fundamental_domain__v):
        if i % 2 == 0:
            color = 'cyan'
        else:
            color = 'green'
        axis.axvspan(fundamental_domain[0], fundamental_domain[1], color=color, alpha=0.1)

    for u_zero_t in u_zero_t__v:
        axis.axvline(u_zero_t, color='black', alpha=0.25)

    plot.fig.set_size_inches(10, 4)

    plot.savefig(plot__p)

def plot_Euclidean_simple (t__v:np.ndarray, qp__t:np.ndarray, plot__p:pathlib.Path) -> None:
    qp = vorpy.experimental.kh.EuclideanSymbolics.qp_coordinates()

    H_initial = vorpy.experimental.kh.EuclideanNumerics.H__fast(qp__t[0])
    J_initial = vorpy.experimental.kh.EuclideanNumerics.J__fast(qp__t[0])
    p_theta_initial = vorpy.experimental.kh.EuclideanNumerics.p_theta__fast(qp__t[0])

    plot = vorpy.experimental.plot.Plot(row_count=2, col_count=1, size=3.5)

    if np.abs(H_initial) < 1.0e-15:
        H_initial = 0
    if np.abs(J_initial) < 1.0e-15:
        J_initial = 0

    if J_initial < 0:
        J_string = 'J < 0'
    elif J_initial > 0:
        J_string = 'J > 0'
    else:
        J_string = 'J = 0'

    plot.fig.suptitle(f'Trajectory with H = {H_initial}, {J_string}')

    axis = plot.axis(0, 0)
    axis.set_title(f'Graph: (x(t), y(t))')
    axis.set_aspect(1.0)
    axis.plot([0.0], [0.0], '.', color='black')
    axis.plot(qp__t[:,0,0], qp__t[:,0,1])
    lim = (min(axis.get_xlim()[0], axis.get_ylim()[0]), max(axis.get_xlim()[1], axis.get_ylim()[1]))
    #axis.set_xlim(lim[0], lim[1])
    #axis.set_ylim(lim[0], lim[1])

    axis = plot.axis(1, 0)
    axis.set_title(f'Graph: (t, z(t))')
    axis.axhline(0.0, color='black')
    axis.plot(t__v, qp__t[:,0,2])

    plot.savefig(plot__p, tight_layout_kwargs=dict(rect=[0, 0.03, 1, 0.95]))

def do_stuff () -> None:
    s_initial__v        = [0.0]
    theta_initial__v    = [0.0]
    u_initial__v        = [0.0]
    p_s_count           = 3
    #p_s_initial__v      = [0.0]
    p_theta_initial__v  = np.linspace(0.0625, 1.0, 9)
    H_initial__v        = [0.0]
    solution_sheet__v   = [0, 1]
    #solution_sheet = 0

    base_dir__p = pathlib.Path('kh_dilation.ls.09')

    image_index = 0

    for args in itertools.product(s_initial__v, theta_initial__v, u_initial__v, p_theta_initial__v, H_initial__v):
        s_initial, theta_initial, u_initial, p_theta_initial, H_initial = args

        p_s_bounds              = vorpy.experimental.kh.LogSizeNumerics.p_s_bounds__fast(np.array([u_initial, p_theta_initial]))
        p_s_initial__v          = np.linspace(p_s_bounds[0], p_s_bounds[1], p_s_count+2)[1:-1]
        #p_s_initial__v          = np.linspace(0.0, p_s_bounds[1], p_s_count)
        #p_s_initial__v          = np.linspace(p_s_bounds[0], 0.0, p_s_count)
        for p_s_initial in p_s_initial__v:
            X                   = np.array([s_initial, theta_initial, u_initial, p_s_initial, p_theta_initial, H_initial])

            for solution_sheet in solution_sheet__v:
                p_u_initial         = vorpy.experimental.kh.LogSizeNumerics.p_u_constrained_by_H__fast(X)[solution_sheet]

                if np.abs(H_initial) < 1.0e-10:
                    t_final = 100.0
                else:
                    t_final = 2000.0

                if image_index == 20:
                    t_final = 40.0
                elif image_index == 21:
                    t_final = 130.0
                elif image_index == 22:
                    t_final = 500.0
                elif image_index == 23:
                    t_final = 1000.0
                #else:
                    #image_index += 1
                    #continue # temp

                if image_index not in [13, 21, 22, 24]:
                    image_index += 1
                    continue

                def filename (prefix:str, suffix:str) -> pathlib.Path:
                    return base_dir__p / f'{prefix}_p_s={p_s_initial}_p_theta={p_theta_initial}_p_u={p_u_initial}_solutionsheet={solution_sheet}{suffix}'

                solve_in_ls         = True

                qp_initial_ls       = np.array([
                    [s_initial,   theta_initial,   u_initial],
                    [p_s_initial, p_theta_initial, p_u_initial],
                ])
                if solve_in_ls:
                    results         = vorpy.experimental.kh.LogSizeNumerics.compute_trajectory(filename('logsize', '.pickle'), qp_initial_ls, t_final, solution_sheet)
                else:
                    qp_initial_qc   = vorpy.experimental.kh.LogSizeNumerics.qp_to_QuadraticCylindrical__fast(qp_initial_ls)
                    results         = vorpy.experimental.kh.QuadraticCylindricalNumerics.compute_trajectory(filename('quadcyl', '.pickle'), qp_initial_qc, t_final, solution_sheet)

                t__v                = results.t_v
                if solve_in_ls:
                    qp_ls__t        = results.y_t
                    qp_qc__t        = vorpy.apply_along_axes(vorpy.experimental.kh.LogSizeNumerics.qp_to_QuadraticCylindrical__fast, (1,2), (qp_ls__t,))
                else:
                    qp_qc__t        = results.y_t
                    qp_ls__t        = vorpy.apply_along_axes(vorpy.experimental.kh.QuadraticCylindricalNumerics.qp_to_LogSize__fast, (1,2), (qp_qc__t,))
                qp_r3__t            = vorpy.apply_along_axes(vorpy.experimental.kh.QuadraticCylindricalNumerics.qp_to_Euclidean__fast, (1,2), (qp_qc__t,))

                #plot_LogSize(t__v, qp_ls__t, filename(f'image_{image_index:04}_logsize', '.png'))
                #plot_QuadraticCylindrical(t__v, qp_qc__t, filename(f'image_{image_index:04}_quadcyl', '.png'))
                #plot_Euclidean(t__v, qp_r3__t, filename(f'image_{image_index:04}_euclidean', '.png'))
                plot_LogSize_simple(t__v, qp_ls__t, filename(f'image_{image_index:04}_logsize', '.pdf'))
                plot_LogSize_u_simple(t__v, qp_ls__t, filename(f'image_{image_index:04}_logsize_u', '.pdf'))
                plot_Euclidean_simple(t__v, qp_r3__t, filename(f'image_{image_index:04}_euclidean', '.pdf'))
                #plot_LogSize_and_Euclidean(t__v, qp_ls__t, qp_r3__t, filename(f'image_{image_index:04}_logsize_and_r3', '.png'))

                image_index += 1

if __name__ == '__main__':
    do_stuff()
