import numpy as np
import sympy as sp
#import typing
import vorpy.riemannian
import vorpy.symbolic
import vorpy.tensor
from vorpy.experimental.coordinates import * # TEMP HACK

#def make_2d_cartesian_cooords () -> np.ndarray:
    #return np.array(sp.var('x, y'))

#def make_3d_cartesian_cooords () -> np.ndarray:
    #return np.array(sp.var('x, y, z'))

#def make_polar_cooords () -> np.ndarray:
    #return np.array(sp.var('r, theta'))

#def make_cylindrical_cooords () -> np.ndarray:
    #return np.array(sp.var('r, theta, z'))

#def make_spherical_coords () -> np.ndarray:
    #return np.array(sp.var('rho, theta, phi'))

# TODO: Other cool ones, like parabolic coordinates

def eye_tensor_field (tensor_bundle_chart:TensorBundleChart) -> TensorBundleSection:
    """
    This just creates an identity matrix field in the given TensorBundleChart, which must
    have order 2 and whose factors must have the same dimension.
    """
    if tensor_bundle_chart.order() != 2:
        raise TypeError(f'eye_tensor_field is only well-defined on 2-tensor bundles.')
    if tensor_bundle_chart.factor(0).fiber_chart.dimension != tensor_bundle_chart.factor(1).fiber_chart.dimension:
        raise TypeError(f'eye_tensor_field is only well-defined when the tensor factors have the same dimension.')

    dimension = tensor_bundle_chart.factor(0).fiber_chart.dimension
    eye = np.diag([sp.Integer(1)]*dimension)

    def eye_fiber_evaluator (base:Coords) -> Coords:
        return tensor_bundle_chart.fiber_chart.make_coords(eye)

    return TensorBundleSection(
        name=f'Eye({tensor_bundle_chart})',
        bundle_chart=tensor_bundle_chart,
        fiber_evaluator=eye_fiber_evaluator,
    )

def standard_metric (base_chart:Chart) -> TensorBundleSection:
    #cotangent_bundle_chart = CotangentBundleChart.induced(base_chart)
    cotangent_bundle_chart = CotangentFunctor_ob(base_chart)
    metric_bundle_chart = TensorBundleChart(
        cotangent_bundle_chart,
        cotangent_bundle_chart,
        fiber_symbolic_coords=vorpy.symbolic.tensor('g', base_chart.coords_shape + base_chart.coords_shape)
    )
    return eye_tensor_field(metric_bundle_chart)

def induced_metric (domain_metric:TensorBundleSection, chart_isomorphism:Isomorphism) -> TensorBundleSection:
    """
    This should produce a metric G on the codomain chart in those coordinates.

    If J := jacobian(chart_isomorphism.inverse()), then G := J^T * (domain_metric \circ J) * J
    """

    #cotangent_bundle_chart = CotangentBundleChart.induced(chart_isomorphism.codomain)
    cotangent_bundle_chart = CotangentFunctor_ob(chart_isomorphism.codomain)
    metric_bundle_chart = TensorBundleChart(
        cotangent_bundle_chart,
        cotangent_bundle_chart,
        fiber_symbolic_coords=vorpy.symbolic.tensor('g', chart_isomorphism.codomain.coords_shape + chart_isomorphism.codomain.coords_shape)
    )

    J = jacobian(chart_isomorphism.inverse())
    #print(f'HIPPO chart_isomorphism:\n{chart_isomorphism}')
    #print()
    #print(f'HIPPO J:\n{J}')
    #print()

    def metric_fiber_evaluator (base:Coords) -> Coords:
        assert base.chart == chart_isomorphism.codomain
        #print(f'HIPPO base:\n{base}')
        #print()
        other_base = chart_isomorphism.inverse()(base)
        #print(f'HIPPO other_base:\n{other_base}')
        #print()
        assert other_base.chart == chart_isomorphism.domain
        J_base = J(base)
        #print(f'HIPPO J_base.base():\n{J_base.base()}')
        #print()
        #print(f'HIPPO J_base.fiber():\n{J_base.fiber()}')
        #print()
        return metric_bundle_chart.fiber_chart.make_coords(
            simplified(vorpy.tensor.contract('ji,jk,kl', J_base.fiber().value, domain_metric(other_base).fiber().value, J_base.fiber().value, dtype=object))
        )

    return TensorBundleSection(
        name=f'InducedMetric({chart_isomorphism})',
        bundle_chart=metric_bundle_chart,
        fiber_evaluator=metric_fiber_evaluator,
    )

def metric_inv (metric:TensorBundleSection) -> TensorBundleSection:
    """
    Computes the inverse to the metric tensor field, i.e. g_inv, where g_inv*g = I.
    """

    # TODO: Create an "inv" function on 2-tensor bundles whose factors have the same dimension.

    #tangent_bundle_chart = TangentBundleChart.induced(metric.domain)
    tangent_bundle_chart = TangentFunctor_ob(metric.domain)
    metric_inv_bundle_chart = TensorBundleChart(
        tangent_bundle_chart,
        tangent_bundle_chart,
        fiber_symbolic_coords=vorpy.symbolic.tensor('g_inv', metric.domain.coords_shape + metric.domain.coords_shape)
    )

    x = metric.domain.symbolic_coords
    #print(f'HIPPO x = {x}')
    metric_x = metric(x)
    #print(f'HIPPO metric_x = {metric_x}')
    #print(f'HIPPO metric_x.fiber() = {metric_x.fiber()}')
    metric_inv_fiber_coords = metric_inv_bundle_chart.fiber_chart.symbolic_coords
    #print(f'HIPPO metric_inv_fiber_coords:\n{metric_inv_fiber_coords}')
    #print()

    eye = np.diag([sp.Integer(1)]*metric.domain.dimension)
    sol__d = sp.solve(
        (np.dot(metric_x.fiber().value, metric_inv_fiber_coords.value) - eye).reshape(-1),
        metric_inv_fiber_coords.value.reshape(-1).tolist(),
        dict=False
    )
    #print(f'HIPPO sol__d = {sol__d}')
    #print()

    metric_inv_fiber_sol = np.array([sol__d[metric_inv_fiber_coord] for metric_inv_fiber_coord in metric_inv_fiber_coords.value.reshape(-1)]).reshape(metric_inv_fiber_coords.value.shape)
    #print(f'HIPPO metric_inv_fiber_sol = {metric_inv_fiber_sol}')
    #print()

    #print(f'HIPPO metric.domain.symbolic_coords = {metric.domain.symbolic_coords}')
    #print(f'HIPPO type(metric.domain.symbolic_coords) = {type(metric.domain.symbolic_coords)}')

    def metric_inv_fiber_evaluator (base:Coords) -> Coords:
        return metric_inv_bundle_chart.fiber_chart.make_coords(substitution(metric_inv_fiber_sol, metric.domain.symbolic_coords.value, base.value))

    return TensorBundleSection(
        name=f'MetricInv({metric})',
        bundle_chart=metric_inv_bundle_chart,
        fiber_evaluator=metric_inv_fiber_evaluator,
    )


def levi_civita_christoffel_symbol (g:TensorBundleSection, g_inv:TensorBundleSection) -> TensorBundleSection:
    """
    Note that the Christoffel symbol(s) is not a tensor, since it's coordinate dependent,
    so it's a lie to return this as a TensorBundleSection, but it does make the code easier.

    TODO: Make a ChristoffelSymbol class
    """

    # TODO: type check on g and g_inv
    assert g.bundle_chart.base_chart == g_inv.bundle_chart.base_chart

    #print(f'HIPPO g.bundle_chart:\n{g.bundle_chart}\n')
    #print(f'HIPPO g.bundle_chart.base_chart:\n{g.bundle_chart.base_chart}\n')

    x = g.bundle_chart.base_chart.symbolic_coords
    #print(f'HIPPO repr(x):\n{repr(x)}\n')
    g_x = g(x)
    g_inv_x = g_inv(x)
    #print(f'HIPPO repr(g_x):\n{repr(g_x)}\n')
    #print(f'HIPPO repr(g_x.fiber()):\n{repr(g_x.fiber())}\n')
    #print(f'HIPPO repr(g_x.fiber().value):\n{repr(g_x.fiber().value)}\n')
    #print(f'HIPPO repr(x.value):\n{repr(x.value)}\n')
    dg_x = vorpy.symbolic.differential(g_x.fiber().value, x.value)
    #print(f'HIPPO dg_x:\n{dg_x}\n')

    # TODO: Make this use
    #tangent_bundle_chart = TangentBundleChart.induced(g.domain)
    vector_bundle_chart = g_inv.bundle_chart.factor(0)
    #cotangent_bundle_chart = CotangentBundleChart.induced(g.domain)
    cotangent_bundle_chart = CotangentFunctor_ob(g.domain)
    christoffel_symbol_bundle_chart = TensorBundleChart(
        #tangent_bundle_chart,
        vector_bundle_chart,
        cotangent_bundle_chart,
        cotangent_bundle_chart,
        fiber_symbolic_coords=vorpy.symbolic.tensor('Gamma', g.domain.coords_shape*3)
    )

    # TODO: See about forming the sum g_{jl,k} + g_{kl,j} - g_{jk,l} beforehand.
    christoffel_symbol_fiber = simplified(
        sp.Rational(1,2)*(
            vorpy.tensor.contract('il,jlk', g_inv_x.fiber().value, dg_x, dtype=object)
            + vorpy.tensor.contract('il,klj', g_inv_x.fiber().value, dg_x, dtype=object)
            - vorpy.tensor.contract('il,jkl', g_inv_x.fiber().value, dg_x, dtype=object)
        )
    )
    #print(f'HIPPO type(christoffel_symbol_fiber):\n{type(christoffel_symbol_fiber)}')
    #print(f'HIPPO christoffel_symbol_fiber:\n{repr(christoffel_symbol_fiber)}')

    #print(f'HIPPO antisymmetrized:\n{christoffel_symbol_fiber - np.swapaxes(christoffel_symbol_fiber, 1, 2)}')
    #print(f'HIPPO antisymmetrized:\n{christoffel_symbol_fiber - vorpy.tensor.contract("ijk", christoffel_symbol_fiber, output="ikj", dtype=object)}')

    def christoffel_symbol_fiber_evaluator (base:Coords) -> Coords:
        return christoffel_symbol_bundle_chart.fiber_chart.make_coords(substitution(christoffel_symbol_fiber, g.domain.symbolic_coords.value, base.value))

    return TensorBundleSection(
        name=f'Gamma({g})',
        bundle_chart=christoffel_symbol_bundle_chart,
        fiber_evaluator=christoffel_symbol_fiber_evaluator,
    )

def covariant_derivative_of (field:VectorBundleSection, Gamma:TensorBundleSection) -> TensorBundleSection:
    """
    Gamma is a Christoffel symbol which defines the covariant derivative.

    TODO: Make a CovariantDerivative class
    """

    # TODO: Real checks with raise
    assert field.bundle_chart.order() == 1
    assert Gamma.bundle_chart.order() == 3
    assert field.bundle_chart.factor(0) == Gamma.bundle_chart.factor(0)

    x = field.domain.symbolic_coords
    field_x = field(x)
    Gamma_x = Gamma(x)

    print(f'HIPPO x:\n{repr(x)}\n')
    print(f'HIPPO field_x:\n{repr(field_x)}\n')
    print(f'HIPPO Gamma_x:\n{repr(Gamma_x)}\n')

    covariant_derivative = vorpy.symbolic.differential(field_x.fiber().value, x.value) + vorpy.tensor.contract('ijk,j', Gamma_x.fiber().value, field_x.fiber().value, dtype=object)

    cotangent_bundle_chart = CotangentFunctor_ob(field.domain)
    covariant_derivative_bundle_chart = TensorBundleChart(
        field.bundle_chart,
        cotangent_bundle_chart,
        fiber_symbolic_coords=vorpy.symbolic.tensor('v', field.bundle_chart.fiber_chart.coords_shape + cotangent_bundle_chart.fiber_chart.coords_shape)
    )
    def covariant_derivative_fiber_evaluator (base:Coords) -> Coords:
        return covariant_derivative_bundle_chart.fiber_chart.make_coords(substitution(covariant_derivative, field.domain.symbolic_coords.value, base.value))

    return TensorBundleSection(
        name=f'Nabla({field})',
        bundle_chart=covariant_derivative_bundle_chart,
        fiber_evaluator=covariant_derivative_fiber_evaluator
    )


if __name__ == '__main__':
    R3 = Chart(
        name='R3',
        coords_shape=(3,),
        symbolic_coords=np.array(sp.symbols('x,y,z', real=True)),
    )
    print(f'R3 = {R3}')
    print(f'repr(R3) = {repr(R3)}')
    print()

    v = R3.symbolic_coords
    print(f'v = {v}')
    print(f'repr(v) = {repr(v)}')
    print(f'v.chart() = {v.chart}')
    print(f'repr(v.chart) = {repr(v.chart)}')
    print()

    Cyl = Chart(
        name='Cyl',
        coords_shape=(3,),
        symbolic_coords=np.array([sp.Symbol('r', real=True, positive=True), sp.Symbol('theta', real=True), sp.Symbol('z', real=True)]),
    )
    print(f'Cyl = {Cyl}')
    print(f'repr(Cyl) = {repr(Cyl)}')
    print()

    c = Cyl.symbolic_coords
    print(f'c = {c}')
    print(f'repr(c) = {repr(c)}')
    print(f'c.chart() = {c.chart}')
    print(f'repr(c.chart) = {repr(c.chart)}')
    print()


    def evaluator_R3_to_Cyl (v:Coords) -> Coords:
        x, y, z = v.value
        r = sp.sqrt(x**2 + y**2)
        theta = sp.atan2(y, x)
        return Cyl.make_coords(simplified(np.array([r, theta, z])))

    def inverse_evaluator_R3_to_Cyl (c:Coords) -> Coords:
        r, theta, z = c.value
        x = r*sp.cos(theta)
        y = r*sp.sin(theta)
        return R3.make_coords(simplified(np.array([x, y, z])))

    R3_to_Cyl = Isomorphism(
        name='R3_to_Cyl',
        domain=R3,
        codomain=Cyl,
        # TODO: Make Coords inherit np.ndarray for convenience
        evaluator=evaluator_R3_to_Cyl,
        inverse_evaluator__o=inverse_evaluator_R3_to_Cyl,
    )
    print(f'R3_to_Cyl = {R3_to_Cyl}')
    print(f'repr(R3_to_Cyl) = {repr(R3_to_Cyl)}')
    print()

    Cyl_to_R3 = R3_to_Cyl.inverse()
    print(f'Cyl_to_R3 = {Cyl_to_R3}')
    print(f'repr(Cyl_to_R3) = {repr(Cyl_to_R3)}')
    print()

    c_v = R3_to_Cyl(v)
    print(f'R3_to_Cyl({v}) = {c_v}')
    print()

    v_c = Cyl_to_R3(c)
    print(f'Cyl_to_R3({c}) = {v_c}')
    print()

    v_c_v = Cyl_to_R3(c_v)
    print(f'Cyl_to_R3(R3_to_Cyl({v})) = {v_c_v}')
    print()

    c_v_c = R3_to_Cyl(v_c)
    print('this does not simplify all the way down; it needs to have a bound for theta')
    print(f'R3_to_Cyl(Cyl_to_R3({c})) = {c_v_c}')
    print()

    J_R3_to_Cyl = jacobian(R3_to_Cyl)
    print(f'J_R3_to_Cyl = {J_R3_to_Cyl}')
    J_R3_to_Cyl_v = J_R3_to_Cyl(v)
    print(f'{J_R3_to_Cyl}({v}):\n{J_R3_to_Cyl_v.base()},\n{J_R3_to_Cyl_v.fiber()}')
    print()

    J_Cyl_to_R3 = jacobian(Cyl_to_R3)
    print(f'J_Cyl_to_R3 = {J_Cyl_to_R3}')
    J_Cyl_to_R3_c = J_Cyl_to_R3(c)
    print(f'J_Cyl_to_R3_c = {J_Cyl_to_R3_c}')
    print(f'{J_Cyl_to_R3}({c}):\n{J_Cyl_to_R3_c.base()},\n{J_Cyl_to_R3_c.fiber()}')
    print()

    J_Cyl_to_R3_c_v = J_Cyl_to_R3(c_v)
    print(f'J_Cyl_to_R3_c_v = {J_Cyl_to_R3_c_v}')
    print(f'{J_Cyl_to_R3}({c_v}):\n{J_Cyl_to_R3_c_v.base()},\n{J_Cyl_to_R3_c_v.fiber()}')
    print()

    print(f'type(J_Cyl_to_R3_c_v) = {type(J_Cyl_to_R3_c_v)}')
    print()

    # TODO: Need pullback bundles and a specialized contract function for this to be correctly typed
    product = simplified(np.dot(J_Cyl_to_R3_c_v.fiber().value, J_R3_to_Cyl_v.fiber().value))
    print(f'{J_Cyl_to_R3}({c_v}) * {J_R3_to_Cyl}({v}):\n{product}')
    print()

    print('checking cotangent bundle coordinate transforms')

    T_star_Cyl = CotangentFunctor_ob(Cyl)
    p_Cyl = T_star_Cyl.symbolic_coords
    print(f'T_star_Cyl = {T_star_Cyl}')
    print(f'p_Cyl = {p_Cyl}')

    T_star_R3 = CotangentFunctor_ob(R3)
    p_R3 = T_star_R3.symbolic_coords
    print(f'T_star_R3 = {T_star_R3}')
    print(f'p_R3 = {p_R3}')
    print()

    v_c = Cyl_to_R3(p_Cyl.base())
    # This is the pullback of J_R3_to_Cyl over Cyl_to_R3, which means that it's a coordinate expression in Cyl coords.
    J = J_R3_to_Cyl(v_c)
    print(f'Cyl_to_R3(p_Cyl.base()) = {v_c}')
    print(f'J = (Cyl_to_R3^{{*}}J_R3_to_Cyl)(p_Cyl.base()) = {J}')
    print(f'as base and fiber: J:\n{J.base()},\n{J.fiber()}')
    print()

    product = simplified(np.dot(p_Cyl.fiber().value, J.fiber().value))
    print(f'p_Cyl * J = {product}')
    print()


    # Define the standard metric on R3
    R3_metric = standard_metric(R3)
    print(f'R3_metric = {R3_metric}')
    R3_metric_v = R3_metric(v)
    print(f'{R3_metric}({v}):\n{R3_metric_v.base()}\n{R3_metric_v.fiber()}')
    print()

    # Induce the metric on Cyl
    Cyl_metric = induced_metric(R3_metric, R3_to_Cyl)
    print(f'Cyl_metric = {Cyl_metric}')
    Cyl_metric_c = Cyl_metric(c)
    print(f'{Cyl_metric}({c}):\n{Cyl_metric_c.base()}\n{Cyl_metric_c.fiber()}')
    print()

    R3_metric_inv = metric_inv(R3_metric)
    print(f'R3_metric_inv = {R3_metric_inv}')
    R3_metric_inv_v = R3_metric_inv(v)
    print(f'{R3_metric_inv}({v}):\n{R3_metric_inv_v.base()}\n{R3_metric_inv_v.fiber()}')
    print()

    Cyl_metric_inv = metric_inv(Cyl_metric)
    print(f'Cyl_metric_inv = {Cyl_metric_inv}')
    Cyl_metric_inv_c = Cyl_metric_inv(c)
    print(f'{Cyl_metric_inv}({v}):\n{Cyl_metric_inv_c.base()}\n{Cyl_metric_inv_c.fiber()}')
    print()

    R3_Gamma = levi_civita_christoffel_symbol(R3_metric, R3_metric_inv)
    Cyl_Gamma = levi_civita_christoffel_symbol(Cyl_metric, Cyl_metric_inv)

    print(f'R3_Gamma:\n{repr(R3_Gamma)}\n{R3_Gamma(v)}\n')
    print(f'Cyl_Gamma:\n{repr(Cyl_Gamma)}\n{Cyl_Gamma(c)}\n')

    R3_Gamma_inv = levi_civita_christoffel_symbol(R3_metric_inv, R3_metric)
    Cyl_Gamma_inv = levi_civita_christoffel_symbol(Cyl_metric_inv, Cyl_metric)

    print(f'R3_Gamma_inv:\n{repr(R3_Gamma_inv)}\n{repr(R3_Gamma_inv(v))}\n')
    print(f'Cyl_Gamma_inv:\n{repr(Cyl_Gamma_inv)}\n{repr(Cyl_Gamma_inv(c))}\n')

    T_R3 = TangentFunctor_ob(R3)
    print(f'T_R3:\n{repr(T_R3)}\n{T_R3}\n')

    U = TangentBundleSection(
        name='U',
        bundle_chart=T_R3,
        fiber_evaluator=lambda base_coords : T_R3.fiber_chart.make_coords(
            np.array([
                base_coords.value[0]**2,
                sp.sin(base_coords.value[1])*base_coords.value[2],
                sp.Integer(1)
            ])
        )
    )

    nabla_U = covariant_derivative_of(U, R3_Gamma)
    nabla_U_v = nabla_U(v)
    print(f'nabla_U:\n{repr(nabla_U)}\n{nabla_U_v.base()}\n{nabla_U_v.fiber()}\n')

    T_Cyl = TangentFunctor_ob(Cyl)
    print(f'T_Cyl:\n{repr(T_Cyl)}\n{T_Cyl}\n')

    V = TangentBundleSection(
        name='V',
        bundle_chart=T_Cyl,
        fiber_evaluator=lambda base_coords : T_Cyl.fiber_chart.make_coords(
            np.array([
                base_coords.value[0]**2,
                sp.sin(base_coords.value[1])*base_coords.value[2],
                sp.Integer(1)
            ])
        )
    )

    nabla_V = covariant_derivative_of(V, Cyl_Gamma)
    nabla_V_c = nabla_V(c)
    print(f'nabla_V:\n{repr(nabla_V)}\n{nabla_V_c.base()}\n{nabla_V_c.fiber()}\n')

