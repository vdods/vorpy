import abc
import itertools
import numpy as np
import sympy as sp
import textwrap
import typing
import vorpy.experimental.require as require
import vorpy.symbolic
import vorpy.symplectic

# Convenience function because sympy's simplify function doesn't preserve the input type
def simplified (arg:np.ndarray, *, preserve_shape:bool=False) -> np.ndarray:
    retval = np.array(sp.simplify(arg)).reshape(np.shape(arg))
    if not preserve_shape:
        if isinstance(retval, np.ndarray) and retval.shape == tuple():
            retval = retval[()]
    return retval

def substitution (expr:typing.Any, point:np.array, value:np.array, *, simplify:bool=True, preserve_shape:bool=False) -> np.ndarray:
    retval = np.array(sp.Subs(expr, point.reshape(-1), value.reshape(-1)).doit()).reshape(np.shape(expr))
    if simplify:
        retval = simplified(retval)
    if not preserve_shape:
        if isinstance(retval, np.ndarray) and retval.shape == tuple():
            retval = retval[()]
    return retval

def prefixed_symbolic_tensor (prefix:str, symbol__t:np.ndarray) -> np.ndarray:
    """
    Creates a new symbolic tensor where each new symbol has assumptions corresponding to
    the original, but the symbol name is prefix+str(symbol).
    """
    return np.array([sp.Symbol(prefix+str(symbol), **symbol.assumptions0) for symbol in symbol__t.flat]).reshape(symbol__t.shape)

class Coords:
    """
    Instances of this class each represent a coordinatized point in a particular chart
    in a particular manifold.

    Users should not construct instances of this class directly; use the corresponding
    Chart instance instead.

    The `value` attribute is the actual coordinates value.
    The `chart` attribute gives the "type" of the coordinates (i.e. the coordinate chart).
    """

    def __init__ (self, value:np.ndarray, *, chart:'Chart') -> None:
        if value.shape != chart.coords_shape:
            raise TypeError(f'Expected value.shape (which was {value.shape}) to be equal to chart.coords_shape (which was {chart.coords_shape}); chart = {chart}')

        # This is the "value" of the coords
        self.value  = value
        # This is the "type" of the coords (i.e. the coordinate chart it belongs to)
        self.chart  = chart

    def __eq__ (self, other:'Coords') -> bool:
        return self.chart == other.chart and np.all(self.value == other.value)

    def __str__ (self) -> str:
        with np.printoptions(linewidth=1000000): # Somewhat of a hack
            return f'{self.value}'

    def __repr__ (self) -> str:
        with np.printoptions(linewidth=1000000): # Somewhat of a hack
            value_string = f'{self.value}'
        if value_string.count('\n') > 0:
            return f'{self.chart.name}{{\n{textwrap.indent(value_string, "    ")}\n}}'
        else:
            return f'{self.chart.name}{{ {value_string} }}'

class BundleCoords(Coords):
    """
    Instances of this class each represent a coordinatized point in a particular trivialized
    chart (i.e. direct product of base and fiber charts) in a particular [fiber] bundle.

    Users should not construct instances of this class directly; use the corresponding
    BundleChart instance instead.

    The `value` attribute is the actual coordinates value.
    The `chart` attribute gives the "type" of the coordinates (i.e. the coordinate chart).

    The base and fiber methods can be used to obtain Coords views into the base
    and fiber components of the bundle element that this instance represents.
    """

    def __init__ (self, value:np.ndarray, *, chart:'BundleChart') -> None:
        if value.shape != chart.coords_shape:
            raise TypeError(f'expected value.shape (which was {value.shape}) to be {chart.coords_shape}')

        Coords.__init__(self, value, chart=chart)

    def base (self) -> Coords:
        assert isinstance(self.chart, BundleChart)
        return self.chart.base_projection(self) # type: ignore

    def fiber (self) -> Coords:
        assert isinstance(self.chart, BundleChart)
        return self.chart.fiber_projection(self) # type: ignore

class VectorBundleCoords(BundleCoords):
    """
    Instances of this class each represent a coordinatized point in a particular trivialized
    chart (i.e. direct product of base and fiber charts) in a particular vector bundle.

    Users should not construct instances of this class directly; use the corresponding
    VectorBundleChart instance instead.

    The `value` attribute is the actual coordinates value.
    The `chart` attribute gives the "type" of the coordinates (i.e. the coordinate chart).

    The base and fiber methods can be used to obtain Coords views into the base
    and fiber components of the vector bundle element that this instance represents.
    """

    # This is just a type-specific override.
    def __init__ (self, value:np.ndarray, *, chart:'VectorBundleChart') -> None:
        BundleCoords.__init__(self, value, chart=chart)

class TangentBundleCoords(VectorBundleCoords):
    """
    Instances of this class each represent a coordinatized point in a particular trivialized
    chart (i.e. direct product of base and fiber charts) in a particular tangent bundle.

    Users should not construct instances of this class directly; use the corresponding
    TangentBundleChart instance instead.

    The `value` attribute is the actual coordinates value.
    The `chart` attribute gives the "type" of the coordinates (i.e. the coordinate chart).

    The base and fiber methods can be used to obtain Coords views into the base
    and fiber components of the tangent bundle element that this instance represents.
    """

    # This is just a type-specific override.
    def __init__ (self, value:np.ndarray, *, chart:'TangentBundleChart') -> None:
        VectorBundleCoords.__init__(self, value, chart=chart)

class CotangentBundleCoords(VectorBundleCoords):
    """
    Instances of this class each represent a coordinatized point in a particular trivialized
    chart (i.e. direct product of base and fiber charts) in a particular cotangent bundle.

    Users should not construct instances of this class directly; use the corresponding
    CotangentBundleChart instance instead.

    The `value` attribute is the actual coordinates value.
    The `chart` attribute gives the "type" of the coordinates (i.e. the coordinate chart).

    The base and fiber methods can be used to obtain Coords views into the base
    and fiber components of the cotangent bundle element that this instance represents.
    """

    # This is just a type-specific override.
    def __init__ (self, value:np.ndarray, *, chart:'CotangentBundleChart') -> None:
        VectorBundleCoords.__init__(self, value, chart=chart)

class TensorBundleCoords(VectorBundleCoords):
    """
    Instances of this class each represent a coordinatized point in a particular trivialized
    chart (i.e. direct product of base and fiber charts) in a particular tensor bundle.

    Users should not construct instances of this class directly; use the corresponding
    TensorBundleChart instance instead.

    The `value` attribute is the actual coordinates value.
    The `chart` attribute gives the "type" of the coordinates (i.e. the coordinate chart).

    The base and fiber methods can be used to obtain Coords views into the base
    and fiber components of the cotangent bundle element that this instance represents.
    """

    # This is just a type-specific override.
    def __init__ (self, value:np.ndarray, *, chart:'CotangentBundleChart') -> None:
        VectorBundleCoords.__init__(self, value, chart=chart)

class PullbackBundleCoords(BundleCoords):
    """
    Instances of this class each represent a coordinatized point in a particular trivialized
    chart (i.e. direct product of base and fiber charts) in a particular pullback bundle.

    Users should not construct instances of this class directly; use the corresponding
    PullbackBundleChart instance instead.

    The `value` attribute is the actual coordinates value.
    The `chart` attribute gives the "type" of the coordinates (i.e. the coordinate chart).

    The base and fiber methods can be used to obtain Coords views into the base
    and fiber components of the cotangent bundle element that this instance represents.
    """

    # This is just a type-specific override.
    def __init__ (self, value:np.ndarray, *, chart:'PullbackBundleChart') -> None:
        BundleCoords.__init__(self, value, chart=chart)

    def target (self) -> BundleCoords:
        assert isinstance(self.chart, PullbackBundleChart)
        return self.chart.target_projection(self)

class PullbackVectorBundleCoords(PullbackBundleCoords, VectorBundleCoords):
    """
    Instances of this class each represent a coordinatized point in a particular trivialized
    chart (i.e. direct product of base and fiber charts) in a particular tensor bundle.

    Users should not construct instances of this class directly; use the corresponding
    TensorBundleChart instance instead.

    The `value` attribute is the actual coordinates value.
    The `chart` attribute gives the "type" of the coordinates (i.e. the coordinate chart).

    The base and fiber methods can be used to obtain Coords views into the base
    and fiber components of the cotangent bundle element that this instance represents.
    """

    # This is just a type-specific override.
    def __init__ (self, value:np.ndarray, *, chart:'PullbackVectorBundleChart') -> None:
        BundleCoords.__init__(self, value, chart=chart)
        # VectorBundleCoords.__init__ doesn't do anything beyond BundleCoords, so no need to call it.

    # This is just a type-specific override.
    def target (self) -> VectorBundleCoords:
        assert isinstance(self.chart, PullbackVectorBundleChart)
        return self.chart.target_projection(self)

class Chart:
    def __init__ (self, *, name:str, coords_shape:typing.Tuple[int,...], symbolic_coords:np.ndarray, coords_class:typing.Any=Coords) -> None:
        if len(name) == 0:
            raise ValueError(f'Expected name to be nonempty')
        if symbolic_coords.shape != coords_shape:
            raise TypeError(f'Expected symbolic_coords.shape (which was {symbolic_coords.shape}) to be coords_shape (which was {coords_shape})')
        if not issubclass(coords_class, Coords):
            raise TypeError(f'Expected coords_class (which was {coords_class}) to be a subclass of {Coords}')

        self.name               = name
        self.dimension          = vorpy.tensor.dimension_of_shape(coords_shape)
        self.coords_shape       = coords_shape
        self.coords_class       = coords_class
        self.symbolic_coords    = self.make_coords(symbolic_coords)

    def verify_chart_type (self, coords:Coords, *, coords_name:str='coords') -> None:
        """Will raise if coords.chart is not equal to self, otherwise will do nothing."""
        if coords.chart is not self and coords.chart != self:
            raise TypeError(f'{coords_name}.chart (which was {coords.chart}) was expected to be {self}')

    def make_coords (self, value:np.ndarray) -> Coords:
        """This constructs a correctly typed Coords from a raw np.ndarray.  The shape must match self.coords_shape."""
        if value.shape != self.coords_shape:
            raise TypeError(f'Expected value.shape (which was {value.shape}) to be self.coords_shape (which was {self.coords_shape})')
        return self.coords_class(value, chart=self)

    def make_coords_uninitialized (self, *, dtype:typing.Any) -> Coords:
        return self.coords_class(np.ndarray(self.coords_shape, dtype=dtype), chart=self)

    def __repr__ (self) -> str:
        return f'Chart({self.name}, coords_shape={self.coords_shape})'

    def __str__ (self) -> str:
        return self.name

    def __eq__ (self, other:'Chart') -> bool:
        # It's not clear if the comparison of symbolic_coords is appropriate.  Use it for now,
        # and only consider changing it if it causes a problem.
        return self.name == other.name and self.dimension == other.dimension and self.coords_shape == other.coords_shape and self.coords_class == other.coords_class and np.all(self.symbolic_coords.value == other.symbolic_coords.value)

class BundleChart(Chart):
    """Represents a trivialized chart (i.e. direct product of base and fiber charts) for a [fiber] bundle."""

    def __init__ (self, *, name:str, base_chart:Chart, fiber_chart:Chart, coords_class:typing.Any=BundleCoords) -> None:
        if not issubclass(coords_class, BundleCoords):
            raise TypeError(f'Expected coords_class (which was {coords_class}) to be a subclass of {BundleCoords}')

        parallel_storage = base_chart.coords_shape == fiber_chart.coords_shape

        if parallel_storage:
            # If the base and fiber chart coords_shape values are the same, then store
            # as shape (2,)+s, where s is the shape of the base/fiber.
            coords_shape = (2,) + base_chart.coords_shape
            symbolic_coords = np.array([base_chart.symbolic_coords.value, fiber_chart.symbolic_coords.value])
        else:
            # Otherwise no such formatted storage is possible, so store the coords as
            # the concatenated 1-tensor views of base and fiber.
            coords_shape            = (base_chart.dimension + fiber_chart.dimension,)
            symbolic_coords         = np.concatenate((base_chart.symbolic_coords.value.reshape(-1), fiber_chart.symbolic_coords.value.reshape(-1)))

        #print(f'HIPPO: symbolic_coords:\n{symbolic_coords}')
        assert symbolic_coords.shape == coords_shape

        Chart.__init__(self, name=name, coords_shape=coords_shape, symbolic_coords=symbolic_coords, coords_class=coords_class)

        self.base_chart         = base_chart
        self.fiber_chart        = fiber_chart
        self._parallel_storage  = parallel_storage

    # Type-specific overload
    def make_coords (self, value:np.ndarray) -> BundleCoords:
        retval = Chart.make_coords(self, value)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, BundleCoords)
        return typing.cast(BundleCoords, retval)

    # Type-specific overload
    def make_coords_uninitialized (self, *, dtype:typing.Any) -> BundleCoords:
        retval = Chart.make_coords_uninitialized(self, dtype=dtype)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, BundleCoords)
        return typing.cast(BundleCoords, retval)

    def make_coords_composed (self, base:Coords, fiber:Coords) -> BundleCoords:
        # TODO: Make this an actual Morphism?
        self.base_chart.verify_chart_type(base, coords_name='base')
        self.fiber_chart.verify_chart_type(fiber, coords_name='fiber')
        if self._parallel_storage:
            return self.make_coords(np.array([base.value, fiber.value]))
        else:
            return self.make_coords(np.concatenate((base.value.reshape(-1), fiber.value.reshape(-1))))

    def base_projection (self, coords:BundleCoords) -> Coords:
        # TODO: Make this an actual Morphism?
        if self._parallel_storage:
            # Easy access as the first row.
            return self.base_chart.make_coords(coords.value[0,...])
        else:
            # Have to slice and reshape the linear storage.
            return self.base_chart.make_coords(coords.value[:self.base_chart.dimension].reshape(self.base_chart.coords_shape))

    def fiber_projection (self, coords:BundleCoords) -> Coords:
        # TODO: Make this an actual Morphism?
        if self._parallel_storage:
            # Easy access as the first row.
            return self.fiber_chart.make_coords(coords.value[1,...])
        else:
            # Have to slice and reshape the linear storage.
            return self.fiber_chart.make_coords(coords.value[self.base_chart.dimension:].reshape(self.fiber_chart.coords_shape))

    def __eq__ (self, other:'BundleChart') -> bool:
        return Chart.__eq__(self, other) and self.base_chart == other.base_chart and self.fiber_chart == other.fiber_chart

class VectorBundleChart(BundleChart):
    """Represents a trivialized chart (i.e. direct product of base and fiber charts) for a vector bundle."""

    def __init__ (self, *, name:str, base_chart:Chart, fiber_chart:Chart, coords_class:typing.Any=VectorBundleCoords) -> None:
        if not issubclass(coords_class, VectorBundleCoords):
            raise TypeError(f'Expected coords_class (which was {coords_class}) to be a subclass of {VectorBundleCoords}')

        BundleChart.__init__(self, name=name, base_chart=base_chart, fiber_chart=fiber_chart, coords_class=coords_class)

    def order (self) -> int:
        """The tensor order of a vector bundle is 1 except when overridden in a base class."""
        return 1

    def factor (self, index:int) -> 'VectorBundleChart':
        """Returns the `index`th tensor factor.  There is only 1 factor, and that factor is this vector bundle itself."""
        if index != 0:
            raise ValueError(f'VectorBundleChart only has 1 tensor factor')
        return self

    # Type-specific overload
    def make_coords (self, value:np.ndarray) -> VectorBundleCoords:
        retval = BundleChart.make_coords(self, value)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, VectorBundleCoords)
        return typing.cast(VectorBundleCoords, retval)

    # Type-specific overload
    def make_coords_uninitialized (self, *, dtype:typing.Any) -> VectorBundleCoords:
        retval = BundleChart.make_coords_uninitialized(self, dtype=dtype)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, VectorBundleCoords)
        return typing.cast(VectorBundleCoords, retval)

    # Type-specific overload
    def make_coords_composed (self, base:Coords, fiber:Coords) -> VectorBundleCoords:
        retval = BundleChart.make_coords_composed(self, base, fiber)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, VectorBundleCoords) # sanity check
        return typing.cast(VectorBundleCoords, retval)

    # Type-specific overload
    def __eq__ (self, other:'VectorBundleChart') -> bool:
        return BundleChart.__eq__(self, other)

class TangentBundleChart(VectorBundleChart):
    """Represents a trivialized chart (i.e. direct product of base and fiber charts) for a tangent bundle."""

    # TODO: Could potentially induce fiber_chart from base_chart (since it's just the vector space version of the base chart)
    def __init__ (self, *, base_chart:Chart, fiber_chart:Chart) -> None:
        if base_chart.coords_shape != fiber_chart.coords_shape:
            raise TypeError(f'Expected base_chart.coords_shape (which was {base_chart.coords_shape}) to be equal to fiber_chart.coords_shape (which was {fiber_chart.coords_shape})')

        # TODO: Maybe make some bool attribute indicating if this TangentBundleChart was induced by the base chart.
        if fiber_chart.name == f'InducedTangentBundleFiberOn({base_chart.name})':
            name                    = f'T({base_chart.name})'
        else:
            name                    = f'T({base_chart.name}, fiber={fiber_chart.name})'

        VectorBundleChart.__init__(self, name=name, base_chart=base_chart, fiber_chart=fiber_chart, coords_class=TangentBundleCoords)

    # Type-specific overload
    def make_coords (self, value:np.ndarray) -> TangentBundleCoords:
        retval = VectorBundleChart.make_coords(self, value)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, TangentBundleCoords)
        return typing.cast(TangentBundleCoords, retval)

    # Type-specific overload
    def make_coords_uninitialized (self, *, dtype:typing.Any) -> TangentBundleCoords:
        retval = VectorBundleChart.make_coords_uninitialized(self, dtype=dtype)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, TangentBundleCoords)
        return typing.cast(TangentBundleCoords, retval)

    # Type-specific overload
    def make_coords_composed (self, base:Coords, fiber:Coords) -> TangentBundleCoords:
        retval = VectorBundleChart.make_coords_composed(self, base, fiber)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, TangentBundleCoords) # sanity check
        return typing.cast(TangentBundleCoords, retval)

    # Type-specific overload
    def __eq__ (self, other:'TangentBundleChart') -> bool:
        return VectorBundleChart.__eq__(self, other)

    @staticmethod
    def induced (base_chart:Chart, *, fiber_symbolic_coords__o:typing.Optional[np.ndarray]=None) -> 'TangentBundleChart':
        """
        Constructs the tangent bundle chart induced by a chart on its base manifold.

        Optionally specify the symbolic coordinate names via fiber_symbolic_coords__o.  The
        default is prefixed_symbolic_tensor('v_', base_chart.symbolic_coords).
        """

        if fiber_symbolic_coords__o is None:
            fiber_symbolic_coords = prefixed_symbolic_tensor('v_', base_chart.symbolic_coords.value)
        else:
            fiber_symbolic_coords = fiber_symbolic_coords__o

        fiber_chart = Chart(
            name=f'InducedTangentBundleFiberOn({base_chart.name})', # Sentinel name used in the name of the Chart
            coords_shape=base_chart.coords_shape,
            symbolic_coords=fiber_symbolic_coords,
            coords_class=base_chart.coords_class,
        )
        return TangentBundleChart(base_chart=base_chart, fiber_chart=fiber_chart)

class CotangentBundleChart(VectorBundleChart):
    """Represents a trivialized chart (i.e. direct product of base and fiber charts) for a cotangent bundle."""

    # TODO: Could potentially induce fiber_chart from base_chart (since it's just the vector space version of the base chart)
    def __init__ (self, *, base_chart:Chart, fiber_chart:Chart) -> None:
        if base_chart.coords_shape != fiber_chart.coords_shape:
            raise TypeError(f'Expected base_chart.coords_shape (which was {base_chart.coords_shape}) to be equal to fiber_chart.coords_shape (which was {fiber_chart.coords_shape})')

        # TODO: Maybe make some bool attribute indicating if this CotangentBundleChart was induced by the base chart.
        if fiber_chart.name == f'InducedCotangentBundleFiberOn({base_chart.name})':
            name                    = f'T^{{*}}({base_chart.name})'
        else:
            name                    = f'T^{{*}}({base_chart.name}, fiber={fiber_chart.name})'

        VectorBundleChart.__init__(self, name=name, base_chart=base_chart, fiber_chart=fiber_chart, coords_class=CotangentBundleCoords)

    # Type-specific overload
    def make_coords (self, value:np.ndarray) -> CotangentBundleCoords:
        retval = VectorBundleChart.make_coords(self, value)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, CotangentBundleCoords)
        return typing.cast(CotangentBundleCoords, retval)

    # Type-specific overload
    def make_coords_uninitialized (self, *, dtype:typing.Any) -> CotangentBundleCoords:
        retval = VectorBundleChart.make_coords_uninitialized(self, dtype=dtype)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, CotangentBundleCoords)
        return typing.cast(CotangentBundleCoords, retval)

    # Type-specific overload
    def make_coords_composed (self, base:Coords, fiber:Coords) -> CotangentBundleCoords:
        retval = VectorBundleChart.make_coords_composed(self, base, fiber)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, CotangentBundleCoords) # sanity check
        return typing.cast(CotangentBundleCoords, retval)

    # Type-specific overload
    def __eq__ (self, other:'CotangentBundleChart') -> bool:
        return VectorBundleChart.__eq__(self, other)

    @staticmethod
    def induced (base_chart:Chart, *, fiber_symbolic_coords__o:typing.Optional[np.ndarray]=None) -> 'CotangentBundleChart':
        """
        Constructs the cotangent bundle chart induced by a chart on its base manifold.

        Optionally specify the symbolic coordinate names via fiber_symbolic_coords__o.  The
        default is prefixed_symbolic_tensor('p_', base_chart.symbolic_coords).
        """

        if fiber_symbolic_coords__o is None:
            fiber_symbolic_coords = prefixed_symbolic_tensor('p_', base_chart.symbolic_coords.value)
        else:
            fiber_symbolic_coords = fiber_symbolic_coords__o

        fiber_chart = Chart(
            name=f'InducedCotangentBundleFiberOn({base_chart.name})', # Sentinel name used in the name of the Chart
            coords_shape=base_chart.coords_shape,
            symbolic_coords=fiber_symbolic_coords,
            coords_class=base_chart.coords_class,
        )
        return CotangentBundleChart(base_chart=base_chart, fiber_chart=fiber_chart)

class TensorBundleChart(VectorBundleChart):
    def __init__ (self, *vector_bundle_chart__v:VectorBundleChart, fiber_symbolic_coords:np.ndarray) -> None:
        #print(f'TensorBundleChart(vector_bundle_chart__v = {vector_bundle_chart__v})')

        if len(vector_bundle_chart__v) == 0:
            raise NotImplementedError(f'0-tensor bundles, while mathematically valid, are not yet implemented')
            # TODO: Specification of scalar field and base chart.
        else:
            #print(f'HIPPO: vector_bundle_chart__v = {vector_bundle_chart__v}')
            if any(vector_bundle_chart__v[0].base_chart != vector_bundle_chart.base_chart for vector_bundle_chart in vector_bundle_chart__v):
                raise TypeError(f'All VectorBundleChart factors in TensorBundleChart must have the same base_chart.')

            name = f'TensorProduct({", ".join(vector_bundle_chart.name for vector_bundle_chart in vector_bundle_chart__v)})'
            base_chart = vector_bundle_chart__v[0].base_chart

            fiber_chart_name = f'TensorProduct({", ".join(vector_bundle_chart.fiber_chart.name for vector_bundle_chart in vector_bundle_chart__v)})'
            fiber_coords_shape = sum((vector_bundle_chart.fiber_chart.coords_shape for vector_bundle_chart in vector_bundle_chart__v), tuple())
            fiber_chart = Chart(
                name=fiber_chart_name,
                coords_shape=fiber_coords_shape,
                symbolic_coords=fiber_symbolic_coords,
                coords_class=Coords, # TODO: TensorProduct of vector spaces
            )

            VectorBundleChart.__init__(self, name=name, base_chart=base_chart, fiber_chart=fiber_chart, coords_class=TensorBundleCoords)

            self.vector_bundle_chart__v = vector_bundle_chart__v

    def order (self) -> int:
        """Returns the order of the tensor bundle (i.e. how many tensor factors there are)."""
        return len(self.vector_bundle_chart__v)

    def factor (self, index:int) -> VectorBundleChart:
        """Returns the `index`th tensor factor."""
        return self.vector_bundle_chart__v[index]

    # Type-specific overload
    def __eq__ (self, other:'TensorBundleChart') -> bool:
        return VectorBundleChart.__eq__(self, other) and self.vector_bundle_chart__v == other.vector_bundle_chart__v

class PullbackBundleChart(BundleChart):
    def __init__ (self, *, pullback_morphism:'Morphism', target_bundle_chart:BundleChart, coords_class=PullbackBundleCoords) -> None:
        if pullback_morphism.codomain != target_bundle_chart.base_chart:
            raise TypeError(f'Expected pullback_morphism.codomain (which was {pullback_morphism.codomain}) to be equal to target_bundle_chart.base_chart (which was {target_bundle_chart.base_chart})')

        assert issubclass(coords_class, PullbackBundleCoords)

        BundleChart.__init__(
            self,
            name=f'{pullback_morphism}^{{*}}{target_bundle_chart}',
            base_chart=pullback_morphism.domain,
            fiber_chart=target_bundle_chart.fiber_chart,
            coords_class=coords_class,
        )

        self.pullback_morphism      = pullback_morphism
        self.target_bundle_chart    = target_bundle_chart

    # Type-specific overload
    def make_coords (self, value:np.ndarray) -> PullbackBundleCoords:
        retval = BundleChart.make_coords(self, value)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, PullbackBundleCoords)
        return typing.cast(PullbackBundleCoords, retval)

    # Type-specific overload
    def make_coords_uninitialized (self, *, dtype:typing.Any) -> PullbackBundleCoords:
        retval = BundleChart.make_coords_uninitialized(self, dtype=dtype)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, PullbackBundleCoords)
        return typing.cast(PullbackBundleCoords, retval)

    # Type-specific overload
    def make_coords_composed (self, base:Coords, fiber:Coords) -> PullbackBundleCoords:
        self.target_bundle_chart.fiber_chart.verify_chart_type(fiber)
        retval = BundleChart.make_coords_composed(self, base, fiber)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, PullbackBundleCoords) # sanity check
        return typing.cast(PullbackBundleCoords, retval)

    def target_projection (self, coords:PullbackBundleCoords) -> BundleCoords:
        return self.target_bundle_chart.make_coords_composed(self.pullback_morphism(coords.base()), coords.fiber())

    # Type-specific overload
    def __eq__ (self, other:'PullbackBundleChart') -> bool:
        return BundleChart.__eq__(self, other) and self.pullback_morphism == other.pullback_morphism and self.target_bundle_chart == other.target_bundle_chart

class PullbackVectorBundleChart(PullbackBundleChart, VectorBundleChart):
    # This is just a type-specific override.
    def __init__ (self, *, pullback_morphism:'Morphism', target_bundle_chart:VectorBundleChart) -> None:
        PullbackBundleChart.__init__(self, pullback_morphism=pullback_morphism, target_bundle_chart=target_bundle_chart, coords_class=PullbackVectorBundleCoords)
        # No need to call VectorBundleChart.__init__ because PullbackBundleChart.__init__ takes care of everything
        # (this fact may change in the future).

    # Type-specific overload
    def make_coords (self, value:np.ndarray) -> PullbackVectorBundleCoords:
        retval = PullbackBundleChart.make_coords(self, value)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, PullbackVectorBundleCoords)
        return typing.cast(PullbackVectorBundleCoords, retval)

    # Type-specific overload
    def make_coords_uninitialized (self, *, dtype:typing.Any) -> PullbackVectorBundleCoords:
        retval = PullbackBundleChart.make_coords_uninitialized(self, dtype=dtype)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, PullbackVectorBundleCoords)
        return typing.cast(PullbackVectorBundleCoords, retval)

    # Type-specific overload
    def make_coords_composed (self, base:Coords, fiber:Coords) -> PullbackVectorBundleCoords:
        self.target_bundle_chart.fiber_chart.verify_chart_type(fiber)
        retval = PullbackBundleChart.make_coords_composed(self, base, fiber)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, PullbackVectorBundleCoords) # sanity check
        return typing.cast(PullbackVectorBundleCoords, retval)

    # This is just a type-specific override.
    def fiber_projection (self, coords:PullbackVectorBundleCoords) -> VectorBundleCoords:
        return PullbackBundleChart.fiber_projection(self, coords)

    # Type-specific overload
    def target_projection (self, coords:PullbackVectorBundleCoords) -> VectorBundleCoords:
        retval = PullbackBundleChart.target_projection(self, coords)
        assert isinstance(retval, VectorBundleCoords)
        return retval

    # Type-specific overload
    def __eq__ (self, other:'PullbackVectorBundleChart') -> bool:
        return PullbackBundleChart.__eq__(self, other) and VectorBundleChart.__eq__(self, other)

class Morphism:
    """Morphism in the category of coordinatized manifolds."""

    def __init__ (
        self,
        *,
        name:str,
        domain:Chart,
        codomain:Chart,
        evaluator:typing.Callable[[Coords],Coords],
    ) -> None:
        self.name       = name
        self.domain     = domain
        self.codomain   = codomain
        self.evaluator  = evaluator

    def __call__ (self, coords:Coords) -> Coords:
        self.domain.verify_chart_type(coords, coords_name='coords')
        retval = self.evaluator(coords)
        self.codomain.verify_chart_type(retval, coords_name=f'return value of evaluator')
        return retval

    def __repr__ (self) -> str:
        return f'{self.name} : {self.__class__.__name__}({self.domain} -> {self.codomain})'

    def __str__ (self) -> str:
        return self.name

    def __eq__ (self, other:'Morphism') -> bool:
        """Equal iff domain and codomain match and the evaluation produces the same thing on the domain chart's symbolic_coords."""

        if self.domain != other.domain or self.codomain != other.codomain:
            return False

        x = self.domain.symbolic_coords
        self_x = self(x)
        other_x = other(x)
        return self_x == other_x

class Isomorphism(Morphism):
    """Isomorphism in the category of coordinatized manifolds."""

    def __init__ (
        self,
        *,
        name:str,
        domain:Chart,
        codomain:Chart,
        evaluator:typing.Callable[[Coords],Coords],
        inverse_evaluator__o:typing.Optional[typing.Callable[[Coords],Coords]]=None,
    ) -> None:
        """
        If inverse_evaluator__o is not specified, then sympy.solve will be used to attempt
        to symbolically compute the inverse.  If this doesn't find a unique solution, then
        it will raise an exception, meaning that the inverse evaluator must be specified
        explicitly.
        """

        if domain.dimension != codomain.dimension:
            raise TypeError(f'domain and codomain must have the same dimension for Isomorphism.')

        Morphism.__init__(self, name=name, domain=domain, codomain=codomain, evaluator=evaluator)

        if inverse_evaluator__o is not None:
            self.inverse_evaluator = inverse_evaluator__o
        else:
            # Attempt to solve `y = f(x)` for x symbolically, giving x = f^{-1}(y).
            x = self.domain.symbolic_coords
            y = self.codomain.symbolic_coords
            solution_v = sp.solve((y.value - evaluator(x).value).reshape(-1), *x.value.reshape(-1).tolist())
            print(f'solution_v:')
            for solution in solution_v:
                print(f'    {solution}')
            if len(solution_v) != 1:
                raise ValueError(f'sympy.solve did not automatically find a unique solution to the inverse coordinate change; inverse_evaluator__o should be specified to define the inverse explicitly..  solution_v = {solution_v}')
            solution = np.array(solution_v[0]).reshape(x.value.shape)

            def inverse_evaluator_ (coords:Coords) -> Coords:
                #return codomain.make_coords(np.array(sp.Subs(solution_v[0], y.value, coords.value).doit()).reshape(x.value.shape))
                return codomain.make_coords(substitution(solution, y.value, coords.value))

            self.inverse_evaluator = inverse_evaluator_

    def evaluate_inverse (self, coords:Coords) -> Coords:
        self.codomain.verify_chart_type(coords, coords_name='coords')
        retval = self.inverse_evaluator(coords)
        self.domain.verify_chart_type(retval, coords_name=f'return value of inverse_evaluator')
        return retval

    def inverse (self) -> 'Isomorphism':
        return Isomorphism(
            name=f'{self.name}^{{-1}}',
            domain=self.codomain,
            codomain=self.domain,
            evaluator=self.inverse_evaluator,
            inverse_evaluator__o=self.evaluator,
        )

    def __eq__ (self, other:'Isomorphism') -> bool:
        """Same as Morphism.__eq__ but also checks inverse."""

        if not Morphism.__eq__(self, other):
            return False

        y = self.codomain.symbolic_coords
        self_y = self.evaluate_inverse(y)
        other_y = other.evaluate_inverse(y)
        return self_y == other_y

def identity_isomorphism (chart:Chart) -> Isomorphism:
    return Isomorphism(
        name=f'Identity_{{{chart}}}',
        domain=chart,
        codomain=chart,
        evaluator=lambda x:x,
        inverse_evaluator__o=lambda x:y,
    )

class BundleSection(Morphism):
    def __init__ (self, *, name:str, bundle_chart:BundleChart, fiber_evaluator:typing.Callable[[Coords],Coords]) -> None:
        domain = bundle_chart.base_chart
        codomain = bundle_chart

        def evaluator (base:Coords) -> BundleCoords:
            return codomain.make_coords_composed(base, fiber_evaluator(base))

        Morphism.__init__(self, name=name, domain=domain, codomain=codomain, evaluator=evaluator)

        self.bundle_chart       = bundle_chart
        self.fiber_evaluator    = fiber_evaluator

    # This is more than a type-specific override.
    def __call__ (self, coords:Coords) -> BundleCoords:
        retval = Morphism.__call__(self, coords)
        if retval.base() != coords:
            raise ValueError(f'If S is a section of a bundle, then for a point p in the bundle\'s base, S(p) (which in this case was {retval.base()}) must have basepoint p (which is this case was {coords})')
        assert isinstance(retval, BundleCoords)
        return retval

class VectorBundleSection(BundleSection):
    # Type-specific override
    def __init__ (self, *, name:str, bundle_chart:VectorBundleChart, fiber_evaluator:typing.Callable[[Coords],Coords]) -> None:
        BundleSection.__init__(self, name=name, bundle_chart=bundle_chart, fiber_evaluator=fiber_evaluator)

class TangentBundleSection(VectorBundleSection):
    # Type-specific override
    def __init__ (self, *, name:str, bundle_chart:TangentBundleChart, fiber_evaluator:typing.Callable[[Coords],Coords]) -> None:
        VectorBundleSection.__init__(self, name=name, bundle_chart=bundle_chart, fiber_evaluator=fiber_evaluator)

class CotangentBundleSection(VectorBundleSection):
    # Type-specific override
    def __init__ (self, *, name:str, bundle_chart:CotangentBundleChart, fiber_evaluator:typing.Callable[[Coords],Coords]) -> None:
        VectorBundleSection.__init__(self, name=name, bundle_chart=bundle_chart, fiber_evaluator=fiber_evaluator)

class TensorBundleSection(VectorBundleSection):
    # Type-specific override
    def __init__ (self, *, name:str, bundle_chart:TensorBundleChart, fiber_evaluator:typing.Callable[[Coords],Coords]) -> None:
        VectorBundleSection.__init__(self, name=name, bundle_chart=bundle_chart, fiber_evaluator=fiber_evaluator)

def jacobian (morphism:Morphism) -> TensorBundleSection:
    # If morphism : A -> B, then the Jacobian is a tensor field which is a section of morphism^{*}TB \otimes T^{*}A.
    linear_morphism_bundle_chart = TensorBundleChart(
        PullbackVectorBundleChart(pullback_morphism=morphism, target_bundle_chart=TangentFunctor_ob(morphism.codomain)),
        CotangentFunctor_ob(morphism.domain),
        fiber_symbolic_coords=vorpy.symbolic.tensor('J', morphism.codomain.coords_shape + morphism.domain.coords_shape),
    )
    #print(f'jacobian; linear_morphism_bundle_chart = {linear_morphism_bundle_chart}')

    # Compute the Jacobian and define the fiber_evaluator for it.

    x = morphism.domain.symbolic_coords
    y = morphism(x)
    J_x = vorpy.symbolic.differential(y.value, x.value)
    assert J_x.shape == y.value.shape + x.value.shape
    assert J_x.shape == linear_morphism_bundle_chart.fiber_chart.coords_shape

    def jacobian_fiber_evaluator (base:Coords) -> Coords:
        #return linear_morphism_bundle_chart.fiber_chart.make_coords(simplified(np.array(sp.Subs(J_x, x.value, base.value).doit()).reshape(linear_morphism_bundle_chart.fiber_chart.coords_shape)))
        return linear_morphism_bundle_chart.fiber_chart.make_coords(substitution(J_x, x.value, base.value))

    retval = TensorBundleSection(
        name=f'Jacobian({morphism})',
        bundle_chart=linear_morphism_bundle_chart,
        fiber_evaluator=jacobian_fiber_evaluator,
    )
    assert isinstance(retval, Morphism)
    assert retval.domain == morphism.domain
    return retval

class BundleMorphism(Morphism):
    """
    Let F denote the vector bundle morphism and let f denote the base morphism.
    The following diagram should commute.

             domain    -- F -->    codomain
                |                     |
                |                     |
            baseproj              baseproj
                |                     |
                V                     V
           domain.base -- f --> codomain.base
    """

    #def __init__ (self, morphism:Morphism) -> None:
        #require.is_instance(morphism.domain, BundleChart)
        #require.is_instance(morphism.codomain, BundleChart)

        #Morphism.__init__(self, domain=morphism.domain, codomain=morphism.codomain, evaluator=morphism.evaluator)

        #x = morphism.domain.symbolic_coords
        #y = morphism(x).base()
        ## Verify that there exists a base morphism (i.e. the base projection of the morphism's image
        ## produces depending only on the base of the domain).
        #assert np.all(vorpy.symbolic.differential(y.value, x.fiber().value) == 0)

        #base_morphism_domain = morphism.domain.base_chart
        #base_morphism_codomain = morphism.codomain.base_chart

        #def base_morphism_evaluator (coords:Coords) -> Coords:
            #return base_morphism_codomain.make_coords(substitute(y.value, x.base().value, coords.value))

        #self.base_morphism = Morphism(
            #name=f'BaseMorphismOf({morphism.name})',
            #domain=base_morphism_domain,
            #codomain=base_morphism_codomain,
            #evaluator=base_morphism_evaluator,
        #)

        #fiber_morphism_domain = morphism.domain
        #fiber_morphism_codomain = morphism.codomain.fiber_chart

        #def fiber_morphism_evaluator (bundle_coords:BundleCoords) -> Coords:
            #return morphism(bundle_coords).fiber()

        #self.fiber_morphism = Morphism(
            #name=f'FiberMorphismOf({morphism.name})',
            #domain=fiber_morphism_domain,
            #codomain=fiber_morphism_codomain,
            #evaluator=fiber_morphism_evaluator,
        #)

    def __init__ (
        self,
        *,
        name:str,
        domain:BundleChart,
        codomain:BundleChart,
        evaluator:typing.Callable[[BundleCoords],BundleCoords],
        base_morphism__o:typing.Optional[Morphism]=None,
    ) -> None:
        """
        If base_morphism__o is not None, then evaluator will be checked to verify that its induced
        base morphism is equal to base_morphism__o (symbolically).
        """

        Morphism.__init__(
            self,
            name=name,
            domain=domain,
            codomain=codomain,
            evaluator=evaluator,
        )

        if base_morphism__o is not None:
            base_morphism = base_morphism__o
        else:
            base_morphism = Morphism(
                name=f'BaseMorphismOf({name})',
                domain=domain.base_chart,
                codomain=codomain.base_chart,
                evaluator=lambda v:codomain.base_projection(self.evaluator(v)), # This relies on the commutativity of the diagram
            )

        self.base_morphism = base_morphism
        # It's not necessarily well-defined to map domain.fiber_chart to codomain.fiber_chart,
        # since the map may depend on the basepoint portion of the domain values.
        self.fiber_morphism = Morphism(
            name=f'FiberMorphismOf({name})',
            domain=domain,
            codomain=codomain.fiber_chart,
            evaluator=lambda v:codomain.fiber_projection(self.evaluator(v)),
        )

        # Verify symbolically that the diagram commutes.
        leg0 = base_morphism(domain.symbolic_coords.base())
        leg1 = self(domain.symbolic_coords).base()
        if not np.all(leg0 == leg1):
            raise ValueError(f'Expected evaluator to produce values in the fiber "above" the image of the base morphism, but base_morphism(domain.symbolic_coords.base()) was {leg0} and self(domain.symbolic_coords).base() was {leg1}; i.e. the diagram defining a bundle morphism did not commute.')

    #def __init__ (
        #self,
        #*,
        #base_morphism:Morphism,
        #fiber_morphism:Morphism,
        #expected_codomain__o:typing.Optional[BundleChart]=None,
    #) -> None:
        #"""
        #If expected_codomain__o is not None, then a type check will be done that the bundle point (base,fiber),
        #whose image under this BundleMorphism is (base_morphism(base), fiber_morphism(base, fiber)), actually
        #maps to expected_codomain__o.  Otherwise, the codomain will be inferred from base_morphism and
        #fiber_morphism.
        #"""

        #if not isinstance(fiber_morphism.domain, BundleChart):
            #raise TypeError(f'Expected fiber_morphism.domain (which was {fiber_morphism.domain}) to be an instance of BundleChart')
        #if base_morphism.domain != fiber_morphism.domain.base_chart:
            #raise TypeError(f'Expected base_morphism.domain (which was {base_morphism.domain}) to be equal to fiber_morphism.domain.base_chart (which was {fiber_morphism.domain.base_chart})')

        #if expected_codomain__o is not None:
            #if base_morphism.codomain != expected_codomain__o.base_chart:
                #raise TypeError(f'Expected base_morphism.codomain (which was {base_morphism.codomain}) to be equal to expected_codomain__o.base_chart (which was {expected_codomain__o.base_chart})')
            #if fiber_morphism.codomain != expected_codomain__o.fiber_chart:
                #raise TypeError(f'Expected fiber_morphism.codomain (which was {fiber_morphism.codomain}) to be equal to expected_codomain__o.fiber_chart (which was {expected_codomain__o.fiber_chart})')
            #codomain = expected_codomain__o
        #else:
            #codomain = BundleChart(
                #name=f'Bundle(base={base_morphism.codomain}, fiber={fiber_morphism.codomain})',
                #base_chart=base_morphism.codomain,
                #fiber_chart=fiber_morphism.codomain,
                #coords_class=coords_class,
            #)

        #def evaluator (bundle_coords:BundleCoords) -> BundleCoords:
            #return codomain.make_coords_composed(
                #base_morphism(bundle_coords.base()),
                #fiber_morphism(bundle_coords),
            #)

        #Morphism.__init__(domain=domain, codomain=codomain, evaluator=evaluator)

        #self.base_morphism = base_morphism
        #self.fiber_morphism = fiber_morphism

        ## Verify symbolically that the diagram commutes.
        #leg0 = base_morphism(domain.symbolic_coords.base())
        #leg1 = self(domain.symbolic_coords).base()
        #if not np.all(leg0 == leg1):
            #raise ValueError(f'Expected evaluator to produce values in the fiber "above" the image of the base morphism, but base_morphism(domain.symbolic_coords.base()) was {leg0} and self(domain.symbolic_coords).base() was {leg1}; i.e. the diagram defining a bundle morphism did not commute.')

class BundleIsomorphism(BundleMorphism, Isomorphism):
    def __init__ (
        self,
        *,
        name:str,
        domain:BundleChart,
        codomain:BundleChart,
        evaluator:typing.Callable[[BundleCoords],BundleCoords],
        inverse_evaluator__o:typing.Optional[typing.Callable[[BundleCoords],BundleCoords]]=None,
        base_isomorphism__o:typing.Optional[Isomorphism]=None,
    ) -> None:
        BundleMorphism.__init__(
            self,
            name=name,
            domain=isomorphism.domain,
            codomain=isomorphism.codomain,
            evaluator=isomorphism.evaluator,
            base_morphism__o=base_isomorphism__o,
        )

        # This is normally done by Isomorphism.__init__ but we're not going to call that because
        # it's redundant with parts of BundleMorphism.
        self.inverse_evaluator = isomorphism.inverse_evaluator

        x = domain.symbolic_coords
        y = self(x).base()
        # Verify that there exists a base isomorphism (i.e. the base projection of the isomorphism's image
        # produces a value depending only on the base of the domain).
        assert np.all(vorpy.symbolic.differential(y.value, x.fiber().value) == 0)

        v = codomain.symbolic_coords
        u = self.evaluate_inverse(v).base()
        # Verify that there exists an inverse base isomorphism (i.e. the base projection of the inverse isomorphism's
        # image produces a value depending only on the base of the codomain).
        assert np.all(vorpy.symbolic.differential(u.value, v.fiber().value) == 0)

        base_isomorphism_domain = domain.base_chart
        base_isomorphism_codomain = codomain.base_chart

        def base_isomorphism_evaluator (coords:Coords) -> Coords:
            return base_isomorphism_codomain.make_coords(substitute(y.value, x.base().value, coords.value))

        def base_isomorphism_inverse_evaluator (coords:Coords) -> Coords:
            return base_isomorphism_domain.make_coords(substitute(u.value, v.base().value, coords.value))

        self.base_isomorphism = Isomorphism(
            name=f'BaseIsomorphismOf({isomorphism.name})',
            domain=base_isomorphism_domain,
            codomain=base_isomorphism_codomain,
            evaluator=base_isomorphism_evaluator,
            inverse_evaluator__o=base_isomorphism_inverse_evaluator,
        )
        self.base_morphism = self.base_isomorphism

        fiber_isomorphism_domain = domain
        fiber_isomorphism_codomain = codomain.fiber_chart

        def fiber_isomorphism_evaluator (bundle_coords:BundleCoords) -> Coords:
            return isomorphism(bundle_coords).fiber()

        def fiber_isomorphism_inverse_evaluator (bundle_coords:BundleCoords) -> Coords:
            return isomorphism.evaluate_inverse(bundle_coords).fiber()

        self.fiber_isomorphism = Isomorphism(
            name=f'FiberIsomorphismOf({name})',
            domain=fiber_isomorphism_domain,
            codomain=fiber_isomorphism_codomain,
            evaluator=fiber_isomorphism_evaluator,
            inverse_evaluator__o=fiber_isomorphism_evaluator,
        )
        self.fiber_morphism = self.fiber_isomorphism

    #def __init__ (self, isomorphism:Isomorphism) -> None:
        #require.is_instance(isomorphism.domain, BundleChart)
        #require.is_instance(isomorphism.codomain, BundleChart)

        #BundleMorphism.__init__(self, domain=isomorphism.domain, codomain=isomorphism.codomain, evaluator=isomorphism.evaluator)

        #self.inverse_evaluator = isomorphism.inverse_evaluator

        #x = isomorphism.domain.symbolic_coords
        #y = isomorphism(x).base()
        ## Verify that there exists a base isomorphism (i.e. the base projection of the isomorphism's image
        ## produces a value depending only on the base of the domain).
        #assert np.all(vorpy.symbolic.differential(y.value, x.fiber().value) == 0)

        #v = isomorphism.codomain.symbolic_coords
        #u = isomorphism.evaluate_inverse(v).base()
        ## Verify that there exists an inverse base isomorphism (i.e. the base projection of the inverse isomorphism's
        ## image produces a value depending only on the base of the codomain).
        #assert np.all(vorpy.symbolic.differential(u.value, v.fiber().value) == 0)

        #base_isomorphism_domain = isomorphism.domain.base_chart
        #base_isomorphism_codomain = isomorphism.codomain.base_chart

        #def base_isomorphism_evaluator (coords:Coords) -> Coords:
            #return base_isomorphism_codomain.make_coords(substitute(y.value, x.base().value, coords.value))

        #def base_isomorphism_inverse_evaluator (coords:Coords) -> Coords:
            #return base_isomorphism_domain.make_coords(substitute(u.value, v.base().value, coords.value))

        #self.base_isomorphism = Isomorphism(
            #name=f'BaseMorphismOf({isomorphism.name})',
            #domain=base_isomorphism_domain,
            #codomain=base_isomorphism_codomain,
            #evaluator=base_isomorphism_evaluator,
            #inverse_evaluator__o=base_isomorphism_inverse_evaluator
        #)

        #fiber_isomorphism_domain = isomorphism.domain
        #fiber_isomorphism_codomain = isomorphism.codomain.fiber_chart

        #def fiber_isomorphism_evaluator (bundle_coords:BundleCoords) -> Coords:
            #return isomorphism(bundle_coords).fiber()

        #def fiber_isomorphism_inverse_evaluator (bundle_coords:BundleCoords) -> Coords:
            #return isomorphism.evaluate_inverse(bundle_coords).fiber()

        #self.fiber_isomorphism = Isomorphism(
            #name=f'FiberMorphismOf({isomorphism.name})',
            #domain=fiber_isomorphism_domain,
            #codomain=fiber_isomorphism_codomain,
            #evaluator=fiber_isomorphism_evaluator,
            #inverse_evaluator__o=fiber_isomorphism_evaluator,
        #)

    def evaluate_inverse (self, coords:BundleCoords) -> BundleCoords:
        self.codomain.verify_chart_type(coords, coords_name='coords')
        retval = self.inverse_evaluator(coords)
        self.domain.verify_chart_type(retval, coords_name=f'return value of inverse_evaluator')
        return retval

    # Type-specific override
    def inverse (self) -> 'BundleIsomorphism':
        return BundleIsomorphism(
            name=f'{self.name}^{{-1}}',
            domain=self.codomain,
            codomain=self.domain,
            evaluator=self.inverse_evaluator,
            inverse_evaluator__o=self.evaluator,
            base_isomorphism__o=self.base_isomorphism.inverse(),
        )

#class BundleMorphismOver:
    #"""This represents the space of bundle morphisms between two bundles over a particular base morphism."""

    #def __init__ (self, *, domain:BundleChart, codomain:BundleChart, base_morphism:Morphism) -> None:
        #if base_morphism.domain != domain.base_chart:
            #raise TypeError(f'Expected base_morphism.domain (which was {base_morphism.domain}) to be equal to domain.base_chart (which was {domain.base_chart})')
        #if base_morphism.codomain != codomain.base_chart:
            #raise TypeError(f'Expected base_morphism.codomain (which was {base_morphism.codomain}) to be equal to codomain.base_chart (which was {codomain.base_chart})')

        #self.domain = domain
        #self.codomain = codomain
        #self.base_morphism = base_morphism

class VectorBundleMorphism(BundleMorphism):
    def __init__ (
        self,
        *,
        name:str,
        domain:BundleChart,
        codomain:BundleChart,
        evaluator:typing.Callable[[BundleCoords],BundleCoords],
        base_morphism__o:typing.Optional[Morphism]=None,
    ) -> None:
        BundleMorphism.__init__(
            self,
            name=name,
            domain=domain,
            codomain=codomain,
            evaluator=evaluator,
            base_morphism__o=base_morphism__o,
        )

    # Nothing else appears to be needed here.

class VectorBundleIsomorphism(VectorBundleMorphism, BundleIsomorphism):
    def __init__ (
        self,
        *,
        name:str,
        domain:BundleChart,
        codomain:BundleChart,
        evaluator:typing.Callable[[BundleCoords],BundleCoords],
        inverse_evaluator__o:typing.Optional[typing.Callable[[BundleCoords],BundleCoords]]=None,
        base_isomorphism__o:typing.Optional[Isomorphism]=None,
    ) -> None:
        BundleIsomorphism.__init__(
            self,
            name=name,
            domain=domain,
            codomain=codomain,
            evaluator=evaluator,
            inverse_evaluator__o=inverse_evaluator__o,
            base_isomorphism__o=base_isomorphism__o,
        )

# TODO: Use mypy's overloading capabilities to combine TangentFunctor_ob and TangentFunctor_mor and
# TangentFunctor_iso into a single one called TangentFunctor or just T.
def TangentFunctor_ob (base_chart:Chart, *, fiber_symbolic_coords__o:typing.Optional[np.ndarray]=None) -> TangentBundleChart:
    """This is the tangent functor's action on objects of the category (i.e. Chart)."""
    return TangentBundleChart.induced(base_chart, fiber_symbolic_coords__o=fiber_symbolic_coords__o)

def CotangentFunctor_ob (base_chart:Chart, *, fiber_symbolic_coords__o:typing.Optional[np.ndarray]=None) -> CotangentBundleChart:
    """This is the cotangent functor's action on objects of the category (i.e. Chart)."""
    return CotangentBundleChart.induced(base_chart, fiber_symbolic_coords__o=fiber_symbolic_coords__o)

def TangentFunctor_mor (morphism:Morphism) -> VectorBundleMorphism:
    """This is the tangent functor's action on morphisms of the category (i.e. Morphism)."""

    J = jacobian(morphism)

    domain = TangentFunctor_ob(morphism.domain)
    codomain = TangentFunctor_ob(morphism.codomain)

    def evaluator (bundle_coords:BundleCoords) -> BundleCoords:
        return codomain.make_coords_composed(
            morphism(bundle_coords.base()),
            codomain.fiber_chart.make_coords(
                np.dot(J(bundle_coords.base()).value, bundle_coords.fiber().value)
            ),
        )

    return VectorBundleMorphism(
        name=f'T({morphism.name})',
        domain=domain,
        codomain=codomain,
        evaluator=evaluator,
        base_morphism__o=morphism,
    )

def CotangentFunctor_mor (morphism:Morphism) -> VectorBundleMorphism:
    """This is the tangent functor's action on morphisms of the category (i.e. Morphism)."""

    J = jacobian(morphism)

    domain = PullbackVectorBundleChart(pullback_morphism=morphism, target_bundle_chart=CotangentFunctor_ob(morphism.codomain))
    codomain = CotangentFunctor_ob(morphism.domain)

    def evaluator (bundle_coords:BundleCoords) -> BundleCoords:
        domain.verify_chart_type(bundle_coords, coords_name='bundle_coords')
        morphism.domain.verify_chart_type(bundle_coords.base(), coords_name='bundle_coords.base()')
        #print(f'HIPPO; bundle_coords.fiber() = {bundle_coords.fiber()}')
        return codomain.make_coords_composed(
            bundle_coords.base(),
            codomain.fiber_chart.make_coords(
                np.dot(bundle_coords.fiber().value, J(bundle_coords.base()).fiber().value)
            ),
        )

    return VectorBundleMorphism(
        name=f'T^{{*}}({morphism.name})',
        domain=domain,
        codomain=codomain,
        evaluator=evaluator,
        base_morphism__o=identity_isomorphism(morphism.domain),
    )

#def TangentFunctor_iso (arg:Isomorphism) -> TangentBundleIsomorphism:
    #"""This is the tangent functor's action on isomorphisms of the category (i.e. Isomorphism)."""
    #return TangentBundleIsomorphism.induced(arg)

# TODO: Make a pullback functor which operates on bundles and sections.

# TODO: Make a dual functor.  This would make Cotangent stuff unnecessary.

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
    J_Cyl_to_R3_c = J_Cyl_to_R3(c)
    print(f'{J_Cyl_to_R3}({c}):\n{J_Cyl_to_R3_c.base()},\n{J_Cyl_to_R3_c.fiber()}')
    print()

    J_Cyl_to_R3_c_v = J_Cyl_to_R3(c_v)
    print(f'{J_Cyl_to_R3}({c_v}):\n{J_Cyl_to_R3_c_v.base()},\n{J_Cyl_to_R3_c_v.fiber()}')
    print()

    print(f'type(J_Cyl_to_R3_c_v) = {J_Cyl_to_R3_c_v}')
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
    print()

    product = simplified(np.dot(p_Cyl.fiber().value, J.fiber().value))
    print(f'p_Cyl * J = {product}')
    print()

    QC = Chart(
        name='QC', # Quadratic cylindrical coordinates, where R = r^2 and w = 4*z
        coords_shape=(3,),
        symbolic_coords=np.array([sp.Symbol('R', real=True, positive=True), sp.Symbol('theta', real=True), sp.Symbol('w', real=True)]),
    )
    print(f'QC = {QC}')
    print(f'repr(QC) = {repr(QC)}')
    print()

    def evaluator_R3_to_QC (v:Coords) -> Coords:
        x, y, z = v.value
        R = x**2 + y**2
        theta = sp.atan2(y, x)
        w = 4*z
        return QC.make_coords(simplified(np.array([R, theta, w])))

    def inverse_evaluator_R3_to_QC (c:Coords) -> Coords:
        R, theta, w = c.value
        x = sp.sqrt(R)*sp.cos(theta)
        y = sp.sqrt(R)*sp.sin(theta)
        z = w/4
        return R3.make_coords(simplified(np.array([x, y, z])))

    R3_to_QC = Isomorphism(
        name='R3_to_QC',
        domain=R3,
        codomain=QC,
        # TODO: Make Coords inherit np.ndarray for convenience
        evaluator=evaluator_R3_to_QC,
        inverse_evaluator__o=inverse_evaluator_R3_to_QC,
    )
    print(f'R3_to_QC = {R3_to_QC}')
    print(f'repr(R3_to_QC) = {repr(R3_to_QC)}')
    print()

    QC_to_R3 = R3_to_QC.inverse()

    T_star_QC = CotangentFunctor_ob(QC)
    p_QC = T_star_QC.symbolic_coords
    J = jacobian(R3_to_QC)(QC_to_R3(p_QC.base()))
    print(f'J:\n{J}')
    print(f'change of coords {T_star_QC} -> {T_star_R3}:')
    product = simplified(np.dot(p_QC.fiber().value, J.fiber().value))
    print(f'{product}')
    print()

    J = jacobian(QC_to_R3)(R3_to_QC(p_R3.base()))
    print(f'J:\n{J}')
    print(f'change of coords {T_star_R3} -> {T_star_QC}:')
    product = simplified(np.dot(p_R3.fiber().value, J.fiber().value))
    print(f'{product}')
    print()

    LS = Chart(
        name='LS', # log-size cylindrical coordinates, where s = log(R^2 + w^2)/4 and u = arg(R, w)
        coords_shape=(3,),
        symbolic_coords=np.array([sp.Symbol('s', real=True), sp.Symbol('theta', real=True), sp.Symbol('u', real=True)]),
    )
    print(f'LS = {LS}')
    print(f'repr(LS) = {repr(LS)}')
    print()

    def evaluator_QC_to_LS (v:Coords) -> Coords:
        R, theta, w = v.value
        s = sp.log(R**2 + w**2) / 4
        u = sp.atan2(w, R)
        return LS.make_coords(simplified(np.array([s, theta, u])))

    def inverse_evaluator_QC_to_LS (c:Coords) -> Coords:
        s, theta, u = c.value
        R = sp.exp(2*s)*sp.cos(u)
        w = sp.exp(2*s)*sp.sin(u)
        return QC.make_coords(simplified(np.array([R, theta, w])))

    QC_to_LS = Isomorphism(
        name='QC_to_LS',
        domain=QC,
        codomain=LS,
        # TODO: Make Coords inherit np.ndarray for convenience
        evaluator=evaluator_QC_to_LS,
        inverse_evaluator__o=inverse_evaluator_QC_to_LS,
    )
    print(f'QC_to_LS = {QC_to_LS}')
    print(f'repr(QC_to_LS) = {repr(QC_to_LS)}')
    print()

    ls = LS.symbolic_coords
    LS_to_QC = QC_to_LS.inverse()
    print(f'LS_to_QC = {LS_to_QC}')
    print(f'LS_to_QC({ls.value}) = {LS_to_QC(ls)}')
    print(f'QC_to_LS(LS_to_QC({ls.value})) = {simplified(QC_to_LS(LS_to_QC(ls)).value)}')
    print()

    T_star_LS = CotangentFunctor_ob(LS)
    p_LS = T_star_LS.symbolic_coords
    J = jacobian(QC_to_LS)(LS_to_QC(p_LS.base()))
    print(f'J:\n{J}')
    print(f'change of coords {T_star_LS} -> {T_star_QC}:')
    product = simplified(np.dot(p_LS.fiber().value, J.fiber().value))
    print(f'{product}')
    print()

    J = jacobian(LS_to_QC)(QC_to_LS(p_QC.base()))
    print(f'J:\n{J}')
    print(f'change of coords {T_star_QC} -> {T_star_LS}:')
    product = simplified(np.dot(p_QC.fiber().value, J.fiber().value))
    print(f'{product}')
    print()

    qp_R3           = T_star_R3.symbolic_coords
    q_R3            = qp_R3.base()
    p_R3            = qp_R3.fiber()
    x,   y,   z     = q_R3.value
    p_x, p_y, p_z   = p_R3.value
    P_x_R3          = p_x - y*p_z/2
    P_y_R3          = p_y + x*p_z/2
    H_R3            = (P_x_R3**2 + P_y_R3**2)/2 - 1/(8*sp.pi*sp.sqrt((x**2 + y**2)**2 + 16*z**2))
    J_R3            = x*p_x + y*p_y + 2*z*p_z
    # Legendre transform (momentum to velocity) is the Hessian of H with respect to momenta.
    L_R3            = vorpy.symbolic.D(H_R3, p_R3.value, p_R3.value)

    # TODO: Make ScalarFunction (i.e. Chart -> Real) and PathFunction (i.e. Real -> Chart)

    print(f'H_R3 = {H_R3}')
    print(f'J_R3 = {J_R3}')
    print(f'L_R3 = {L_R3}')
    print()

    J_R3_to_QC = jacobian(R3_to_QC)

    qp_QC = T_star_QC.symbolic_coords
    p_QC = qp_QC.fiber()
    # TODO: This should really be handled by the induced change of coords on cotangent bundle as
    # a vector bundle isomorphism.
    q_R3_from_QC = QC_to_R3(qp_QC.base())
    p_R3_from_QC = T_star_R3.fiber_chart.make_coords(np.dot(qp_QC.fiber().value, J_R3_to_QC(q_R3_from_QC).fiber().value))
    qp_R3_from_QC = T_star_R3.make_coords_composed(q_R3_from_QC, p_R3_from_QC)

    print(f'qp_QC:\n{qp_QC}')
    print(f'p_QC:\n{p_QC}')
    print(f'qp_R3_from_QC:\n{qp_R3_from_QC}')
    print()

    H_QC = substitution(H_R3, qp_R3.value, qp_R3_from_QC.value)
    J_QC = substitution(J_R3, qp_R3.value, qp_R3_from_QC.value)
    # Legendre transform is the Hessian of H with respect to momenta.
    L_QC = vorpy.symbolic.D(H_QC, p_QC.value, p_QC.value)

    R, theta, w = qp_QC.base().value
    p_R, p_theta, p_w = qp_QC.fiber().value
    P_R_QC = 2*sp.sqrt(R)*p_R
    P_theta_QC = p_theta/sp.sqrt(R) + 2*sp.sqrt(R)*p_w

    print(f'P_R_QC = {P_R_QC}')
    print(f'P_theta_QC = {P_theta_QC}')
    print()
    print(f'H_QC = {H_QC}')
    print(f'J_QC = {J_QC}')
    print(f'L_QC = {L_QC}')
    print()

    X_H_QC = simplified(vorpy.symplectic.symplectic_gradient_of(H_QC, qp_QC.value))
    print(f'X_H_QC:\n{X_H_QC}')
    print()

    J_QC_to_LS = jacobian(QC_to_LS)

    qp_LS = T_star_LS.symbolic_coords
    p_LS = qp_LS.fiber()
    q_QC_from_LS = LS_to_QC(qp_LS.base())
    p_QC_from_LS = T_star_QC.fiber_chart.make_coords(np.dot(qp_LS.fiber().value, J_QC_to_LS(q_QC_from_LS).fiber().value))
    qp_QC_from_LS = T_star_QC.make_coords_composed(q_QC_from_LS, p_QC_from_LS)

    print(f'qp_LS:\n{qp_LS}')
    print(f'qp_QC_from_LS:\n{qp_QC_from_LS}')
    print()

    P_R_LS = substitution(P_R_QC, qp_QC.value, qp_QC_from_LS.value)
    P_theta_LS = substitution(P_theta_QC, qp_QC.value, qp_QC_from_LS.value)
    H_LS = substitution(H_QC, qp_QC.value, qp_QC_from_LS.value)
    J_LS = substitution(J_QC, qp_QC.value, qp_QC_from_LS.value)
    # Legendre transform is the Hessian of H with respect to momenta.
    L_LS = vorpy.symbolic.D(H_LS, p_LS.value, p_LS.value)

    s, theta, u = qp_LS.value[0]
    L_LS_formula = sp.exp(-2*s) * np.array([
        [sp.cos(u), sp.sin(u), 0],
        [sp.sin(u), sp.sec(u), 2*sp.cos(u)],
        [0, 2*sp.cos(u), 4*sp.cos(u)],
    ])
    L_LS_formula_error = simplified(L_LS_formula - L_LS)
    assert np.all(L_LS_formula_error == 0), f'L_LS_formula_error was not identically zero; L_LS_formula_error = {L_LS_formula_error}'

    print(f'P_R_LS = {P_R_LS}')
    print(f'P_theta_LS = {P_theta_LS}')
    print()
    print(f'H_LS = {H_LS}')
    print(f'J_LS = {J_LS}')
    print(f'L_LS =\n{L_LS}')
    print()
    print(f'exp(2*s)*cos(u)*L_LS =\n{sp.exp(2*qp_LS.value[0,0])*sp.cos(qp_LS.value[0,2])*L_LS}')
    print()
    print(f'L_LS_formula =\n{L_LS_formula}')
    print()

    K_LS = vorpy.tensor.contract('i,ij,j', p_LS.value, L_LS_formula, p_LS.value, dtype=object)/2
    U_LS = -sp.exp(-2*s)/(8*sp.pi)

    H_LS_formula = K_LS + U_LS
    H_LS_formula_error = simplified(H_LS - H_LS_formula)
    assert H_LS_formula_error == 0, f'H_LS_formula_error was not identically zero; H_LS_formula_error = {H_LS_formula_error}'

    print(f'type(H_LS) = {type(H_LS)}')
    print(f'type(H_LS) = {type(H_LS)}')
    X_H_LS = simplified(vorpy.symplectic.symplectic_gradient_of(H_LS, qp_LS.value))
    print(f'X_H_LS:\n{X_H_LS}')
    print()

    # Solve for p_u in terms of the other variables in H = 0 and substitute that into X_H
    p_u = qp_LS.value[1,2]
    p_u_constrained_by_H_v = sp.solve(H_LS, p_u)
    assert len(p_u_constrained_by_H_v) == 2
    p_u_constrained_by_H_v = np.array(p_u_constrained_by_H_v)
    print(f'p_u_constrained_by_H_v = {p_u_constrained_by_H_v}')
    print()

    dpu_dt = X_H_LS[1,2]
    print(f'd/dt(p_u) = {dpu_dt}')
    print()

    dpu_dt_subs_v = np.array([substitution(dpu_dt, np.array(p_u), np.array(p_u_constrained_by_H)) for p_u_constrained_by_H in p_u_constrained_by_H_v])
    print(f'dpu_dt_subs_v = {dpu_dt_subs_v}')
    print()

    # Induced change of cotangent bundle coordinates.
    T_star_R3_to_QC = CotangentFunctor_mor(R3_to_QC)
    require.is_equal(T_star_R3_to_QC.domain, PullbackVectorBundleChart(pullback_morphism=R3_to_QC, target_bundle_chart=T_star_QC))
    print(f'T_star_R3_to_QC = {T_star_R3_to_QC}')
    print(f'T_star_R3_to_QC.domain = {T_star_R3_to_QC.domain}')
    print(f'QC_to_R3(qp_QC.base()), qp_QC = {QC_to_R3(qp_QC.base()), qp_QC}')
    print(f'T_star_R3_to_QC.domain.make_coords_composed(QC_to_R3(qp_QC.base()), qp_QC.fiber()) = {T_star_R3_to_QC.domain.make_coords_composed(QC_to_R3(qp_QC.base()), qp_QC.fiber())}')
    print(T_star_R3_to_QC(T_star_R3_to_QC.domain.make_coords_composed(QC_to_R3(qp_QC.base()), qp_QC.fiber())))
    print()

    if True:
        X_H_LS__fast = vorpy.symbolic.lambdified(
            X_H_LS,
            qp_LS.value,
            replacement_d={
                'array':'np.array',
                'cos':'np.cos',
                'sin':'np.sin',
                'exp':'np.exp',
                'pi':'np.pi',
                'dtype=object':'dtype=float',
            },
            verbose=True,
        )
