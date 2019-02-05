import abc
import itertools
import numpy as np
import sympy as sp
import textwrap
import typing
import vorpy.symbolic

# Convenience function because sympy's simplify function doesn't preserve the input type
def simplified (arg:np.ndarray) -> np.ndarray:
    return np.array(sp.simplify(arg)).reshape(arg.shape)

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
        if coords.chart is not self:
            raise TypeError(f'{coords_name}.chart (which was {coords.chart}) was expected to be {self}')

    def make_coords (self, value:np.ndarray) -> Coords:
        """This constructs a correctly typed Coords from a raw np.ndarray.  The shape must match self.coords_shape."""
        if value.shape != self.coords_shape:
            raise TypeError(f'Expected coords.shape (which was {coords.shape}) to be self.coords_shape (which was {self.coords_shape})')
        return self.coords_class(value, chart=self)

    def make_coords_uninitialized (self, *, dtype:typing.Any) -> Coords:
        return self.coords_class(np.ndarray(self.coords_shape, dtype=dtype), chart=self)

    def __repr__ (self) -> str:
        return f'Chart({self.name}, coords_shape={self.coords_shape})'

    def __str__ (self) -> str:
        return self.name

class BundleChart(Chart):
    """Represents a trivialized chart (i.e. direct product of base and fiber charts) for a [fiber] bundle."""

    def __init__ (self, *, name:str, base_chart:Chart, fiber_chart:Chart, coords_class=typing.Any) -> None:
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

class VectorBundleChart(BundleChart):
    """Represents a trivialized chart (i.e. direct product of base and fiber charts) for a vector bundle."""

    def __init__ (self, *, name:str, base_chart:Chart, fiber_chart:Chart, coords_class:typing.Any=VectorBundleCoords) -> None:
        if not issubclass(coords_class, VectorBundleCoords):
            raise TypeError(f'Expected coords_class (which was {coords_class}) to be a subclass of {VectorBundleCoords}')

        BundleChart.__init__(self, name=name, base_chart=base_chart, fiber_chart=fiber_chart, coords_class=coords_class)

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

class TangentBundleChart(VectorBundleChart):
    """Represents a trivialized chart (i.e. direct product of base and fiber charts) for a tangent bundle."""

    # TODO: Could potentially induce fiber_chart from base_chart (since it's just the vector space version of the base chart)
    def __init__ (self, *, base_chart:Chart, fiber_chart:Chart) -> None:
        if base_chart.coords_shape != fiber_chart.coords_shape:
            raise TypeError(f'Expected base_chart.coords_shape (which was {base_chart.coords_shape}) to be equal to fiber_chart.coords_shape (which was {fiber_chart.coords_shape})')

        # TODO: Maybe make some bool attribute indicating if this TangentBundleChart was induced by the base chart.
        if fiber_chart.name == '%Induced%':
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
            name='%Induced%', # Sentinel name used in the name of the Chart
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
        if fiber_chart.name == '%Induced%':
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
            name='%Induced%', # Sentinel name used in the name of the Chart
            coords_shape=base_chart.coords_shape,
            symbolic_coords=fiber_symbolic_coords,
            coords_class=base_chart.coords_class,
        )
        return CotangentBundleChart(base_chart=base_chart, fiber_chart=fiber_chart)

class TensorBundleChart(VectorBundleChart):
    def __init__ (self, *vector_bundle_chart__v:VectorBundleChart, fiber_symbolic_coords:np.ndarray) -> None:
        print(f'TensorBundleChart(vector_bundle_chart__v = {vector_bundle_chart__v})')

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

    def factor (self, index:int) -> VectorBundleChart:
        return self.vector_bundle_chart__v[index]

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
        retval = BundleChart.make_coords_composed(self, base, fiber)
        self.verify_chart_type(retval, coords_name='retval')
        assert isinstance(retval, PullbackBundleCoords) # sanity check
        return typing.cast(PullbackBundleCoords, retval)

    def target_projection (self, coords:PullbackBundleCoords) -> BundleCoords:
        return self.target_bundle_chart.make_coords_composed(self.pullback_morphism(coords.base()), coords.fiber())

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

            def inverse_evaluator_ (coords:Coords) -> Coords:
                return codomain.make_coords(np.array(sp.Subs(solution_v[0], y.value, coords.value).doit()).reshape(x.value.shape))

            self.inverse_evaluator = inverse_evaluator_

    def inverse (self) -> 'Isomorphism':
        return Isomorphism(
            name=f'{self.name}^{{-1}}',
            domain=self.codomain,
            codomain=self.domain,
            evaluator=self.inverse_evaluator,
            inverse_evaluator__o=self.evaluator,
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
    print(f'jacobian; linear_morphism_bundle_chart = {linear_morphism_bundle_chart}')

    # Compute the Jacobian and define the fiber_evaluator for it.

    x = morphism.domain.symbolic_coords
    y = morphism(x)
    J_x = vorpy.symbolic.differential(y.value, x.value)
    assert J_x.shape == y.value.shape + x.value.shape
    assert J_x.shape == linear_morphism_bundle_chart.fiber_chart.coords_shape

    def jacobian_fiber_evaluator (base:Coords) -> Coords:
        return linear_morphism_bundle_chart.fiber_chart.make_coords(simplified(np.array(sp.Subs(J_x, x.value, base.value).doit()).reshape(linear_morphism_bundle_chart.fiber_chart.coords_shape)))

    retval = TensorBundleSection(
        name=f'Jacobian({morphism})',
        bundle_chart=linear_morphism_bundle_chart,
        fiber_evaluator=jacobian_fiber_evaluator,
    )
    assert isinstance(retval, Morphism)
    assert retval.domain == morphism.domain
    return retval

#class VectorBundleMorphism(Morphism):
    #def __init__ (
        #self,
        #*,
        #domain:VectorBundleChart,
        #codomain:VectorBundleChart,
        #evaluator:typing.Callable[[VectorBundleCoords],VectorBundleCoords],
    #) -> None:
        #Morphism.__init__(
            #domain=domain,
            #codomain=codomain,
            #evaluator=evaluator,
        #)

        ## Sanity check -- this is for FiberBundleMorphism
        ## TODO: Verify that there exists a base morphism which makes the following diagram commute:
        ## Let F denote the vector bundle morphism and let f denote the base morphism.
        ##
        ##   domain    -- F -->    codomain
        ##      |                     |
        ##   baseproj              baseproj
        ##      V                     V
        ## domain.base -- f --> codomain.base

        #self.base_morphism = Morphism(
            #domain=domain.base_chart,
            #codomain=codomain.base_chart,
            #evaluator=lambda v:codomain.base_projection(self.evaluator(v)), # This relies on the commutativity of the diagram
        #)
        ## It's not necessarily well-defined to map domain.fiber_chart to codomain.fiber_chart.
        #self.fiber_morphism = Morphism(
            #domain=domain,
            #codomain=codomain.fiber_chart,
            #evaluator=lambda v:codomain.fiber_projection(self.evaluator(v)),
        #)

    ## TODO: static method "make_composed" which takes a base morphism and basepoint-independent fiber morphism.

# TODO: VectorBundleIsomorphism, TangentBundleMorphism, TangentBundleIsomorphism, etc.

def TangentFunctor_ob (base_chart:Chart, *, fiber_symbolic_coords__o:typing.Optional[np.ndarray]=None) -> TangentBundleChart:
    """This is the tangent functor's action on objects of the category (i.e. Chart)."""
    return TangentBundleChart.induced(base_chart, fiber_symbolic_coords__o=fiber_symbolic_coords__o)

def CotangentFunctor_ob (base_chart:Chart, *, fiber_symbolic_coords__o:typing.Optional[np.ndarray]=None) -> CotangentBundleChart:
    """This is the cotangent functor's action on objects of the category (i.e. Chart)."""
    return CotangentBundleChart.induced(base_chart, fiber_symbolic_coords__o=fiber_symbolic_coords__o)

#def TangentFunctor_mor (arg:Morphism) -> TangentBundleMorphism:
    #"""This is the tangent functor's action on morphisms of the category (i.e. Morphism)."""
    #return TangentBundleMorphism.induced(arg)

#def TangentFunctor_iso (arg:Isomorphism) -> TangentBundleIsomorphism:
    #"""This is the tangent functor's action on isomorphisms of the category (i.e. Isomorphism)."""
    #return TangentBundleIsomorphism.induced(arg)

# TODO: Make a pullback functor which operates on bundles and sections.

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
