import abc
import itertools
import numpy as np
import sympy as sp
import typing
import vorpy.symbolic

# Convenience function because sympy's simplify function doesn't preserve the input type
def simplified (arg:np.ndarray) -> np.ndarray:
    return np.array(sp.simplify(arg)).reshape(arg.shape)

#TODO: Just use composition for Coordinates and CoordinatesTangentBundle

class Coordinates(np.ndarray):
    # The constructor is the awkward numpy.ndarray one, which looks like
    #
    #     numpy.ndarray(shape, dtype=dtype, buffer=actual_value)
    #
    # instead of the convenient numpy.array function.  But you shouldn't be using the
    # Coordinates class directly.

    @classmethod
    @abc.abstractmethod
    def coordinate_system (cls) -> 'CoordinateSystem':
        # Could implement a default which returns R^{s_1} \otimes ... \otimes R^{s_n}
        # where (s_1, ..., s_n) is the shape.
        pass

# TODO: Create CoordinatesTangentBundle, etc. where you can get basepoint and fiber component.

#class CoordinatesTangentBundle(Coordinates):
    #@classmethod
    #@abc.abstractmethod
    #def base_coordinate_system (cls) -> 'CoordinateSystem':
        #pass

    #@classmethod
    #@abc.abstractmethod
    #def fiber_coordinate_system (cls) -> 'CoordinateSystem':
        #pass


class CoordinateSystem:
    def __init__ (
        self,
        name:str,
        *,
        shape:typing.Tuple[int,...],
        assumption__d:typing.Dict[str,typing.Any]={},
        symbol_assumption__d:typing.Dict[str,typing.Dict[str,typing.Any]]={},
        default_name__to:typing.Optional[np.ndarray]=None,
    ) -> None:
        if default_name__to is not None:
            if default_name__to.shape != shape:
                raise TypeError(f'Expected default_name__to to be None or have shape (which was {default_name__to.shape}) equal to shape (which was {shape}')

        class SpecificCoordinates(Coordinates):
            @classmethod
            def coordinate_system (cls) -> 'CoordinateSystem':
                return self

            def __str__ (self) -> str:
                return f'{self.coordinate_system().name}{np.ndarray.__str__(self)}'

            def __repr__ (self) -> str:
                return f"Coordinates('{self.coordinate_system().name}', {np.ndarray.__str__(self)})"

        self.name                   = name
        self.shape                  = shape
        self.assumption__d          = assumption__d
        self.symbol_assumption__d   = symbol_assumption__d
        self.default_name__to       = default_name__to
        self.coordinates_type       = SpecificCoordinates

    def dimension (self) -> int:
        return vorpy.tensor.dimension_of_shape(self.shape)

    def create_coordinates (
        self,
        *,
        extra_base_assumption__d:typing.Dict[str,typing.Any]={},
        extra_symbol_assumption__d:typing.Dict[str,typing.Dict[str,typing.Any]]={},
    ) -> Coordinates:
        """Same as create_coordinates_named, but uses the default names (if present, otherwise raise ValueError)."""
        if self.default_name__to is None:
            raise ValueError(f'Can\'t call create_coordinates when self.default_name__to is None')
        return self.create_coordinates_named(
            self.default_name__to,
            extra_base_assumption__d=extra_base_assumption__d,
            extra_symbol_assumption__d=extra_symbol_assumption__d,
        )

    def create_coordinates_named (
        self,
        name__t:np.ndarray,
        *,
        extra_base_assumption__d:typing.Dict[str,typing.Any]={},
        extra_symbol_assumption__d:typing.Dict[str,typing.Dict[str,typing.Any]]={},
    ) -> Coordinates:
        """
        name__t should be a numpy.ndarray with shape matching this CoordinateSystem, whose elements
        are strings.  extra_base_assumptions__d will be added to the assumptions for all symbols.
        extra_symbol_assumption__d should map symbol names (occuring in name__t) to a dict
        of additional assumptions to make for that specific symbol.

        Note that if there are any common keys between any of the assumption dictionaries, the
        following precedence will be used to disambiguate.  From lowest to highest priority:

            self.assumption__d         -- least specific; default for all
            self.symbol_assumption__d  -- default for specific symbols (using default_name__to if present)
            extra_base_assumption__d        -- creation-specific
            extra_symbol_assumption__d      -- per-symbol creation-specific
        """
        if name__t.shape != self.shape:
            raise TypeError(f'name__t.shape (which was {name__t.shape}) was expected to be equal to self.shape (which was {self.shape}')

        if self.default_name__to is None:
            default_name__t = name__t
        else:
            default_name__t = self.default_name__to

        retval = self.coordinates_type(self.shape, dtype=object)

        retval_reshaped = retval.reshape(-1)
        for i,(default_name,name) in enumerate(zip(default_name__t.flat, name__t.flat)):
            if not isinstance(name, str):
                raise TypeError(f'expected name__t element (the {i}th element in the flattened tensor, which was {name}) to be a string')

            # Construct the assumption dict based on the precedence described above.
            assumption__d = {}
            for k,v in itertools.chain(
                self.assumption__d.items(),
                self.symbol_assumption__d.get(default_name, {}).items(), # Default empty dict if symbol is not present
                extra_base_assumption__d.items(),
                extra_symbol_assumption__d.get(name, {}).items(), # Default empty dict if symbol is not present
            ):
                assumption__d[k] = v

            retval_reshaped[i] = sp.Symbol(name, **assumption__d)

        return retval

    def __call__ (self, coordinates__o:typing.Optional[np.ndarray]=None, *, dtype:typing.Any=object) -> Coordinates:
        """Use this function to construct a Coordinates expression belonging to this CoordinateSystem."""
        if coordinates__o is not None and coordinates__o.shape != self.shape:
            raise TypeError(f'Expected coordinates__o to be None or to have shape {self.shape} (but it had shape {coordinates__o.shape}) ')

        if coordinates__o is not None:
            dtype = coordinates__o.dtype
        else:
            dtype = dtype

        return self.coordinates_type(self.shape, dtype=dtype, buffer=coordinates__o)

    def __str__ (self) -> str:
        return self.name

    def __repr__ (self) -> str:
        return f"CoordinateSystem({self.name}, shape={self.shape})"

    def __eq__ (self, other:'CoordinateSystem') -> bool:
        #print(f'CoordinateSystem.__eq__; self = {self}, other = {other}')

        if self is other:
            return True

        return self.name == other.name and self.shape == other.shape and self.assumption__d == other.assumption__d and self.symbol_assumption__d == other.symbol_assumption__d and np.all(self.default_name__to == other.default_name__to)

class Morphism:
    """Morphism in the category of coordinatized manifolds."""

    def __init__ (
        self,
        *,
        domain:CoordinateSystem,
        codomain:CoordinateSystem,
        evaluator:typing.Callable[[Coordinates],Coordinates],
        compute_jacobian:bool=True,
    ) -> None:
        self.domain = domain
        self.codomain = codomain
        self.evaluator = evaluator

        if compute_jacobian:
            # Compute the Jacobian (Dphi) of the morphism (phi).
            x = domain.create_coordinates()
            phi = evaluator(x)
            Dphi = vorpy.symbolic.differential(phi, x)
            assert Dphi.shape == phi.shape + x.shape

            # TODO: Make the return type more specific (the tensor product type corresponding to the matrix field)
            def jacobian_evaluator (base_point:Coordinates) -> np.ndarray:
                if base_point.coordinate_system() is not domain:
                    raise TypeError(f'Expected base_point.coordinate_system() (which was {base_point.coordinate_system()}) to be {domain}')
                # TODO: Could probably use vorpy.symbolic.lambdified
                return np.array(sp.Subs(Dphi, x, base_point).doit()).reshape(Dphi.shape)

            self.jacobian_evaluator = jacobian_evaluator

            # TODO: Could also represent Jacobian as a Morphism of tangent bundles.

    def __call__ (self, coordinates:Coordinates) -> Coordinates:
        if coordinates.coordinate_system() != self.domain:
            raise TypeError(f'coordinates belong to {coordinates.coordinate_system()} but were expected to belong to {self.domain}')
        retval = self.evaluator(coordinates)
        if retval.coordinate_system() != self.codomain:
            raise TypeError(f'changed coordinates belong to {retval.coordinate_system()} but were expected to belong to {self.codomain}')
        return retval

    def __repr__ (self) -> str:
        return f'Morphism({self.domain} -> {self.codomain})'

class Isomorphism(Morphism):
    """Isomorphism in the category of coordinatized manifolds."""

    def __init__ (
        self,
        *,
        domain:CoordinateSystem,
        codomain:CoordinateSystem,
        evaluator:typing.Callable[[Coordinates],Coordinates],
        inverse_evaluator__o:typing.Optional[typing.Callable[[Coordinates],Coordinates]]=None,
        compute_jacobian:bool=True,
    ) -> None:
        """
        If inverse_evaluator__o is not specified, then sympy.solve will be used to attempt
        to symbolically compute the inverse.  If this doesn't find a unique solution, then
        it will raise an exception, meaning that the inverse evaluator must be specified
        explicitly.
        """

        if domain.dimension() != codomain.dimension():
            raise TypeError(f'domain and codomain must have the same dimension for Isomorphism.')

        Morphism.__init__(self, domain=domain, codomain=codomain, evaluator=evaluator, compute_jacobian=compute_jacobian)

        if inverse_evaluator__o is not None:
            self.inverse_evaluator = inverse_evaluator__o
        else:
            # Attempt to solve symbolically.
            dom = self.domain.create_coordinates()
            cod = self.codomain.create_coordinates()
            solution_v = sp.solve((cod - evaluator(dom)).reshape(-1), *dom.reshape(-1).tolist())
            print(f'solution_v:')
            for solution in solution_v:
                print(f'    {solution}')
            if len(solution_v) != 1:
                raise ValueError(f'sympy.solve did not automatically find a unique solution to the inverse coordinate change; inverse_evaluator__o should be specified to define the inverse explicitly..  solution_v = {solution_v}')

            def inverse_evaluator_ (coordinates:Coordinates) -> Coordinates:
                return codomain(np.array(sp.Subs(solution_v[0], cod, coordinates).doit()).reshape(dom.shape))

            self.inverse_evaluator = inverse_evaluator_

        if compute_jacobian:
            # Compute the inverse Jacobian (Dpsi) of the isomorphism inverse (psi)
            y = codomain.create_coordinates()
            psi = self.inverse_evaluator(y)
            Dpsi = vorpy.symbolic.differential(psi, y)
            assert Dpsi.shape == psi.shape + y.shape

            def inverse_jacobian_evaluator (base_point:Coordinates) -> np.ndarray: # TODO: more-specific typing.
                if base_point.coordinate_system() is not codomain:
                    raise TypeError(f'Expected base_point.coordinate_system() (which was {base_point.coordinate_system()}) to be {codomain}')
                # TODO: Could probably use vorpy.symbolic.lambdified
                return np.array(sp.Subs(Dpsi, y, base_point).doit()).reshape(Dpsi.shape)

            self.inverse_jacobian_evaluator = inverse_jacobian_evaluator

    #def __call__ (self, coordinates:Coordinates) -> Coordinates:
        #if coordinates.coordinate_system() != self.domain:
            #raise TypeError(f'coordinates belong to {coordinates.coordinate_system()} but were expected to belong to {self.domain}')
        #retval = self.evaluator(coordinates)
        #if retval.coordinate_system() != self.codomain:
            #raise TypeError(f'changed coordinates belong to {retval.coordinate_system()} but were expected to belong to {self.codomain}')
        #return retval

    def inverse (self) -> 'Isomorphism':
        # This is somewhat inefficient because jacobian_evaluator and inverse_jacobian_evaluator are already computed and
        # could just be switched here without being recomputed.
        return Isomorphism(
            domain=self.codomain,
            codomain=self.domain,
            evaluator=self.inverse_evaluator,
            inverse_evaluator__o=self.evaluator,
            compute_jacobian=hasattr(self, 'jacobian_evaluator'),
        )

    def __repr__ (self) -> str:
        return f'Isomorphism({self.domain} -> {self.codomain})'

# TODO: Could make a function which returns a type-specific version of this
class CoordinateChange(Isomorphism):
    def __init__ (
        self,
        *,
        domain:CoordinateSystem,
        codomain:CoordinateSystem,
        evaluator:typing.Callable[[Coordinates],Coordinates],
        inverse_evaluator__o:typing.Optional[typing.Callable[[Coordinates],Coordinates]]=None,
        compute_jacobian:bool=True,
    ) -> None:
        Isomorphism.__init__(
            self,
            domain=domain,
            codomain=codomain,
            evaluator=evaluator,
            inverse_evaluator__o=inverse_evaluator__o,
            compute_jacobian=compute_jacobian,
        )

    #def __call__ (self, coordinates:Coordinates) -> Coordinates:
        #if coordinates.coordinate_system() != self.domain:
            #raise TypeError(f'coordinates belong to {coordinates.coordinate_system()} but were expected to belong to {self.domain}')
        #retval = self.evaluator(coordinates)
        #if retval.coordinate_system() != self.codomain:
            #raise TypeError(f'changed coordinates belong to {retval.coordinate_system()} but were expected to belong to {self.codomain}')
        #return retval

    def inverse (self) -> 'CoordinateChange':
        # This is somewhat inefficient because jacobian_evaluator and inverse_jacobian_evaluator are already computed and
        # could just be switched here without being recomputed.
        return CoordinateChange(
            domain=self.codomain,
            codomain=self.domain,
            evaluator=self.inverse_evaluator,
            inverse_evaluator__o=self.evaluator,
            compute_jacobian=hasattr(self, 'jacobian_evaluator'),
        )

    def __repr__ (self) -> str:
        return f'CoordinateChange({self.domain} -> {self.codomain})'

#def CoordinateSystemVectorBundle(CoordinateSystem):
    #def __init__ (self, name:str, *, base_coordinate_system:CoordinateSystem, fiber_coordinate_system:CoordinateSystem, **kwargs) -> None:
        #CoordinateSystem.__init__(
            #self,
            #name,
            #shape= TODO -- isnt just a simple (2,)+base_coordinate_system.shape, since the fiber could have any dimension.
            #**kwargs,
        #)
        #self.base_coordinate_system     = base_coordinate_system
        #self.fiber_coordinate_system    = fiber_coordinate_system

class CoordinateSystemTangentBundle(CoordinateSystem):
    def __init__ (
        self,
        *,
        base_coordinate_system:CoordinateSystem,
        fiber_coordinate_system:CoordinateSystem,
        assumption__d:typing.Dict[str,typing.Any]={},
        symbol_assumption__d:typing.Dict[str,typing.Dict[str,typing.Any]]={},
        default_name__to:typing.Optional[np.ndarray]=None,
    ) -> None:
        if base_coordinate_system.shape != fiber_coordinate_system.shape:
            raise TypeError(f'Expected the shapes of base_coordinate_system (which was {base_coordinate_system}) and fiber_coordinate_system (which was {fiber_coordinate_system}) to be equal')

        # Omit the fiber from the name if it's the induced fiber.
        if fiber_coordinate_system.name == 'Induced':
            name = f'T({base_coordinate_system})'
        else:
            name = f'T({base_coordinate_system}, fiber={fiber_coordinate_system})'

        shape = (2,) + base_coordinate_system.shape

        if default_name__to is None and base_coordinate_system.default_name__to is not None and fiber_coordinate_system.default_name__to is not None:
            default_name__to = np.array(base_coordinate_system.default_name__to.reshape(-1).tolist() + fiber_coordinate_system.default_name__to.reshape(-1).tolist()).reshape(shape)

        CoordinateSystem.__init__(
            self,
            name,
            shape=shape,
            assumption__d=assumption__d,
            symbol_assumption__d=symbol_assumption__d,
            default_name__to=default_name__to,
        )

        self.base_coordinate_system     = base_coordinate_system
        self.fiber_coordinate_system    = fiber_coordinate_system

    # TODO: Should probably return a Morphism instead
    def base_point_projection (self, coordinates_tangent_bundle:Coordinates) -> Coordinates:
        if coordinates_tangent_bundle.coordinate_system() != self:
            raise TypeError(f'coordinates_tangent_bundle.coordinate_system() (which was {coordinates_tangent_bundle.coordinate_system()}) was expected to be {self}')
        return self.base_coordinate_system(coordinates_tangent_bundle[0,...])

    # TODO: Should probably return a Morphism instead
    def fiber_projection (self, coordinates_tangent_bundle:Coordinates) -> Coordinates:
        if coordinates_tangent_bundle.coordinate_system() != self:
            raise TypeError(f'coordinates_tangent_bundle.coordinate_system() (which was {coordinates_tangent_bundle.coordinate_system()}) was expected to be {self}')
        return self.fiber_coordinate_system(coordinates_tangent_bundle[1,...])

    # TODO: Should probably return a Morphism instead
    def composed_tangent_vector (self, base_point:Coordinates, fiber_component:Coordinates) -> Coordinates:
        if base_point.dtype != fiber_component.dtype:
            raise TypeError(f'base_point.dtype (which was {base_point.dtype}) and fiber_component.dtype (which was {fiber_component.dtype}) were expected to be equal')
        retval = self(dtype=base_point.dtype)
        retval[0,...] = base_point
        retval[1,...] = fiber_component
        return retval

    @staticmethod
    def induced (base_coordinate_system:CoordinateSystem, *, default_name__to:typing.Optional[np.ndarray]=None) -> 'CoordinateSystemTangentBundle':
        if base_coordinate_system.default_name__to is None:
            fiber_default_name__to = None
        else:
            fiber_default_name__to = np.array([
                'v_'+default_name
                for default_name in base_coordinate_system.default_name__to.reshape(-1)
            ]).reshape(base_coordinate_system.shape)

        fiber_coordinate_system = CoordinateSystem(
            'Induced', # Sentinel name used in __init__
            shape=base_coordinate_system.shape,
            assumption__d=base_coordinate_system.assumption__d,
            symbol_assumption__d={},
            default_name__to=fiber_default_name__to,
        )

        return CoordinateSystemTangentBundle(
            base_coordinate_system=base_coordinate_system,
            fiber_coordinate_system=fiber_coordinate_system,
            assumption__d=base_coordinate_system.assumption__d,
            symbol_assumption__d=base_coordinate_system.symbol_assumption__d,
            # TODO: Fiber symbol assumptions
            default_name__to=default_name__to,
        )

class CoordinateChangeTangentBundle(CoordinateChange):
    def __init__ (
        self,
        *,
        domain:CoordinateSystemTangentBundle,
        codomain:CoordinateSystemTangentBundle,
        evaluator:typing.Callable[[Coordinates],Coordinates],
        inverse_evaluator__o:typing.Optional[typing.Callable[[Coordinates],Coordinates]]=None,
        fiber_matrix_evaluator:typing.Callable[[Coordinates],np.ndarray], # TODO: more-specific type
        inverse_fiber_matrix_evaluator__o:typing.Optional[typing.Callable[[Coordinates],np.ndarray]], # TODO: more-specific type
        compute_jacobian:bool=True,
        compute_base_coordinate_change_jacobian:bool=True,
    ) -> None:
        """
        This assumes that the coordinate change decomposes into a base coordinate change
        and a fiber coordinate change (which may depend on the basepoint and might not be
        the Jacobian of the base coordinate change).
        """

        CoordinateChange.__init__(
            self,
            domain=domain,
            codomain=codomain,
            evaluator=evaluator,
            inverse_evaluator__o=inverse_evaluator__o,
            compute_jacobian=compute_jacobian,
        )

        x = domain.create_coordinates()
        p = domain.base_point_projection(x)

        y = evaluator(x)
        q = codomain.base_point_projection(y)
        # TODO: verify that q contains only symbols from p.  (though this might be tricky because there could
        # be other symbols that are considered constant).

        def base_coordinate_change_evaluator (coordinates:Coordinates) -> Coordinates:
            return np.array(sp.Subs(q, p, coordinates).doit()).reshape(q.shape)

        if inverse_evaluator__o is None:
            base_coordinate_change_inverse_evaluator__o = None
        if inverse_evaluator__o is not None:
            u = codomain.create_coordinates()
            a = codomain.base_point_projection(u)

            v = inverse_evaluator__o(u)
            b = domain.base_point_projection(v)
            # TODO: verify that b contains only symbols from a.  (though this might be tricky because there could
            # be other symbols that are considered constant).

            def base_coordinate_change_inverse_evaluator (coordinates:Coordinates) -> Coordinates:
                return np.array(sp.Subs(b, a, coordinates).doit()).reshape(b.shape)

            base_coordinate_change_inverse_evaluator__o = base_coordinate_change_inverse_evaluator

        # Compute the base coordinate change -- TODO: Make a more-specific type (e.g. if the base coordinate change
        # is actually CoordinateChangeTangentBundle or something specific).
        self.base_coordinate_change = CoordinateChange(
            domain=domain.base_coordinate_system,
            codomain=codomain.base_coordinate_system,
            evaluator=base_coordinate_change_evaluator,
            inverse_evaluator__o=base_coordinate_change_inverse_evaluator__o,
            compute_jacobian=compute_base_coordinate_change_jacobian,
        )

        self.fiber_matrix_evaluator = fiber_matrix_evaluator

        # If not specified, compute the inverse fiber matrix evaluator.
        if inverse_fiber_matrix_evaluator__o is not None:
            self.inverse_fiber_matrix_evaluator__o = inverse_fiber_matrix_evaluator__o
        else:
            # Want to compute the function y |-> Jac(f^{-1}(y)), where f is the
            # base coordinate change function.

            y = codomain.base_coordinate_system.create_coordinates()
            x = self.base_coordinate_change.inverse_evaluator(y)
            FM = fiber_matrix_evaluator(x)
            inverse_FM = sp.Matrix(FM).inv()
            inverse_FM.simplify()
            inverse_FM = np.array(inverse_FM)

            def inverse_fiber_matrix_evaluator (base_point:Coordinates) -> np.ndarray:
                return np.array(sp.Subs(inverse_FM, y, base_point).doit()).reshape(inverse_FM.shape)

            self.inverse_fiber_matrix_evaluator__o = inverse_fiber_matrix_evaluator

    @staticmethod
    def induced (base_coordinate_change:CoordinateChange, *, compute_jacobian:bool=True) -> 'CoordinateChangeTangentBundle':
        domain   = TangentFunctor_ob(base_coordinate_change.domain)
        codomain = TangentFunctor_ob(base_coordinate_change.codomain)

        def evaluator (tangent_bundle_coords:Coordinates) -> Coordinates:
            # TODO: Create CoordinatesTangentBundle with basepoint and fiber component projections with correct typing.
            base_point      = base_coordinate_change.domain(tangent_bundle_coords[0,...])
            fiber_component = tangent_bundle_coords[1,...]

            retval          = codomain(dtype=object)
            # Set the basepoint
            retval[0,...]   = base_coordinate_change.evaluator(base_point)
            # Set the fiber component
            retval[1,...]   = np.dot(base_coordinate_change.jacobian_evaluator(base_point), fiber_component)

            return retval

        def inverse_evaluator (tangent_bundle_coords:Coordinates) -> Coordinates:
            base_point      = base_coordinate_change.codomain(tangent_bundle_coords[0,...])
            fiber_component = tangent_bundle_coords[1,...]

            retval          = domain(dtype=object)
            # Set the basepoint
            retval[0,...]   = base_coordinate_change.inverse_evaluator(base_point)
            # Set the fiber component
            retval[1,...]   = np.dot(base_coordinate_change.inverse_jacobian_evaluator(base_point), fiber_component)

            return retval

        return CoordinateChangeTangentBundle(
            domain=domain,
            codomain=codomain,
            evaluator=evaluator,
            inverse_evaluator__o=inverse_evaluator,
            fiber_matrix_evaluator=base_coordinate_change.jacobian_evaluator,
            inverse_fiber_matrix_evaluator__o=base_coordinate_change.inverse_jacobian_evaluator,
            compute_jacobian=compute_jacobian,
            compute_base_coordinate_change_jacobian=True, # NOTE: This is sort of redundant, because we already have the base coordinate change.
        )

def TangentFunctor_ob (arg:CoordinateSystem) -> CoordinateSystemTangentBundle:
    """This is the tangent functor's action on objects of the category."""
    return CoordinateSystemTangentBundle.induced(arg)

# TODO: This should actually operate on Morphism and produce MorphismOfTangentBundle,
# analogously, Isomorphism -> IsomorphismOfTangentBundle
#              CoordinateChange -> CoordinateChangeTangentBundle
def TangentFunctor_mor (arg:CoordinateChange) -> CoordinateChangeTangentBundle: # TODO: MorphismOfTangentBundles
    """This is the tangent functor's action on morphisms of the category."""
    return CoordinateChangeTangentBundle.induced(arg)

if __name__ == '__main__':
    R3 = CoordinateSystem('R3', shape=(3,), default_name__to=np.array(['x','y','z']), assumption__d=dict(real=True))
    v = R3.create_coordinates()
    print(f'v = {v}')
    print(f'repr(v) = {repr(v)}')
    print(f'v.coordinate_system() = {v.coordinate_system()}')
    print(f'repr(v.coordinate_system()) = {repr(v.coordinate_system())}')

    w = R3.create_coordinates_named(np.array(['x','y','z']))
    print(f'w = {w}')

    G = R3.create_coordinates(
        extra_base_assumption__d=dict(
            real=True,
        ),
        extra_symbol_assumption__d=dict(
            x=dict(positive=True),
            y=dict(integer=True),
            z=dict(negative=True),
        ),
    )
    print(f'G = {G}')

    Cyl = CoordinateSystem(
        'Cyl',
        shape=(3,),
        default_name__to=np.array(['r','theta','z']),
        assumption__d=dict(real=True),
        symbol_assumption__d=dict(
            r=dict(
                positive=True,
            ),
        ),
    )
    c = Cyl.create_coordinates()
    print(f'c = {c}')
    assert c[0].assumptions0['real']
    assert c[0].assumptions0['positive']

    R3ToCyl = CoordinateChange(
        domain=R3,
        codomain=Cyl,
        evaluator=lambda euc:Cyl(np.array([sp.sqrt(euc[0]**2 + euc[1]**2), sp.atan2(euc[1], euc[0]), euc[2]])),
        inverse_evaluator__o=lambda cyl:R3(np.array([cyl[0]*sp.cos(cyl[1]), cyl[0]*sp.sin(cyl[1]), cyl[2]])),
    )
    print(f'R3ToCyl = {R3ToCyl}')
    print(f'R3ToCyl(v) = {R3ToCyl(v)}')
    print()

    CylToR3 = R3ToCyl.inverse()
    print(f'CylToR3 = {CylToR3}')
    print()

    print(f'CylToR3(R3ToCyl(v)) = {CylToR3(R3ToCyl(v))}')
    print()

    print(f'CoordinateSystemTangentBundle = {CoordinateSystemTangentBundle}')
    #T_R3 = CoordinateSystemTangentBundle.induced(R3)
    T_R3 = TangentFunctor_ob(R3)
    print(f'T_R3 = {T_R3}')
    print(f'repr(T_R3) = {repr(T_R3)}')
    print()

    U = T_R3.create_coordinates()
    print(f'U = {U}')
    print()

    T_R3ToCyl = TangentFunctor_mor(R3ToCyl)
    print(f'T_R3ToCyl = {T_R3ToCyl}')
    # T of a function should really produce a matrix field (i.e. it takes a base point and produces a matrix)
    V = T_R3ToCyl(U)
    print(f'V = T_R3ToCyl(U) = {V}')
    print(f'V.shape = {V.shape}')
    print()

    u = R3.create_coordinates()
    J_R3ToCyl = R3ToCyl.jacobian_evaluator(u)
    print(f'u = {u}')
    print(f'J(R3ToCyl) = {J_R3ToCyl}')
    print()

    p = Cyl.create_coordinates()
    J_CylToR3 = CylToR3.jacobian_evaluator(p)
    print(f'p = {p}')
    print(f'J(CylToR3) = {J_CylToR3}')
    print()

    print('the following should be the identity matrix (J and J_inv composed)')
    print(f'{simplified(np.dot(CylToR3.jacobian_evaluator(R3ToCyl(u)), R3ToCyl.jacobian_evaluator(u)))}')
    print()

    T_Cyl = TangentFunctor_ob(Cyl)
    print(f'T_Cyl = {T_Cyl}')
    print(f'repr(T_Cyl) = {repr(T_Cyl)}')
    print()

    X = T_Cyl.create_coordinates()
    print(f'X = {X}')
    print()

    T_CylToR3 = TangentFunctor_mor(CylToR3)
    print(f'T_CylToR3 = {T_CylToR3}')
    # T of a function should really produce a matrix field (i.e. it takes a base point and produces a matrix)
    Y = T_CylToR3(X)
    print(f'Y = T_CylToR3(X) = {Y}')
    print(f'Y.shape = {Y.shape}')
    print()

    QC = CoordinateSystem(
        'QC',
        shape=(3,),
        default_name__to=np.array(['R','theta','w']),
        assumption__d=dict(real=True),
        symbol_assumption__d=dict(
            R=dict(
                positive=True,
            ),
        ),
    )
    R3ToQC = CoordinateChange(
        domain=R3,
        codomain=QC,
        evaluator=lambda euc:QC(np.array([euc[0]**2 + euc[1]**2, sp.atan2(euc[1], euc[0]), 4*euc[2]])),
        inverse_evaluator__o=lambda qc:R3(np.array([sp.sqrt(qc[0])*sp.cos(qc[1]), sp.sqrt(qc[0])*sp.sin(qc[1]), qc[2]/4])),
    )

    T_QC = TangentFunctor_ob(QC)
    T_R3ToQC = TangentFunctor_mor(R3ToQC)

    # TEMP HACK: just use tangent bundle to test this formula; it really should use cotangent bundle.
    p_qc = T_QC.create_coordinates()
    p_r3_fiber = simplified(np.dot(T_QC.fiber_projection(p_qc), R3ToQC.jacobian_evaluator(R3ToQC.inverse_evaluator(T_QC.base_point_projection(p_qc)))))
    print(f'p_r3_fiber = {p_r3_fiber}')

'''
class CoordinateSystem(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def shape (cls) -> typing.Tuple[int,...]:
        pass

    @classmethod
    def create (cls, name:str='q') -> np.ndarray:
        return vorpy.symbolic.tensor(name, cls.shape())

class CoordinateSystemVectorBundle(CoordinateSystem):
    @classmethod
    @abc.abstractmethod
    def base_coordinate_system (cls) -> CoordinateSystem:
        pass

#class CoordinateSystemTangentBundle(CoordinateSystem):
    #@classmethod
    #@abc.abstractmethod
    #def base_coordinates (cls) -> CoordinateSystem:
        #pass

    #@classmethod
    #def shape (cls) -> typing.Tuple[int,...]:
        #return (2,) + cls.base_coordinates().shape()

    #@classmethod
    #def create (cls, basepoint_name:str='q', fiber_name:str='v') -> np.ndarray:
        #retval = np.ndarray(cls.shape(), dtype=object)
        #retval[0,...] = cls.base_coordinates().create(basepoint_name)
        #retval[1,...] = cls.base_coordinates().create(fiber_name)
        #return retval

#class CoordinateSystemCotangentBundle(CoordinateSystem):
    #@classmethod
    #@abc.abstractmethod
    #def base_coordinates (cls) -> CoordinateSystem:
        #pass

    #@classmethod
    #def shape (cls) -> typing.Tuple[int,...]:
        #return (2,) + cls.base_coordinates().shape()

    #@classmethod
    #def create (cls, basepoint_name:str='q', fiber_name:str='p') -> np.ndarray:
        #retval = np.ndarray(cls.shape(), dtype=object)
        #retval[0,...] = cls.base_coordinates().create(basepoint_name)
        #retval[1,...] = cls.base_coordinates().create(fiber_name)
        #return retval

class CoordinateChange(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def domain_coordinate_system (cls) -> CoordinateSystem:
        pass

    @classmethod
    @abc.abstractmethod
    def codomain_coordinate_system (cls) -> CoordinateSystem:
        pass

    @classmethod
    @abc.abstractmethod
    def change (cls, coords:np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def change_inv (cls, coords:np.ndarray) -> np.ndarray:
        dom = cls.domain_coordinate_system().create()
        cod = cls.codomain_coordinate_system().create()
        solution_v = sp.solve((cod - cls.change(dom)).reshape(-1), *dom.reshape(-1).tolist())
        #print(f'solution_v:')
        #for solution in solution_v:
            #print(f'    {solution}')
        if len(solution_v) != 1:
            raise ValueError(f'Default implementation of CoordinateChange.change_inv did not find a solution; this method must be implemented explicitly with the solution.  solution_v = {solution_v}')
        return np.array(sp.Subs(solution_v[0], cod, coords).doit()).reshape(dom.shape)

#class CoordinateChangeTangentBundle(CoordinateChange):
    #@classmethod
    #@abc.abstractmethod
    #def base_coordinate_change (cls) -> CoordinateChange:
        #pass

    #@classmethod
    #def domain_coordinate_system (cls) -> CoordinateSystem:
        #pass

    #@classmethod
    #def codomain_coordinate_system (cls) -> CoordinateSystem:
        #pass

    #@classmethod
    #def change (cls, coords:np.ndarray) -> np.ndarray:
        ## TODO: Better yet, check that the type is correct (coords would need
        ## to carry type info on top of ndarray).
        #assert coords.shape == cls.domain_coordinate_system().shape()
        #changed_coords = np.ndarray(cls.codomain_coordinate_system().shape(), dtype=object)
        #changed_coords[0,...] = domain_coord

    #@classmethod
    #def change_inv (cls, coords:np.ndarray) -> np.ndarray:
        ## TODO: Check that the size is correct (or better yet, that the type is correct)
        #pass

if __name__ == '__main__':
    class Euclidean3CoordinateSystem(CoordinateSystem):
        @classmethod
        @abc.abstractmethod
        def shape (cls) -> typing.Tuple[int,...]:
            return (3,)

        @classmethod
        def create (cls, name_o:typing.Optional[str]=None) -> np.ndarray:
            if name_o is not None:
                return super(Euclidean3CoordinateSystem, cls).create(name_o)
            else:
                return np.array(sp.symbols('x,y,z'))

    class CylindricalCoordinateSystem(CoordinateSystem):
        @classmethod
        @abc.abstractmethod
        def shape (cls) -> typing.Tuple[int,...]:
            return (3,)

        @classmethod
        def create (cls) -> np.ndarray:
            return np.array([sp.Symbol('r', positive=True), sp.Symbol('theta'), sp.Symbol('z')])

    class EuclideanCylindricalCoordinateChange(CoordinateChange):
        # This is not quite right, since the type of Euclidean3CoordinateSystem
        # is not CoordinateSystem (it's a subclass of CoordinateSystem)
        @classmethod
        def domain_coordinate_system (cls) -> CoordinateSystem:
            return Euclidean3CoordinateSystem

        @classmethod
        def codomain_coordinate_system (cls) -> CoordinateSystem:
            return CylindricalCoordinateSystem

        @classmethod
        def change (cls, coords:np.ndarray) -> np.ndarray:
            x,y,z = coords
            return np.array([sp.sqrt(x**2 + y**2), sp.atan2(y,x), z])

        @classmethod
        def change_inv (cls, coords:np.ndarray) -> np.ndarray:
            # This implementation is needed because the default baseclass
            # implementation doesn't find a unique solution.
            r,theta,z = coords
            return np.array([r*sp.cos(theta), r*sp.sin(theta), z])

    class QuadraticCylindricalCoordinateSystem(CoordinateSystem):
        @classmethod
        @abc.abstractmethod
        def shape (cls) -> typing.Tuple[int,...]:
            return (3,)

        @classmethod
        def create (cls) -> np.ndarray:
            return np.array([sp.Symbol('R', positive=True), sp.Symbol('theta'), sp.Symbol('w')])

    class CylindricalQuadraticCylindricalCoordinateChange(CoordinateChange):
        @classmethod
        def domain_coordinate_system (cls) -> CoordinateSystem:
            return CylindricalCoordinateSystem

        @classmethod
        def codomain_coordinate_system (cls) -> CoordinateSystem:
            return QuadraticCylindricalCoordinateSystem

        @classmethod
        def change (cls, coords:np.ndarray) -> np.ndarray:
            r,theta,z = coords
            return np.array([r**2, theta, 4*z])

        #@classmethod
        #def change_inv (cls, coords:np.ndarray) -> np.ndarray:
            ## This implementation is needed because the default baseclass
            ## implementation doesn't find a unique solution.
            #R,theta,w = coords
            #return np.array([sp.sqrt(R), theta, w/4])

    class SizeCylindricalCoordinateSystem(CoordinateSystem):
        @classmethod
        @abc.abstractmethod
        def shape (cls) -> typing.Tuple[int,...]:
            return (3,)

        @classmethod
        def create (cls) -> np.ndarray:
            return np.array([sp.Symbol('s', positive=True), sp.Symbol('theta'), sp.Symbol('w')])

    class QuadraticCylindricalSizeCylindricalCoordinateChange(CoordinateChange):
        @classmethod
        def domain_coordinate_system (cls) -> CoordinateSystem:
            return QuadraticCylindricalCoordinateSystem

        @classmethod
        def codomain_coordinate_system (cls) -> CoordinateSystem:
            return SizeCylindricalCoordinateSystem

        @classmethod
        def change (cls, coords:np.ndarray) -> np.ndarray:
            R,theta,w = coords
            return np.array([R**2 + w**2, theta, w])

        @classmethod
        def change_inv (cls, coords:np.ndarray) -> np.ndarray:
            # This implementation is needed because the default baseclass
            # implementation doesn't find a unique solution.
            s,theta,w = coords
            return np.array([sp.sqrt(s - w**2), theta, w])

    print(type(Euclidean3CoordinateSystem))
    euc = Euclidean3CoordinateSystem.create()
    cyl = CylindricalCoordinateSystem.create()

    print(euc)
    print(cyl)
    print(EuclideanCylindricalCoordinateChange.change(euc))
    print()

    print(EuclideanCylindricalCoordinateChange.change_inv(cyl))
    print(EuclideanCylindricalCoordinateChange.change_inv(EuclideanCylindricalCoordinateChange.change(euc)))
    print(sp.simplify(EuclideanCylindricalCoordinateChange.change(EuclideanCylindricalCoordinateChange.change_inv(cyl)))) # Doesn't quite simplify down all the way -- probably need the assumption that theta \in (-pi, pi).
    print()

    qc = QuadraticCylindricalCoordinateSystem.create()
    print(qc)
    print(CylindricalQuadraticCylindricalCoordinateChange.change(cyl))
    print(CylindricalQuadraticCylindricalCoordinateChange.change_inv(qc))
    print(CylindricalQuadraticCylindricalCoordinateChange.change_inv(CylindricalQuadraticCylindricalCoordinateChange.change(cyl)))
    print()

    sc = SizeCylindricalCoordinateSystem.create()
    print(sc)
    print(QuadraticCylindricalSizeCylindricalCoordinateChange.change(qc))
    print(QuadraticCylindricalSizeCylindricalCoordinateChange.change_inv(sc))
    print(QuadraticCylindricalSizeCylindricalCoordinateChange.change_inv(QuadraticCylindricalSizeCylindricalCoordinateChange.change(qc)))
    print()

    #print(vorpy.symbolic.differential(QuadraticCylindricalSizeCylindricalCoordinateChange.change(qc), qc))
    #M = vorpy.symbolic.differential(QuadraticCylindricalSizeCylindricalCoordinateChange.change_inv(sc), sc)
    #print(M)
    #print(np.array(sp.Subs(M.T, sc, QuadraticCylindricalSizeCylindricalCoordinateChange.change(qc)).doit()).reshape(3,3))
'''
