"""
Implements a family of symplectic integrators (for separable and nonseparable Hamiltonians).  A symplectic
integrator is one which preserves the symplectic form on the cotangent bundle of phase space.  Coordinates on
phase space are typically written as (q,p), which denote the position and momentum coordinates respectively.
A symplectic integrator will then integrate Hamilton's equations

    dq/dt =   \partial H / \partial p
    dp/dt = - \partial H / \partial q

where H(q,p) is the Hamiltonian (aka total energy) of the system.  A Hamiltonian is a scalar function H(q,p)
defining the total energy for the system.  A separable Hamiltonian has the form

    H(q,p) = K(p) + V(q)

where K and V are prototypically the kinetic and potential energy functions, respectively, and therefore a
nonseparable Hamiltonian can not be written in this form.

References

    https://en.wikipedia.org/wiki/Symplectic_integrator
    https://en.wikipedia.org/wiki/Energy_drift
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.043303 - Molei Tao - Explicit symplectic approximation of nonseparable Hamiltonians: Algorithm and long time performance
"""

from . import nonseparable_hamiltonian
from . import separable_hamiltonian
from . import exceptions
