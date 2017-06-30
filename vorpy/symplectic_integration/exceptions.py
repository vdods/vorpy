class SalvagedResultException(Exception):
    """
    Integrating an ODE can be an expensive operation, and getting some obscure numerical error
    during integration and losing all results is frustrating.  Thus this class is intended to
    solve that problem -- the symplectic integrators will catch instances of Exception, and re-raise
    an instance of this class, attaching the results computed so far.  This way, the cause of the
    failure can be more readily determined, such as by plotting the salvaged curve.

    The salvaged_qp_v attribute will be a numpy.ndarray of shape (S,...), where S is the number of
    successfully computed steps thus far (successful meaning "did not generate an exception"), and
    the ... denotes the shape of the initial conditions parameter qp_0 passed into the integrator.
    The value salvaged_qp_v[N,...] is the integrated coordinates of the Nth timestep of the call to
    the integrator.  The intent is for the salvaged values to all be valid (e.g. no infinities or NaNs).
    The corresponding timesteps, having shape (S,), are stored in the salvaged_t_v attribute.
    """

    def __init__(self, *, original_exception, salvaged_t_v, salvaged_qp_v):
        # TODO: Should this pass in original_exception or original_exception.args or something?
        super().__init__('salvaged {0} integration steps from nonseparable_hamiltonian.integrate that failed because of exception {1}'.format(salvaged_qp_v.shape[0], original_exception))

        self.with_traceback(original_exception.__traceback__)

        if len(salvaged_t_v.shape) != 1:
            print('WARNING: SalvagedResultException expected len(salvaged_t_v.shape) == 1, but got len(salvaged_t_v.shape): {0}'.format(len(salvaged_t_v.shape)))
        if salvaged_t_v.shape[0] != salvaged_qp_v.shape[0]:
            print('WARNING: SalvagedResultException expected salvaged_t_v.shape[0] == salvaged_qp_v.shape[0], but got salvaged_t_v.shape[0]: {0} and salvaged_qp_v.shape[0]: {1}'.format(salvaged_t_v.shape[0], salvaged_qp_v.shape[0]))

        self.salvaged_t_v    = salvaged_t_v
        self.salvaged_qp_v   = salvaged_qp_v
