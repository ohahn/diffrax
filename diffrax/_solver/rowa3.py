from collections.abc import Callable
from typing import ClassVar, Union, Literal, Optional
from typing_extensions import TypeAlias

from equinox.internal import ω
import lineax as lx
from jaxtyping import PyTree

import jax.numpy as jnp
import jax

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._heuristics import is_sde
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractAdaptiveSolver

_SolverState: TypeAlias = None


class RoWa3(AbstractAdaptiveSolver):
  r"""Original linearly implicit Rosenbrock-Wanner method of 3rd order with embedded 2nd order error control.

  The Rosenbrock-Wanner methods are a class of implicit Runge-Kutta methods that are linearly implicit, i.e. the implicit stage equations can be solved with a matrix inversion.

  This method corresponds to Matlab's ode23s method.
  """

  scan_kind: Union[None, Literal["lax", "checkpointed", "bounded"]] = None

  term_structure: ClassVar = AbstractTerm

  # jac_f: Callable = None
  linear_solver: lx.AbstractLinearSolver = lx.LU
  init_later_state: Optional[PyTree] = None

  interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation
  
  def order(self, terms):
    return 3
  
  def error_order(self, terms):
    if is_sde(terms):
      return None
    else:
      return 2
  
  def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None
        
  def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        # del made_jump
        control = terms.contr(t0, t1)

        # coefficients        
        sqrt2 = jnp.sqrt(2)
        a  = 1/(2 + sqrt2)
        d31 = - (4 + sqrt2) / (2 + sqrt2)
        d32 = (6 + sqrt2) / (2 + sqrt2)


        # Jacobian
        J = lx.JacobianLinearOperator(lambda y,arg : terms.vf_prod(t0, y, arg, control), y0, args)
        J = lx.linearise(J)
        n = y0.shape[0]

        # linear inverse and its LU factorization
        I  = jnp.eye(n)
        A  = I - a * J.as_matrix()
        LU_and_piv = jax.scipy.linalg.lu_factor(A, overwrite_a=True, check_finite=False)

        # Stage 1
        b1 = terms.vf_prod(t0, y0, args, control)
        k1 = jax.scipy.linalg.lu_solve( LU_and_piv, b1 )
        
        # Stage 2
        t1, y1 = t0, (y0**ω + 0.5 * k1**ω).ω
        Jk1 = J.mv(k1)
        b2 = (terms.vf_prod(t1, y1, args, control)**ω - a * Jk1**ω).ω
        k2 = jax.scipy.linalg.lu_solve( LU_and_piv, b2 )

        # Stage 3
        t2, y2 = t0, (y0**ω + k2**ω).ω
        Jk2 = J.mv(k2)
        b3 = (terms.vf_prod(t2, y2, args, control)**ω - (d31 * Jk1**ω + d32 * Jk2**ω)).ω
        k3 = jax.scipy.linalg.lu_solve( LU_and_piv, b3 )

        # Advance solution
        y1 = (y0**ω + 1/6 * (k1**ω + 4 * k2**ω + k3**ω)).ω
        y_error = (y1**ω - y0**ω - k2**ω).ω

        dense_info = dict(y0=y0, y1=y1)

        return y1, y_error, dense_info, solver_state, RESULTS.successful


  def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
  


