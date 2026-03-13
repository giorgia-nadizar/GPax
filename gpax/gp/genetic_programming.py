from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import jax.numpy as jnp
from flax import struct
from qdax.custom_types import Genotype, RNGKey


@struct.dataclass
class GP(ABC):

    @abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> Any:
        """Initialize a random genotype."""
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        genotype: Genotype,
        inputs: jnp.ndarray,
        weights: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Evaluate the GP program on an observation.

        Args:
            genotype: GP genotype.
            inputs: Observation input array.
            weights: Optional dictionary of trainable weights to use during evaluation.

        Returns:
            jnp.ndarray: vector of program output(s).
        """
        raise NotImplementedError

    @abstractmethod
    def size(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the number of active (expressed) elements in a genotype.

        Args:
            genotype: GP genotype.

        Returns:
            jnp.ndarray: the size of a genotype.
        """

        raise NotImplementedError

    @abstractmethod
    def mutate(
        self,
        genotype: Genotype,
        rnd_key: RNGKey,
        *,
        mutation_probabilities: Optional[Dict[str, float]] = None,
    ) -> Genotype:
        """Mutate a genotype.

        Args:
            genotype: GP genotype to mutate.
            rnd_key: JAX PRNG key used for stochastic mutation.
            mutation_probabilities: Optional dictionary overriding mutation
                probabilities for genotype components.

        Returns:
            Genotype: Mutated GP genotype.
        """
        raise NotImplementedError

    @abstractmethod
    def get_readable_expression(
        self,
        genotype: Genotype,
        *,
        inputs_mapping: Union[Dict[int, str], Callable[[int], str], None] = None,
    ) -> str:
        """Generate a human-readable symbolic representation of a GP genotype.

        Unary functions are printed in the form:
            f(x)
        Binary functions are printed in the form:
            (x op y)
        where `op` is the function symbol (e.g., `+`, `*`, `sin`).

        Args:
            genotype: GP genotype.
            inputs_mapping (dict[int,str] | callable[[int], str], optional):
                Mapping from input indices to custom names.
                - If a dict, keys are input indices
                - If a callable, it is called with the input index and must
                  return the desired string
                Defaults to "i0", "i1", ...

        Returns:
            str: A showing the symbolic expression computed for the genotype.

        Example:
            y = ((i0+i1) * sin(i2))
        """
        raise NotImplementedError
