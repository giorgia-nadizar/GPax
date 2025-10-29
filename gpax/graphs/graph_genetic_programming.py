from flax import struct
import jax.numpy as jnp
from typing import Callable, Dict, Union, List, Tuple, Optional

from jax import random
from jax.lax import fori_loop

from qdax.custom_types import RNGKey, Genotype, Mask

from gpax.graphs.functions import FunctionSet


@struct.dataclass
class GGP:
    """Base class for Graph-based Genetic Programming (GGP) representations.

    Common parameters for GP encodings:

    Args:
        n_inputs: number of input values provided to the program/graph (excluding constants).
            Typically set to the environment’s observation size, e.g., `env.observation_size`.
        n_outputs: number of outputs produced by the GP individual.
            Typically set to the environment’s action size, e.g., `env.action_size`.
        function_set: set of allowed functions that nodes/instructions can use.
        input_constants: array of constant values that can be used as additional inputs.
        outputs_wrapper: function applied to the outputs before returning them
            (e.g., `tanh` to bound outputs).
        weighted_functions: whether the genotype will contain weighting factors for each node/program line.
        weighted_inputs: whether the genotype will contain weighting factors for each connection.
    """

    n_inputs: int
    n_outputs: int
    function_set: FunctionSet = FunctionSet()
    input_constants: jnp.ndarray = jnp.asarray([0.1, 1.0])
    outputs_wrapper: Callable = jnp.tanh
    weighted_functions: bool = False
    weighted_inputs: bool = False

    @property
    def n_functions(self) -> int:
        """Max number of functions that can be performed by GGP."""
        raise NotImplementedError

    def init(self, rnd_key: RNGKey, *args):
        """Initialize a random genotype (to be implemented by subclasses)."""
        raise NotImplementedError

    def apply(self, genotype: Genotype, obs: jnp.ndarray, weights: Dict[str, jnp.ndarray] = None, ) -> jnp.ndarray:
        """Evaluate a genotype on an input observation (subclass-specific)."""
        raise NotImplementedError

    def compute_active_mask(self, genotype: Genotype, ) -> Mask:
        """Compute the mask of active (expressed) elements in a genotype (subclass-specific)."""
        raise NotImplementedError

    def mutate(self,
               genotype: Genotype,
               rnd_key: RNGKey,
               p_mut_inputs: float = 0.1,
               p_mut_functions: float = 0.1,
               weights_mut_sigma: float = 0.1,
               mutation_probabilities: Optional[Dict[str, float]] = None
               ) -> Genotype:
        """Mutates a GGP genotype using int-flip mutation. If the genotype is weighted, the weights
            are mutated with Gaussian mutation.

            This mutation is implemented as a form of crossover with a newly
            generated "donor" genotype: for each gene, the value is taken from the
            donor with a low probability, otherwise kept from the original genotype.
            This ensures that all mutated genes remain valid (i.e., within the
            correct index ranges for their respective genotype section).

            The function is compatible with standard emitters when wrapped using
            `functools.partial`.

            Mutation probabilities and sigma can be specified either via individual arguments or by
            passing a dictionary to `mutation_probabilities`, the dictionary values override
            the individual arguments.

            Args:
                genotype: the CGP genotype parameters to mutate.
                rnd_key: JAX PRNG key for randomness.
                p_mut_inputs: probability of mutating each input connection gene
                    (ignored if overridden via `mutation_probabilities`).
                p_mut_functions: probability of mutating each function gene
                    (ignored if overridden via `mutation_probabilities`).
                weights_mut_sigma: mutation step for weights Gaussian mutation
                    (ignored if overridden via `mutation_probabilities`).
                mutation_probabilities: optional dictionary mapping genotype parts
                 to their mutation probabilities.

            Returns:
                The mutated genotype.
            """
        return self._mutate(genotype, rnd_key, p_mut_inputs, p_mut_functions, weights_mut_sigma,
                            mutation_probabilities)[0]

    def _mutate(self,
                genotype: Genotype,
                rnd_key: RNGKey,
                p_mut_inputs: float = 0.1,
                p_mut_functions: float = 0.1,
                weights_mut_sigma: float = 0.1,
                mutation_probabilities: Optional[Dict[str, float]] = None
                ) -> Tuple[Genotype, Genotype]:
        """Worker class for mutation that returns both the mutated genotype and the donor."""
        # extract mutation probabilities if passed through a dictionary
        mutation_probabilities = mutation_probabilities or {}
        p_mut_inputs = mutation_probabilities.get("inputs", p_mut_inputs)
        p_mut_functions = mutation_probabilities.get("functions", p_mut_functions)
        weights_mut_sigma = mutation_probabilities.get("weights_sigma", weights_mut_sigma)

        new_key, x_key, y_key, f_key, weights_key = random.split(rnd_key, 5)
        # generate the donor genotype -> only few genes from this will be used
        donor_genotype = self.init(new_key)
        weights_noise = weights_mut_sigma * random.normal(weights_key, shape=(self.n_functions * 3,))
        fn_w_noise, i1_w_noise, i2_w_noise = jnp.split(weights_noise, 3)

        return {
            "genes": {
                "inputs1": _mutate_subgenome(genotype["genes"]["inputs1"],
                                             donor_genotype["genes"]["inputs1"],
                                             x_key,
                                             p_mut_inputs),
                "inputs2": _mutate_subgenome(genotype["genes"]["inputs2"],
                                             donor_genotype["genes"]["inputs2"],
                                             y_key,
                                             p_mut_inputs),
                "functions": _mutate_subgenome(genotype["genes"]["functions"],
                                               donor_genotype["genes"]["functions"],
                                               f_key,
                                               p_mut_functions),
            },
            "weights": {
                "inputs1": genotype["weights"]["inputs1"] + self.weighted_inputs * i1_w_noise,
                "inputs2": genotype["weights"]["inputs2"] + self.weighted_inputs * i2_w_noise,
                "functions": genotype["weights"]["functions"] + self.weighted_functions * fn_w_noise,
            }
        }, donor_genotype

    def get_readable_expression(
            self,
            genotype: Genotype,
            inputs_mapping: Union[Dict[int, str], Callable[[int], str]] = None,
            outputs_mapping: Union[Dict[int, str], Callable[[int], str]] = None
    ) -> str:
        """Generate a human-readable symbolic representation of a GGP genotype.

            Unary functions are printed in the form:
                f(x)
            Binary functions are printed in the form:
                (x op y)
            where `op` is the function symbol (e.g., `+`, `*`, `sin`).

            Args:
                genotype: GGP genotype.
                inputs_mapping (dict[int,str] | callable[[int], str], optional):
                    Mapping from input indices to custom names.
                    - If a dict, keys are input indices
                    - If a callable, it is called with the input index and must
                      return the desired string
                    Defaults to "i0", "i1", ...
                outputs_mapping (dict[int,str] | callable[[int], str], optional):
                    Mapping from output indices to custom names.
                    - If a dict, keys are output indices
                    - If a callable, it is called with the output index and must
                      return the desired string
                    Defaults to "o0", "o1", ...

            Returns:
                str: A multi-line string, with one line per output, showing the
                symbolic expression computed for each CGP output node.

            Example:
                o0 = (i0+i1)
                o1 = sin(i2)
            """
        inputs_mapping = inputs_mapping or {}
        if isinstance(inputs_mapping, dict):
            inputs_mapping_fn = lambda idx: inputs_mapping.get(idx, f"i{idx}")
        else:
            inputs_mapping_fn = inputs_mapping

        outputs_mapping = outputs_mapping or {}
        if isinstance(outputs_mapping, dict):
            outputs_mapping_fn = lambda idx: outputs_mapping.get(idx, f"o{idx}")
        else:
            outputs_mapping_fn = outputs_mapping

        targets = self._get_readable_expression(genotype, inputs_mapping_fn, outputs_mapping_fn)
        return "\n".join(targets)

    def _get_readable_expression(
            self,
            genotype: Genotype,
            inputs_mapping_fn: Callable[[int], str],
            outputs_mapping_fn: Callable[[int], str]
    ) -> List[str]:
        """Worker class for computing the readable symbolic representation of a GGP genotype."""
        raise NotImplementedError

    def _weights_representations(self, genotype: Genotype, gene_idx: int) -> Tuple[str, str, str]:
        input_weight = f"{genotype['weights']['functions'][gene_idx]:.2f}*" if self.weighted_functions else ""
        x_weight = f"{genotype['weights']['inputs1'][gene_idx]:.2f}*" if self.weighted_inputs else ""
        y_weight = f"{genotype['weights']['inputs2'][gene_idx]:.2f}*" if self.weighted_inputs else ""
        return input_weight, x_weight, y_weight

    def init_weights(self, key: RNGKey) -> Dict[str, jnp.ndarray]:
        """Initialize the weights' dictionary."""
        random_weights = random.uniform(key=key, shape=(self.n_functions * 3,)) * 2 - 1
        random_node_weights, random_input_weights1, random_input_weights2 = jnp.split(random_weights, 3)
        return {
            "functions": random_node_weights if self.weighted_functions else jnp.ones_like(random_node_weights),
            "inputs1": random_input_weights1 if self.weighted_inputs else jnp.ones_like(random_input_weights1),
            "inputs2": random_input_weights2 if self.weighted_inputs else jnp.ones_like(random_input_weights2),
        }

    def _update_memory(self,
                       genotype: Genotype,
                       weights: Dict[str, jnp.ndarray],
                       memory: jnp.ndarray,
                       gene_idx: int,
                       memory_idx: Union[int, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Updates the memory at a given index computing the function at the genotype index."""
        f_idx = genotype["genes"]["functions"].at[gene_idx].get()
        x_arg = memory.at[genotype["genes"]["inputs1"].at[gene_idx].get()].get() * weights["inputs1"].at[gene_idx].get()
        y_arg = memory.at[genotype["genes"]["inputs2"].at[gene_idx].get()].get() * weights["inputs2"].at[gene_idx].get()
        f_computed = self.function_set.apply(f_idx, x_arg, y_arg) * weights["functions"].at[gene_idx].get()
        memory = memory.at[memory_idx].set(f_computed)
        return genotype, memory

    # descriptors that can be used with MAP Elites
    def compute_complexity(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the relative complexity of the graph/program.
            Relative complexity is measured as the fraction of computing power used w.r.t.
            that allowed. For CGP this boils down to the amount of used nodes, for LGP the
            amount of program lines used.
        """
        return jnp.expand_dims(jnp.mean(self.compute_active_mask(genotype)), axis=0)

    def compute_function_count(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the number of functions of each type used by the graph/program."""
        active_mask = self.compute_active_mask(genotype)
        f_genes = genotype["genes"]["functions"]

        def _count_functions(
                idx: int,
                f_counter: jnp.ndarray,
        ) -> jnp.ndarray:
            f_id = f_genes.at[idx].get()
            f_counter = f_counter.at[f_id].set(f_counter.at[f_id].get() + active_mask.at[idx].get())
            return f_counter

        functions_count = fori_loop(
            lower=0,
            upper=len(f_genes),
            body_fun=_count_functions,
            init_val=(jnp.zeros(len(self.function_set)))
        )
        return functions_count

    def compute_function_arities(self, genotype: Genotype) -> jnp.ndarray:
        """Compute the fraction of one/two arity functions employed in the graph/program."""
        functions_count = self.compute_function_count(genotype)
        one_arity_total = jnp.sum(jnp.where(self.function_set.arities == 1, functions_count, 0))
        two_arity_total = jnp.sum(jnp.where(self.function_set.arities == 2, functions_count, 0))
        return jnp.asarray([one_arity_total, two_arity_total]) / self.n_functions


def _mutate_subgenome(
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        key: RNGKey,
        p_mut: float
) -> jnp.ndarray:
    """Performs elementwise mutation of a genotype section.

        For each gene, a random number in [0, 1) is drawn. If the number is
        greater than `p_mut`, the gene is kept from the original subgenome (`x1`);
        otherwise, it is replaced with the corresponding gene from the donor
        subgenome (`x2`).

        Args:
            x1: Original subgenome array.
            x2: Donor subgenome array (must be the same shape as `x1`).
            key: JAX PRNG key used to generate mutation probabilities.
            p_mut: Probability of replacing each gene with the donor's value.

        Returns:
            The mutated subgenome array.
        """
    mutation_probs = random.uniform(key=key, shape=x1.shape)
    return jnp.where(mutation_probs > p_mut, x1, x2).astype(int)
