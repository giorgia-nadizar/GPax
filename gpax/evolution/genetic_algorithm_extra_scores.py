from typing import Optional, Tuple

import jax

from qdax.core.emitters.emitter import EmitterState
from qdax.custom_types import Genotype, Metrics, RNGKey
from qdax.baselines.genetic_algorithm import GeneticAlgorithm

from gpax.evolution.ga_repertoire_extra_scores import GARepertoireExtraScores


class GeneticAlgorithmWithExtraScores(GeneticAlgorithm):
    """
        Extends GeneticAlgorithm to track additional metrics ('extra scores')
        for each individual.

        The main GA behavior is unchanged.
    """

    def init(
            self, genotypes: Genotype, population_size: int, key: RNGKey, lamarckian: bool = False
    ) -> Tuple[GARepertoireExtraScores, Optional[EmitterState], Metrics]:
        """Initialize a GARepertoire with an initial population of genotypes.

        Args:
            genotypes: the initial population of genotypes
            population_size: the maximal size of the repertoire
            key: a random key to handle stochastic operations
            lamarckian: a flag to state whether the genomes are replaced by their
             updated version before entering the repertoire

        Returns:
            The initial repertoire, an initial emitter state and a new random key.

        Note: it differs from the original GA as it stores the extra_scores in the
            GARepertoire.
        """

        # score initial genotypes
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(genotypes, subkey)
        genotypes = jax.lax.cond(
            lamarckian,
            lambda _: extra_scores["updated_params"],
            lambda _: genotypes,
            operand=None,
        )

        # init the repertoire
        repertoire = GARepertoireExtraScores.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            population_size=population_size,
            extra_scores=extra_scores,
            keys_extra_scores=extra_scores.keys(),
        )

        # get initial state of the emitter
        key, subkey = jax.random.split(key)
        emitter_state = self._emitter.init(
            key=subkey,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=None,
            extra_scores=extra_scores,
        )

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics

    def update(
            self,
            repertoire: GARepertoireExtraScores,
            emitter_state: Optional[EmitterState],
            key: RNGKey,
            lamarckian: bool = False,
    ) -> Tuple[GARepertoireExtraScores, Optional[EmitterState], Metrics]:
        """
        Performs one iteration of a Genetic algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.

        Args:
            repertoire: a repertoire
            emitter_state: state of the emitter
            key: a jax PRNG random key
            lamarckian: a flag to state whether the genomes are replaced by their
             updated version before entering the repertoire

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key

        Note: it differs from the original GA as it stores the extra_scores in the
            GARepertoire.
        """

        # generate offsprings
        key, subkey = jax.random.split(key)
        genotypes, extra_info = self._emitter.emit(repertoire, emitter_state, subkey)

        # score the offsprings
        key, subkey = jax.random.split(key)
        fitnesses, extra_scores = self._scoring_function(genotypes, subkey)

        genotypes = jax.lax.cond(
            lamarckian,
            lambda _: extra_scores["updated_params"],
            lambda _: genotypes,
            operand=None,
        )

        # update the repertoire
        repertoire = repertoire.add(genotypes, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=None,
            extra_scores={**extra_scores, **extra_info},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics  # type: ignore
