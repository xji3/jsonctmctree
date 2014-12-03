In this documentation file we explain an example input file.

The input file is in the [JSON](http://json.org/) meta-format.
It can be read and written by hand and by standard library functions
available for most programming languages.
As a meta-format, JSON has a similar but not identical niche as
[XML](http://en.wikipedia.org/wiki/XML) and
[RDF](http://en.wikipedia.org/wiki/Resource_Description_Framework).

JSON consists of recursively nested objects and arrays,
with terminal values consisting of strings, numbers, true, false, and nil.

The input consists primarily of information about the structure
of the model, specific rates associated with state transitions
and with edges of the branching timeline, and observations
at points on the branching timeline.

The top-level input json object has the following members:
 * `node_count` : The number of nodes in the branching timeline.
   These include the root node, branching points, terminal points,
   and any additional points where observations are available along an edge.
   In our example, there are nine nodes.
 * `process_count` : The number of different processes.
   Each edge between nodes on the branching timeline is assumed
   to evolve according to exactly one of these processes.
   In our example, there are two distinct processes, corresponding to
   presence or absence of a gene duplicate.
 * `state_space_shape` : This is an array of integers defining
   the shape of the multivariate state space.  Our example model
   is bivariate with each variable having four possible states,
   so the multivariate state space shape is `[4, 4]`.
 * `prior_feasible_states` : An array of prior feasible multivariate states
   at the root of the branching timeline.
   In our case, the root corresponds to a duplication event,
   so at the time of duplication the two variables in the bivariate
   model must share the same state.
   Therefore only four of the sixteen possible joint states are feasible.
 * `prior_distribution` : An array giving a prior probability for each
   of the listed feasible states.
 * `tree` : An object defining the shape of the branching timeline,
   the index of the process acting along each edge,
   and the edge-specific scaling factors of the process rates.
   The names `row` and `col` are in analogy to the representation
   of the branching timeline as a sparse matrix.
    * `row` : An array specifying the endpoint node closer to the root
      for each directed edge of the branching timeline.
    * `col` : An array specifying the endpoint node farther from the root
      for each directed edge of the branching timeline.
    * `rate` : An array specifying the rate scaling factor
      for each directed edge of the branching timeline.
    * `process` : An array specifying the index of the process
      acting on each directed edge of the branching timeline.
 * `requested_derivatives` : An array listing the edge indices
   for which edge-specific derivatives of log-likelihood are requested.
   In particular, it requests derivatives of log-likelihood
   with respect to the logarithm of the edge-specific rate scaling
   factors associated with the given edge indices.
 * `processes` : An array of objects representing processes
   that may act along one or more pre-specified edges of the branching timeline.
   Each object in the array is defined as follows:
    * `row` : An array of initial multivariate states for each
      feasible multivariate state transition.
      Because each multivariate state is defined by an array,
      this is an array of arrays.
    * `col` : An array of final multivariate states for each
      feasible multivariate state transition.
      Because each multivariate state is defined by an array,
      this is an array of arrays.
    * `rate` : An array of rates associated with each feasible
      multivariate state transition.
 * `observable_nodes` : An array of observable nodes.
   If more than one variable of the multivariate state is observable
   at the node, then the node index may be repeated in this list.
 * `observable_axes` : An array of observable axes.
   Together with the observable nodes array,
   these two arrays define which variables of the multivariate
   state are observable at which nodes.
   In our example, neither of the two variables of the bivariate process
   are observable at nodes that do not correspond to terminal nodes
   branching timeline, and at the terminal node that corresponds
   to the out-group, only one of the variables is observable.
 * `site_weights` : A weight associated to each of the observations
   that are assumed to be independently and identically distributed.
   This gives the caller flexibility to collapse multiple
   identical distributions into a single pattern
   with an appropriately increased weight.
   In our example we have not made use of this flexibility,
   so all of the site weights are 1 and the array
   of observations has repeated identical observations.
 * `iid_observations` : An array of joint observations
   of variables at nodes on the branching timeline.
   Each observation is an array defining the variable state
   for each observable axis on each observable node.
   In our example, the number of observations corresponds
   to the length of the sequence alignment,
   and each observation has length 9 because we have
   four terminal nodes for which both variables are observable
   and one terminal node for which a single variable is observable.