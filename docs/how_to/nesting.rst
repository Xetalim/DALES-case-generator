Prepare nested DALES domains
============================

Nesting is a sequence of simulations, not a single preprocessing call. Each
child needs boundary and initial fields derived from its parent's run. First
prepare and run the parent; confirm its required output files exist; then
configure and prepare the child using those paths and a matching time interval.
Generating a child before its parent outputs are available cannot produce
valid parent-derived boundary data.

Topology and grid constraints
-----------------------------

Use ``NestingTopology`` to record the grids from outermost to innermost. The
simulation's ``my_idx`` identifies its grid in that ordered list. Grid
boundaries must align with the parent's open-boundary coordinates, and the
child must remain inside the parent. Vertical coordinates must also align; use
``child_grid.inherit_vertical_grid(parent_grid)`` when the child uses a subset
of the parent's levels.

The parent needs topology entries for the next child and the cross-sections
that supply its boundaries. Each child needs the corresponding topology plus
``Nest_in_Dales``. Its initial-field path points at the parent's ``input``
directory; its time-dependent boundary paths point at parent run output.

Prerequisites and workflow
--------------------------

The :doc:`IBM/LSM/nesting guide <../working_with_ibm_lsm_nesting>` has the
topology wiring and ``Nest_in_Dales`` path example. It assumes the parent
simulation has already been generated and run. The nested integration tests
also require a working DALES executable and parent output; they are listed in
the :doc:`developer test catalog <../developer/test_catalog>`.

See :doc:`LBC API <../api/lbc>` for the exported topology and open-boundary
objects.