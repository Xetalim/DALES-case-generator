Geometry
========

Grid definitions and geometric modification helpers exported by
``moda.Geometry``. The :doc:`first-case tutorial <../quickstart>` shows a
complete ``GridDales`` attachment.

``GridDales`` requires ``itot``, ``jtot``, ``kmax``, ``xsize``, ``ysize``,
``kmax_soil``, and ``dz0``. Horizontal lengths, origin coordinates, and
vertical spacing are in metres; latitude/longitude are degrees. ``x0`` and
``y0`` default to zero, while ``alpha=None`` or ``alpha=1`` selects uniform
vertical spacing. ``proj4`` or ``wkt`` may define a CRS; without one, raster
reprojection and other geospatial operations may be unavailable.

The grid writes the DALES ``DOMAIN`` namelist and provides coordinates used by
atmosphere profiles, surface datasets, IBM geometry, and nesting calculations.
``itot`` and ``jtot`` should leave at least four horizontal points per MPI
subdomain; the preprocessor reduces the requested core count when it can.

.. automodule:: moda.Geometry
   :members: GeometricModification, GridDales, GridDalesOpenBC, ModifierClass, masks
   :imported-members:
   :exclude-members: idxmesh, idymesh, meshx, meshy, x, y

Masks
-----

The geometry namespace also re-exports these masks from
``moda.Geometry.masks``:

.. autoclass:: moda.Geometry.masks.AllMask
   :members:

.. autoclass:: moda.Geometry.masks.CheckerboardIdxMask
   :members:

.. autoclass:: moda.Geometry.masks.CircleIdxMask
   :members:

.. autoclass:: moda.Geometry.masks.CircleRealMask
   :members:

.. autoclass:: moda.Geometry.masks.ExplicitArrayMask
   :members:

.. autoclass:: moda.Geometry.masks.FuncMask
   :members:

.. autoclass:: moda.Geometry.masks.RectangleIdxMask
   :members:

.. autoclass:: moda.Geometry.masks.RectangleRealMask
   :members:

.. autodata:: moda.Geometry.masks.MaskSpec