"""Scientific primitives: structure parsing, geometry measurement, numbering.

This package holds the layer that *measures* rather than *judges*. Nothing in
here decides whether a candidate is good. It reads coordinates, computes
distances and angles, and maps residue indices between the three numbering
systems a run has to keep straight (candidate sequence index, structure author
numbering, reference numbering).

The split matters because the judgement layer is where thresholds live, and
thresholds must come from a sourced :class:`~eagent.schemas.templates.CatalyticTemplate`,
never from a measurement helper that quietly decided what "close enough" means.

Nothing is re-exported here on purpose. Importing ``eagent.science`` must not
drag in the structure parser, the geometry tables or the aligner as a side
effect, so that a module which only needs one of them states which one. Import
the submodule you need:

    from eagent.science.structure_io import read_mmcif
    from eagent.science.geometry import measure_all
    from eagent.science.numbering import build_map
"""

from __future__ import annotations

__all__: list[str] = []
