"""Eddy tracking on an analytic double-gyre flow.

Pipeline: analytic flow -> closed-form vortex centers -> noisy detections ->
per-frame linear assignment (Hungarian with a verified dual certificate) ->
track lifecycle -> optional fragment stitching -> metrics against truth.
"""

__version__ = "0.1.0"
