"""E3: human-robot safety scenarios in a 2D simulator (outside the E1 freeze).

* ``world``: a built scene as 2D occupancy grids (robot- and human-inflated) and semantic task points.
* ``walkers``: seeded task tours for workers, A* + smoothing, speed profile with corner slowing and lognormal dwell.
* ``sim``: the robot (diff drive, pure pursuit) under three policies (blind / fields, an ISO 3691-4-style stop-and-wait / human-aware) and the per-trial metrics.
"""
