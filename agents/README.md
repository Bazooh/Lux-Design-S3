

Observation :
-
    units :
        position     : int  (n_player, max_units, 2)
        energy       : int  (n_player, max_units)
-
    units_mask       : bool (n_player, max_units)
-
    sensor_mask      : bool (map_width, map_height)
-
    map_features     :
        energy       : int  (map_width, map_height)
        tile_type    : int  (map_width, map_height)
-
    relic_nodes      : int  (max_relic_nodes, 2)
-
    relic_nodes_mask : bool (max_relic_nodes)
-
    team_points      : int  (n_player)
-
    team_wins        : int  (n_player)
-
    steps            : int
-
    match_steps      : int