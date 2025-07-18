mkdir params
mkdir stats
cd params
wget https://storage.googleapis.com/dm_graphcast/graphcast/params/GraphCast_operational%20-%20ERA5-HRES%201979-2021%20-%20resolution%200.25%20-%20pressure%20levels%2013%20-%20mesh%202to6%20-%20precipitation%20output%20only.npz
cd ../stats
wget https://storage.googleapis.com/dm_graphcast/graphcast/stats/stddev_by_level.nc
wget https://storage.googleapis.com/dm_graphcast/graphcast/stats/mean_by_level.nc
wget https://storage.googleapis.com/dm_graphcast/graphcast/stats/diffs_stddev_by_level.nc