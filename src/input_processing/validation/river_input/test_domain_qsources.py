import random

import matplotlib.pyplot as plt
from geopandas import GeoDataFrame

from src.input_processing.config.loader import config
from src.input_processing.utils.defining_model_domain import get_relevant_basins_and_rivers, clip_basin_boundary_from_coast
from src.input_processing.utils.validation.modify_delta_masks import classify_points
from src.input_processing.utils.identify_river_source_points import extract_cells_within_delta
from src.input_processing.utils.plotting import plot_model_domain
from src.input_processing.utils.loading_files import load_data_delta_domain
import geopandas as gpd
from shapely.geometry import  Point, Polygon, MultiLineString, LineString


def test_validate_process_map(
        testcase_id: str='id_delta1',
        use_basins: bool=True,
) -> None:
    delta_domain = gpd.read_file(config['filepaths']['delta_polygons_used'])
    delta_polygon_gdf = delta_domain[delta_domain['BasinID2'] == config['Testcase'][testcase_id]].to_crs(
        epsg=config['CRS']['standard'])
    delta_polygon = delta_polygon_gdf.geometry.values[0]

    river_basins_gpd = gpd.read_file(config['filepaths']['new_domains'])
    delta_basins_gpd = river_basins_gpd[river_basins_gpd.BasinID2 == config['Testcase'][testcase_id]]

    mask_basin = delta_basins_gpd.union_all()
    rivers_gpd, coast_polygon, coastline_gpd, glofas_min = load_data_delta_domain(mask_basin)

    # Relevant rivers
    relevant_rivers = gpd.sjoin(rivers_gpd, delta_basins_gpd, how='inner')
    relevant_rivers_gpd = relevant_rivers[rivers_gpd.columns]

    # Domain boundaries
    if use_basins:  # using and identifying points at the watershed
        inland_boundary = clip_basin_boundary_from_coast(delta_basins_gpd, coast_polygon)
    else:   # using the delta polygons by Edmonds
        bw, s1, s2, _, _ = classify_points(delta_polygon, coast_polygon)
        inland_boundary = gpd.GeoSeries(
            MultiLineString([LineString([s1, bw]), LineString([bw, s2])]),
            crs=config['CRS']['standard']
        ).union_all()

    # Extract cells and unique sources
    (gdf_unique_sources,
     gdf_all_sources,
     glofas_p) = extract_cells_within_delta(
        glofas_min,
        inland_boundary,
        relevant_rivers_gpd
    )

    # Plot
    plot_model_domain(
        delta_polygon,
        river_basins_gpd,
        delta_basins_gpd,
        relevant_rivers_gpd,
        rivers_gpd,
        delta_polygon.boundary,
        inland_boundary,
        glofas_p,
        gdf_all_sources,
        gdf_unique_sources
    )


def test_domain_qsource_flow(n_plots: int = 10):

    delta_domain = gpd.read_file(config['filepaths']['delta_polygons'])
    all_basins = sorted(set(delta_domain['BasinID2'].values))
    random_domains = random.sample(all_basins, n_plots)

    for coastal_delta in random_domains:
        delta_polygon_gdf = delta_domain[delta_domain['BasinID2'] == coastal_delta].to_crs(
            epsg=config['CRS']['standard'])
        create_process_map(delta_polygon_gdf)

def create_process_map(
        delta_polygon_gdf: GeoDataFrame,
        use_basins: bool = True,
        river_basin_path: str = config['filepaths']['river_basins_applied']
) -> None:

    # Identify mask
    polygon_id = delta_polygon_gdf["BasinID2"].values[0]
    print(type(polygon_id), polygon_id)
    river_basins_gpd = gpd.read_file(config['filepaths']['new_domains'])
    if polygon_id not in river_basins_gpd.BasinID2.values:
        print('Deltapolygon by Edmonds not in new Basins', polygon_id)

    print(type(river_basins_gpd.BasinID2.values[0]), river_basins_gpd.BasinID2.values[0])
    print(river_basins_gpd)
    river_basins_gpd = river_basins_gpd[river_basins_gpd.BasinID2 == polygon_id]
    print(river_basins_gpd)

    print(river_basins_gpd)
    _, river_basins_gpd, rivers_gpd, coast_polygon, coastline_gpd, glofas_min = load_data_delta_domain(
        river_basins_gpd,
        river_basin_path)
    print(delta_polygon_gdf)
    delta_polygon = delta_polygon_gdf.geometry.values[0]

    # Relevant basins and rivers
    relevant_basins, relevant_rivers_gpd = get_relevant_basins_and_rivers(
        delta_polygon_gdf, river_basins_gpd, rivers_gpd
    )

    # Domain boundaries
    if use_basins:  # using and identifying points at the watershed
        print(relevant_basins)
        inland_boundary = clip_basin_boundary_from_coast(relevant_basins, coast_polygon)
    else:  # using the delta polygons by Edmonds
        bw, s1, s2, _, _ = classify_points(delta_polygon, coast_polygon)
        inland_boundary = gpd.GeoSeries(
            MultiLineString([LineString([s1, bw]), LineString([bw, s2])]),
            crs=config['CRS']['standard']
        ).union_all()

    # Extract cells and unique sources
    (gdf_unique_sources,
     gdf_all_sources,
     glofas_p) = extract_cells_within_delta(
        glofas_min,
        inland_boundary,
        relevant_rivers_gpd
    )

    # Plot
    plot_model_domain(
        delta_polygon,
        river_basins_gpd,
        relevant_basins,
        relevant_rivers_gpd,
        rivers_gpd,
        delta_polygon.boundary,
        inland_boundary,
        glofas_p,
        gdf_all_sources,
        gdf_unique_sources,
        delta_polygon_gdf["BasinID2"]

    )
