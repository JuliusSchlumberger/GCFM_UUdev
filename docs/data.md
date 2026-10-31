# Input data

Generated from `config/data_catalogue.yml`; do not edit. Regenerate with
`python tools/make_data_table.py`.

Every `file_path` is relative to `<raw_data_root>`, the `raw_data_root` key of
`config/site.yml` (see [install.md](install.md#site-settings)), for example:

```
raw_data_root: /data/gcfm/raw_data
```

The pipeline reads only `name`, `file_path` and `attributes` of each entry; the
other fields are documentation. "How to obtain" names the script in `tools/` that
downloads or builds the dataset; datasets without one are downloaded from the
source by hand. — marks a field the catalogue does not record.

| dataset | description | version | source / citation | licence | file_path | how to obtain |
|---|---|---|---|---|---|---|
| `coastal_bathymetry` | Global terrain model for ocean and land providing elevation in meters on a 15 arc-second interval grid. | — | GEBCO Compilation Group (2026) GEBCO 2026 Grid. Available at https://www.gebco.net/data-products/gridded-bathymetry-data. Published 23 April 2026 by the Nippon Foundation-GEBCO Seabed 2030 Project. Base grid: SRTM15+ V2.8 (Tozer et al. 2019) augmented with regional multibeam compilations from the four Seabed 2030 Regional Centers. | — | `gebco_2026_geotiff/` | manual download from the source |
| `delta_polygons` | Dataset of manually drawn polygons representing the delta areas globally based on earth-observation data. | — | Edmonds, D.A., Caldwell, R.L., Brondizio, E.S. et al. Coastal flooding will disproportionately impact people on river deltas. Nat Commun 11, 4741 (2020). https://doi.org/10.1038/s41467-020-18531-4 | — | `DeltaWebs/modified/9_polygons.gpkg` | manual download from the source |
| `egm2008_geoid` | EGM2008 Earth Gravitational Model (degree/order 2190). | — | Pavlis et al. (2012), The development and evaluation of the EGM2008. J. Geophys. Res. 117, B04406. | — | `ICGM/EGM2008.gfc` | manual download from the source |
| `fathomdem` | Global terrain elevation model covering 60°S–80°N at ~30 m (1 arcsecond) resolution, produced by removing surface artifacts (buildings, veg… | — | Uhe, P., Lucas, C., Hawker, L., Brine, M., Wilkinson, H., Cooper, A., Saoulis, A.A., Savage, J., & Sampson, C. (2025). FathomDEM: an improved global terrain map using a hybrid vision transformer model. Environmental Research Letters, 20, 034002. https://doi.org/10.1088/1748-9326/ada972 . Data: Zenodo record 14523356 (https://doi.org/10.5281/zenodo.14523356), restricted access via share-link token — see tools/download_fathomdem.py. | — | `FathomDEM/` | `tools/download_fathomdem.py` |
| `goco06s` | GOCO06s combined gravity field model (satellite-only, degree/order 300). | — | Kvas et al. (2021), GOCO06s — A satellite-only gravity field model. Earth System Science Data 13(1), 99–118. | — | `ICGM/GOCO06s.gfc` | manual download from the source |
| `grdc_discharge` | Daily river discharge observations from the Global Runoff Data Centre (GRDC), used as ground-truth gauge data to bias-correct GloFAS reanal… | — | Global Runoff Data Centre (GRDC), Koblenz, Germany. https://www.bafg.de/GRDC/ | — | `GRDC/GRDC-Daily.nc` | manual download from the source |
| `gtsm_station_subset` | Hourly total water level and surge residual of the GTSMv3 reanalysis at the GTSM stations behind every basin's surge stations, one station-… | — | Derived from gtsm_total_water_level and gtsm_surge_residual with tools/subset_gtsm_stations.py | — | `GTSM_storm_tide_hourly/station_subset.nc` | `tools/subset_gtsm_stations.py` |
| `gtsm_surge_residual` | Hourly storm surge residual (storm tide minus tide-only run) of the GTSMv3 reanalysis 1950-2024, same stations and layout as gtsm_total_wat… | — | Copernicus CDS sis-water-level-change-timeseries-cmip6 (GTSMv3 reanalysis, hourly). Downloaded with tools/download_cds_waterlevel.py | — | `GTSM_storm_tide_hourly/storm_surge_residual/` | `tools/download_cds_waterlevel.py` |
| `gtsm_total_water_level` | Hourly total water level (tide + surge + mean sea level) of the GTSMv3 reanalysis 1950-2024 at ~43,000 output stations worldwide, one NetCD… | — | Copernicus CDS sis-water-level-change-timeseries-cmip6 (GTSMv3 reanalysis, hourly). Downloaded with tools/download_cds_waterlevel.py | — | `GTSM_storm_tide_hourly/total_water_level/` | `tools/download_cds_waterlevel.py` |
| `land_use` | Global dynamic land cover map at 100 m spatial resolution, discrete classification layer. | — | Buchhorn, M., Smets, B., Bertels, L., De Roo, B., Lesiv, M., Tsendbazar, N.-E., Herold, M., Fritz, S. (2020). Copernicus Global Land Service: Land Cover 100m: collection 3: epoch 2019: Globe. https://doi.org/10.5281/zenodo.3939050. Methodology: Buchhorn et al. (2020), Remote Sensing 12(6), 1044. https://doi.org/10.3390/rs12061044. Product page: https://land.copernicus.eu/global/products/lc | — | `Copernicus/Copernicus_LandUse.tif` | manual download from the source |
| `land_use_esa_worldcover` | ESA WorldCover 2021 (v200): global land cover at 10 m from Sentinel-1 + Sentinel-2, 11 classes. | — | Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, W., Brockmann, C., Kirches, G., Wevers, J., Cartus, O., Santoro, M., Fritz, S., Lesiv, M., Herold, M., Tsendbazar, N.E., Xu, P., Ramoino, F., Arino, O. (2022). ESA WorldCover 10 m 2021 v200. https://doi.org/10.5281/zenodo.7254221 | — | `ESA_Worldcover/esa_worldcover_2021_v200.vrt` | `tools/build_esa_worldcover_vrt.py` |
| `lu_to_roughness_lookup` | Lookup table to convert land use classes from the Copernicus Land Use dataset to Manning's n roughness coefficients for hydrodynamic modeli… | — | https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/latest/developing-a-terrain-model-and-geospatial-layers/creating-land-cover-mannings-n-values-and-impervious-layers | — | `Copernicus/lu_to_roughness_lookup.csv` | manual download from the source |
| `mdt_cnes_cls22` | Mean Dynamic Topography CNES-CLS22 from AVISO (HYBRID-CNES-CLS2022). | — | CNES-CLS22 MDT, AVISO+ altimetry data centre. https://doi.org/10.24400/527896/a01-2021.001 | — | `Aviso/mdt_hybrid_cnes_cls22_cmems2020_global.nc` | manual download from the source |
| `protection_levels_flopros` | FLOPROS flood protection standards (design return period, years) joined to WRI Aqueduct's "geogunit_107" sub-national administrative units,… | — | FLOPROS: Scussolini, P., Aerts, J.C.J.H., Jongman, B., Bouwer, L.M., Winsemius, H.C., de Moel, H., Ward, P.J. (2016). FLOPROS: an evolving global database of flood protection standards. Natural Hazards and Earth System Sciences, 16, 1049-1061. https://doi.org/10.5194/nhess-16-1049-2016 . Spatial join to WRI Aqueduct geographical units: http://floods.wri.org/ | — | `protection_levels/FLOPROS_NL_geogunit_107.xlsx` | manual download from the source |
| `river_discharge` | Gridded daily hydrological reanalysis from the Global Flood Awareness System (GloFAS), produced by forcing the LISFLOOD hydrological model… | GloFAS v4.0 (released 2023-07-26) | Copernicus Emergency Management Service (CEMS), Global Flood Awareness System (GloFAS) v4.0, released 2023-07-26. Produced by ECMWF on behalf of the European Commission's Joint Research Centre. Available via the Copernicus Early Warning Data Store (EWDS): https://ewds.climate.copernicus.eu/ | — | `GloFAS/` | `tools/download_glofas.py` |
| `river_network` | SWORD (SWOT River Database) v17c, global reach-level vector river network with SWOT observation statistics, one file per region. | — | Altenau, E. H., Pavelsky, T. M., Durand, M. T., Yang, X., Frasson, R. P. d. M., & Bendezu, L. (2021). The Surface Water and Ocean Topography (SWOT) Mission River Database (SWORD): A global river network for satellite data products. Water Resources Research, 57, e2021WR030054. https://doi.org/10.1029/2021WR030054 . Data: SWORD v17c (July 2026, exported May 2026), see SWORD/v17c/v17c_release_notes.md and v17c_variable_reference.md. | — | `SWORD/v17c/sword_{region}_v17c.gpkg` | manual download from the source |
| `slr_ar6_regional` | IPCC AR6 regional sea-level-change projections (combined confidence-level output: ~1030 gauge-style locations plus 65160 points on a 1°x1°… | — | IPCC AR6 Sea Level Projections (Garner et al. 2021), via Zenodo DOI 10.5281/zenodo.5914710 | — | `SLR/ar6-regional-confidence/regional/confidence_output_files/` | manual download from the source |
| `storm_tide_return_periods` | Global dataset of coastal storm tide return periods (storm surge + tide) at output locations of the Global Tide and Surge Model (GTSM). | — | Dullaart, J.C.M., Muis, S., Bloemendaal, N., Chertova, M.V., Couasnon, A., & Aerts, J.C.J.H. (2021). Accounting for tropical cyclones more than doubles the global population exposed to low-probability coastal flooding. Institute for Environmental Studies, Vrije Universiteit Amsterdam. Contact: job.dullaart@vu.nl | — | `CoastRP/COAST-RP.nc` | manual download from the source |
| `wri_geogunit_107` | Global ~30 arcsec (~1 km) raster of WRI Aqueduct "geogunit_107" sub-national administrative unit IDs (Geogunits variable, float32, _FillVal… | — | WRI Aqueduct Global Flood Risk / GLOFRIS impact model. http://floods.wri.org/ . Institute for Environmental Studies VU Amsterdam. | — | `WRI/geogunit_107_all.nc` | manual download from the source |
| `wri_geogunit_107_list` | Row index -> ISO3 country code lookup for wri_geogunit_107's Geogunits raster values (same row order as protection_levels_flopros). | — | WRI Aqueduct Global Flood Risk / GLOFRIS impact model. http://floods.wri.org/ | — | `WRI/geogunit_107_all_list.xlsx` | manual download from the source |

## Adaptation geometries

Generated from `config/adaptation_strategies.yml`. Measure files needed by the
`adapt` target, relative to `<adaptation_data_root>` (site setting
`adaptation_data_root`). Only the strategies selected with
`target_strategies` need their files. `auto/<name>` entries are not raw data:
rule `build_adaptation_lines` derives them from the basin's SFINCS skeleton when
an `adapt` job needs them; the others are drawn per basin.

| file_path | strategy / measure | how to obtain |
|---|---|---|
| `auto/coastal_protection_weir_PC` | `protect_closed/coastal_levee` | built by rule `build_adaptation_lines` into `<results_dir>/{basin_id}/adaptation_lines/<name>.geojson` |
| `auto/coastal_protection_weir_PO` | `grey_protect_open/coastal_levee` | built by rule `build_adaptation_lines` into `<results_dir>/{basin_id}/adaptation_lines/<name>.geojson` |
| `auto/pump_river_mouth` | `protect_closed/pumps` | built by rule `build_adaptation_lines` into `<results_dir>/{basin_id}/adaptation_lines/<name>.geojson` |
| `auto/river_protection_weir_PO` | `grey_protect_open/river_levee` | built by rule `build_adaptation_lines` into `<results_dir>/{basin_id}/adaptation_lines/<name>.geojson` |
| `{basin_id}/advance_pumps.geojson` | `advance/pumps` | project-specific; obtain from the project team |
| `{basin_id}/land_reclamation.geojson` | `NbS_protect_open/nbs_land_reclamation` | project-specific; obtain from the project team |
| `{basin_id}/offshore_island.geojson` | `advance/offshore_barrier` | project-specific; obtain from the project team |
| `{basin_id}/water_retention_area_river.geojson` | `NbS_protect_open/water_retention` | project-specific; obtain from the project team |
