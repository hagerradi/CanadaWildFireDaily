from pyproj import Transformer, CRS

def lonlat_to_canada_lambert(df, lon_col='lon', lat_col='lat'):
    """
    Convert lon/lat in EPSG:4269 (NAD83) to EPSG:3347 (NAD83 / Canada Lambert) in meters.
    Returns a copy of df with 'easting' and 'northing' columns and the target CRS object.
    """
    # Safety: require lon/lat columns
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{lon_col}' and '{lat_col}'")

    source_crs = CRS.from_epsg(4269)   # NAD83 geographic (degrees)
    target_crs = CRS.from_epsg(3347)   # NAD83 / Canada Lambert (meters)

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    lons = df[lon_col].to_numpy(dtype=float)
    lats = df[lat_col].to_numpy(dtype=float)

    # transform (vectorized)
    eastings, northings = transformer.transform(lons, lats)

    df['easting'] = eastings
    df['northing'] = northings
    df.attrs['target_crs'] = target_crs.to_string()

    return df, target_crs