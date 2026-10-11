"""Pinned PX4 WMM lookup, not a contemporary geographic field measurement.

Same bilinear grid and scaling as geo_mag_declination.cpp; no new dependency.
Fields in NED gauss, declination positive east, inclination positive down.
"""
from pathlib import Path
import re
import numpy as np
from .diagnostics import sha256

def field_vector(declination_deg,inclination_deg,strength_gauss):
    values=np.array([declination_deg,inclination_deg,strength_gauss],dtype=float)
    if not np.isfinite(values).all() or not -90<=inclination_deg<=90 or strength_gauss<=0:
        raise ValueError("invalid geographic magnetic field")
    dec,inc=np.deg2rad(values[:2]);s=values[2]
    return s*np.array([np.cos(inc)*np.cos(dec),np.cos(inc)*np.sin(dec),np.sin(inc)])

def table_value(table,lat,lon):
    if not np.isfinite([lat,lon]).all():raise ValueError("finite geographic position required")
    lat=np.clip(lat,-90,90)
    if lon>180:lon-=360
    if lon<-180:lon+=360
    minlat=np.clip(np.floor(lat/10)*10,-90,80)
    minlon=np.clip(np.floor(lon/10)*10,-180,170)
    i,j=int((minlat+90)/10),int((minlon+180)/10)
    x,y=np.clip((lon-minlon)/10,0,1),np.clip((lat-minlat)/10,0,1)
    bottom=table[i,j]+x*(table[i,j+1]-table[i,j])
    top=table[i+1,j]+x*(table[i+1,j+1]-table[i+1,j])
    return float(bottom+y*(top-bottom))

def field_from_px4(root,latitude=47.397742,longitude=8.545594):
    root=Path(root);header=root/"src/lib/world_magnetic_model/geo_magnetic_tables.hpp"
    text=header.read_text();values=[]
    for name,scale in (("declination","WMM_DECLINATION_SCALE_TO_DEGREES"),
                       ("inclination","WMM_INCLINATION_SCALE_TO_DEGREES"),
                       ("totalintensity","WMM_TOTALINTENSITY_SCALE_TO_NANOTESLA")):
        match=re.search(name+r"_table\[19\]\[37\]\s*\{(.*?)\n\};",text,re.S)
        if match is None:raise ValueError("unsupported PX4 WMM table layout")
        body=re.sub(r"//[^\n]*|/\*.*?\*/","",match[1],flags=re.S)
        table=np.array([int(x) for x in re.findall(r"-?\d+",body)],dtype=float)
        if table.size!=19*37:raise ValueError("incomplete PX4 WMM table")
        multiplier=float(re.search(scale+r"\s*=\s*([0-9.eE+-]+)f;",text)[1])
        values.append(table_value(table.reshape(19,37),latitude,longitude)*multiplier)
    dec,inc,strength=values;strength*=1e-5 # nT -> gauss
    return {"reference_latitude_deg":latitude,"reference_longitude_deg":longitude,
            "declination_deg":dec,"inclination_deg":inc,"strength_gauss":strength,
            "field_ned_gauss":field_vector(dec,inc,strength).tolist(),
            "table_sha256":sha256(header),
            "lookup_source_sha256":sha256(root/"src/lib/world_magnetic_model/geo_mag_declination.cpp"),
            "assumption":"PX4-pinned WMM2020 grid dated2024.41257, not real2026 geomagnetic measurement"}
