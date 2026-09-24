"""
Plant location registry for the Hydro Forecasting Agent.

Plant coordinates are kept here so OpenWeather does not need to geocode
plant names at runtime. Verify coordinates before operational use.
"""

PLANTS = {
    "BBO": {
        "name": "Bambarabatuoya MHPP",
        "country": "LK",
        "location_query": "Bambarabatuoya, Sri Lanka",
        "lat": 6.701453,
        "lon": 80.509725,
        "status": "coordinate_from_public_Vidullanka_annual_report",
    },
    "BTO": {
        "name": "Batathota MHPP",
        "country": "LK",
        "location_query": "Kuruwita, Sri Lanka",
        "lat": 6.812639,
        "lon": 80.375750,
        "status": "coordinate_from_public_Vidullanka_annual_report",
    },
    "EME": {
        "name": "Ethamala Ella MHPP",
        "country": "LK",
        "location_query": "Pitabeddara, Sri Lanka",
        "lat": 6.226917,
        "lon": 80.497917,
        "status": "coordinate_from_public_Vidullanka_annual_report",
    },
    "RDP": {
        "name": "Rideepana MHPP",
        "country": "LK",
        "location_query": "Badulla, Sri Lanka",
        "lat": 7.009389,
        "lon": 81.064083,
        "status": "coordinate_from_public_Vidullanka_annual_report",
    },
    "UDW": {
        "name": "Udawela MHPP",
        "country": "LK",
        "location_query": "Badulla, Sri Lanka",
        "lat": 7.056444,
        "lon": 81.060806,
        "status": "coordinate_from_public_Vidullanka_annual_report",
    },
    "LKM": {
        "name": "Lower Kothmale Oya MHPP",
        "country": "LK",
        "location_query": "Lower Kothmale Oya, Sri Lanka",
        "lat": 7.033347,
        "lon": 80.650803,
        "status": "coordinate_from_Vidullanka_public_disclosure",
    },
    "MGT": {
        "name": "Madugeta MHPP",
        "country": "LK",
        "location_query": "Neluwa, Sri Lanka",
        "lat": 6.370775,
        "lon": 80.408741,
        "status": "public_map_listing_verify_before_operational_use",
    },
    "WMB": {
        "name": "Wembiyagoda MHPP",
        "country": "LK",
        "location_query": "Wembiyagoda, Kalawana, Sri Lanka",
        "lat": 6.517639,
        "lon": 80.413278,
        "status": "coordinate_from_public_Vidullanka_annual_report",
    },
    "GNT": {
        "name": "Ganthuna MHPP",
        "country": "LK",
        "location_query": "Ganthuna, Aranayake, Sri Lanka",
        "lat": 7.127444,
        "lon": 80.404833,
        "status": "coordinate_from_public_Vidullanka_annual_report",
    },
    "BKN": {
        "name": "Bukinda SHPP",
        "country": "UG",
        "location_query": "Bukinda, Uganda",
        "lat": 1.074036,
        "lon": 30.760961,
        "status": "coordinate_from_public_Vidullanka_disclosure",
    },
    "MVB": {
        "name": "Muvumbe SHPP",
        "country": "UG",
        "location_query": "Kabale, Uganda",
        "lat": -1.308361,
        "lon": 30.148111,
        "status": "coordinate_from_public_Vidullanka_annual_report",
    },
}

NON_HYDRO = {
    "HRN": "Horana Solar",
    "VBL": "Vidul Biomass",
    "ORIC": "Orik Corporation solar project",
}


def get_plant(site):
    site = str(site).upper().strip()
    if site not in PLANTS:
        raise KeyError(
            f"Unknown hydro site '{site}'. "
            f"Available: {', '.join(sorted(PLANTS))}"
        )
    return PLANTS[site].copy()
