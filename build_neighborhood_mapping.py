import requests
import json
import numpy as np

# Coordinates of key beaches in Vancouver
BEACHES = {
    "Kitsilano Beach": (49.27389, -123.15511),
    "English Bay Beach": (49.28648, -123.14349),
    "Spanish Banks Beach": (49.2762, -123.2178),
    "Sunset Beach": (49.2799, -123.1362),
    "Jericho Beach": (49.2718, -123.1900)
}

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in km
    R = 6371.0
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

def get_min_distance_to_beach(lat, lon):
    min_dist = float('inf')
    for beach_name, coords in BEACHES.items():
        dist = haversine(lat, lon, coords[0], coords[1])
        if dist < min_dist:
            min_dist = dist
    return min_dist

def build_mapping():
    print("Fetching local area boundary centroids...")
    url_boundary = "https://opendata.vancouver.ca/api/records/1.0/search/?dataset=local-area-boundary&rows=30"
    r = requests.get(url_boundary)
    boundary_records = r.json().get('records', [])
    
    local_area_centroids = {}
    for rec in boundary_records:
        fields = rec.get('fields', {})
        name = fields.get('name')
        geo_point = fields.get('geo_point_2d')
        if name and geo_point:
            lat, lon = geo_point
            local_area_centroids[name] = {
                "lat": lat,
                "lon": lon,
                "beach_dist_km": get_min_distance_to_beach(lat, lon)
            }
            
    print(f"Centroids retrieved for {len(local_area_centroids)} local areas.")
    
    print("\nFetching sample from property-tax-report to map codes...")
    # Fetch 5000 records to extract unique neighbourhood codes and their pcoords
    url_tax = "https://opendata.vancouver.ca/api/records/1.0/search/?dataset=property-tax-report&rows=5000"
    r_tax = requests.get(url_tax)
    tax_records = r_tax.json().get('records', [])
    
    # Group pcoords by neighborhood code
    code_to_pcoords = {}
    for rec in tax_records:
        fields = rec.get('fields', {})
        code = fields.get('neighbourhood_code')
        pcoord = fields.get('land_coordinate')
        if code and pcoord:
            if code not in code_to_pcoords:
                code_to_pcoords[code] = []
            code_to_pcoords[code].append(pcoord)
            
    print(f"Found {len(code_to_pcoords)} unique neighbourhood codes in tax sample.")
    
    # Map code to local area name
    code_to_local_area = {}
    for code, pcoords in code_to_pcoords.items():
        # Query property-addresses for these pcoords
        mapped = False
        # Try first few pcoords until we get a match
        for pcoord in pcoords[:10]:
            url_addr = f"https://opendata.vancouver.ca/api/records/1.0/search/?dataset=property-addresses&q=pcoord:{pcoord}"
            r_addr = requests.get(url_addr)
            addr_records = r_addr.json().get('records', [])
            if addr_records:
                fields = addr_records[0].get('fields', {})
                area = fields.get('geo_local_area')
                if area:
                    code_to_local_area[code] = area
                    mapped = True
                    break
        if mapped:
            print(f"Code {code} -> {code_to_local_area[code]}")
        else:
            print(f"Code {code} -> Could not map to a local area")
            
    # Combine to build final static mapping
    final_mapping = {}
    for code, area in code_to_local_area.items():
        if area in local_area_centroids:
            final_mapping[code] = {
                "neighbourhood_name": area,
                "latitude": local_area_centroids[area]["lat"],
                "longitude": local_area_centroids[area]["lon"],
                "distance_to_beach_km": round(local_area_centroids[area]["beach_dist_km"], 2)
            }
            
    print("\nFinal Mapping:")
    print(json.dumps(final_mapping, indent=4, ensure_ascii=False))
    
    # Save final mapping to JSON
    with open("neighbourhood_mapping.json", "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, indent=4, ensure_ascii=False)
    print("\nSaved mapping to neighbourhood_mapping.json")

if __name__ == "__main__":
    build_mapping()
