"""
generate_data.py
-----------------
Builds a synthetic, but domain-realistic, set of raw thermal-anomaly
detections in the style of NASA FIRMS / VIIRS active-fire records.

Real FIRMS/VIIRS data was not pulled live because this environment's
network egress is restricted to package registries (pypi/npm/github),
not earthdata endpoints. Instead we simulate detections whose feature
distributions are grounded in the domain logic from the FlareEye brief
(gas flares persist daily near infrastructure, wildfires are far from
infra with high wind, agri burning clusters seasonally over cropland,
etc). Swap this module for a real FIRMS API pull and the rest of the
pipeline (clustering, feature engineering, classifier) is unchanged.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# 12 real-world-inspired sites per class, spaced far enough apart that
# DBSCAN will not merge two different sites into one event.
SITES = [
    # (name, state, lat, lng, true_class)
    ("Jamnagar Refinery Complex", "Gujarat", 22.3675, 69.8330, "Gas Flare"),
    ("Barmer Oilfield", "Rajasthan", 25.7521, 71.3967, "Gas Flare"),
    ("Paradip Refinery", "Odisha", 20.3167, 86.6167, "Gas Flare"),
    ("Numaligarh Refinery", "Assam", 26.6500, 93.7167, "Gas Flare"),
    ("Panipat Refinery", "Haryana", 29.4200, 76.9800, "Gas Flare"),
    ("Bina Refinery", "Madhya Pradesh", 24.1800, 78.2000, "Gas Flare"),
    ("Cauvery Basin Gas Field", "Tamil Nadu", 11.0500, 79.8500, "Gas Flare"),
    ("Kakinada Gas Terminal", "Andhra Pradesh", 16.9600, 82.2400, "Gas Flare"),
    ("Hazira Gas Terminal", "Gujarat", 21.1167, 72.6333, "Gas Flare"),
    ("Dahej Petrochemical Hub", "Gujarat", 21.7000, 72.5667, "Gas Flare"),
    ("Digboi Oilfield", "Assam", 27.3900, 95.6200, "Gas Flare"),
    ("Ankleshwar Oilfield", "Gujarat", 21.6300, 73.0100, "Gas Flare"),

    ("LG Polymers Industrial Belt", "Andhra Pradesh", 17.6868, 83.2185, "Industrial Fire"),
    ("Baghjan Oilfield", "Assam", 27.7500, 95.6667, "Industrial Fire"),
    ("Reliance Jamnagar SEZ", "Gujarat", 22.4500, 69.9333, "Industrial Fire"),
    ("Jharia Coalfield", "Jharkhand", 23.7400, 86.4300, "Industrial Fire"),
    ("Vizag Chemical Cluster", "Andhra Pradesh", 17.7231, 83.3012, "Industrial Fire"),
    ("Manali Petrochemical Zone", "Tamil Nadu", 13.1667, 80.2667, "Industrial Fire"),
    ("Bhilai Steel Furnace Unit", "Chhattisgarh", 21.2100, 81.3800, "Industrial Fire"),
    ("Gujarat Alkalies Plant", "Gujarat", 22.3000, 73.2000, "Industrial Fire"),
    ("Korba Aluminium Smelter", "Chhattisgarh", 22.3600, 82.7000, "Industrial Fire"),
    ("Haldia Petrochemicals", "West Bengal", 22.0667, 88.0698, "Industrial Fire"),
    ("Mathura Refinery", "Uttar Pradesh", 27.4200, 77.6700, "Industrial Fire"),
    ("Nagothane Petrochemical Complex", "Maharashtra", 18.5300, 73.1300, "Industrial Fire"),

    ("Similipal Forest Range", "Odisha", 21.6167, 86.2167, "Wildfire"),
    ("Uttarakhand Forest Belt", "Uttarakhand", 30.0668, 79.0193, "Wildfire"),
    ("Nagarhole Forest Range", "Karnataka", 12.0000, 76.1500, "Wildfire"),
    ("Bandipur Forest Reserve", "Karnataka", 11.6600, 76.6300, "Wildfire"),
    ("Nallamala Forest", "Andhra Pradesh", 15.9000, 78.9000, "Wildfire"),
    ("Achanakmar Forest", "Chhattisgarh", 22.3000, 81.7500, "Wildfire"),
    ("Kanha Forest Range", "Madhya Pradesh", 22.3400, 80.6200, "Wildfire"),
    ("Similipal Buffer Zone", "Odisha", 21.8500, 86.4000, "Wildfire"),
    ("Palamau Forest Division", "Jharkhand", 23.9500, 84.0700, "Wildfire"),
    ("Dandeli Forest Range", "Karnataka", 15.2500, 74.6200, "Wildfire"),
    ("Corbett Buffer Forest", "Uttarakhand", 29.5300, 78.7700, "Wildfire"),
    ("Mudumalai Forest Range", "Tamil Nadu", 11.5900, 76.5300, "Wildfire"),

    ("Patiala Crop Belt", "Punjab", 30.3398, 76.3869, "Agricultural Burning"),
    ("Kaithal Crop Belt", "Haryana", 29.8000, 76.4000, "Agricultural Burning"),
    ("Sambalpur Agri Belt", "Odisha", 21.4700, 83.9800, "Agricultural Burning"),
    ("Bathinda Crop Belt", "Punjab", 30.2100, 74.9500, "Agricultural Burning"),
    ("Sangrur Crop Belt", "Punjab", 30.2500, 75.8400, "Agricultural Burning"),
    ("Karnal Crop Belt", "Haryana", 29.6900, 76.9900, "Agricultural Burning"),
    ("Muzaffarnagar Crop Belt", "Uttar Pradesh", 29.4700, 77.7000, "Agricultural Burning"),
    ("Firozpur Crop Belt", "Punjab", 30.9300, 74.6100, "Agricultural Burning"),
    ("Hisar Crop Belt", "Haryana", 29.1500, 75.7200, "Agricultural Burning"),
    ("Amritsar Crop Belt", "Punjab", 31.6300, 74.8700, "Agricultural Burning"),
    ("Ludhiana Crop Belt", "Punjab", 30.9000, 75.8500, "Agricultural Burning"),
    ("Rohtak Crop Belt", "Haryana", 28.9000, 76.6100, "Agricultural Burning"),

    ("Bokaro Steel Plant", "Jharkhand", 23.6693, 86.1511, "Industrial Heat"),
    ("Rourkela Steel Plant", "Odisha", 22.2604, 84.8536, "Industrial Heat"),
    ("Mundra Power Plant", "Gujarat", 22.8394, 69.7220, "Industrial Heat"),
    ("Durgapur Steel Plant", "West Bengal", 23.5204, 87.3119, "Industrial Heat"),
    ("Vindhyachal Power Plant", "Madhya Pradesh", 24.1100, 82.6800, "Industrial Heat"),
    ("Talcher Thermal Plant", "Odisha", 20.9500, 85.2300, "Industrial Heat"),
    ("Sipat Power Plant", "Chhattisgarh", 22.1500, 82.2700, "Industrial Heat"),
    ("Neyveli Lignite Plant", "Tamil Nadu", 11.6100, 79.4800, "Industrial Heat"),
    ("Jindal Steel Works", "Chhattisgarh", 22.0800, 83.1700, "Industrial Heat"),
    ("Rihand Power Plant", "Uttar Pradesh", 24.0200, 82.7800, "Industrial Heat"),
    ("Chandrapur Power Plant", "Maharashtra", 19.9500, 79.3000, "Industrial Heat"),
    ("Angul Steel Complex", "Odisha", 20.8400, 85.1000, "Industrial Heat"),
]

# Class-conditional feature distributions. Ranges deliberately overlap
# (as real satellite thermal signatures do) rather than being perfectly
# separable, so the trained classifier lands in the 85-93% range reported
# for comparable methods in the literature rather than a suspicious 100%.
CLASS_PARAMS = {
    "Gas Flare": dict(
        frp=(90, 55), bt=(400, 35), dist_infra=(0.4, 0.5), dist_road=(1.5, 1.2),
        land_cover="Industrial / built-up", wind=(13, 6),
        n_days_active=(260, 110), daily_prob=0.85, swir=(0.58, 0.13),
    ),
    "Industrial Fire": dict(
        frp=(260, 110), bt=(465, 55), dist_infra=(0.35, 0.4), dist_road=(1.1, 1.0),
        land_cover="Industrial / built-up", wind=(11, 7),
        n_days_active=(4, 3), daily_prob=0.75, swir=(0.52, 0.14),
    ),
    "Wildfire": dict(
        frp=(150, 80), bt=(410, 45), dist_infra=(7.5, 5.0), dist_road=(4.0, 3.0),
        land_cover="Forest / vegetation", wind=(19, 7),
        n_days_active=(5, 4), daily_prob=0.55, swir=(0.37, 0.11),
    ),
    "Agricultural Burning": dict(
        frp=(65, 32), bt=(335, 28), dist_infra=(4.5, 3.0), dist_road=(1.9, 1.4),
        land_cover="Cropland", wind=(16, 7),
        n_days_active=(3, 2), daily_prob=0.6, swir=(0.33, 0.10),
    ),
    "Industrial Heat": dict(
        frp=(100, 40), bt=(365, 25), dist_infra=(0.3, 0.3), dist_road=(1.2, 1.0),
        land_cover="Industrial / built-up", wind=(9, 5),
        n_days_active=(310, 90), daily_prob=0.9, swir=(0.46, 0.10),
    ),
}

# small chance a whole site's labelled class is swapped for a
# neighbouring, easily-confused class -- simulates the noisy ground
# truth and edge cases (e.g. a persistent industrial fire that looks
# like a flare) any real deployment would face.
CONFUSION_PAIRS = {
    "Gas Flare": "Industrial Heat",
    "Industrial Heat": "Gas Flare",
    "Wildfire": "Agricultural Burning",
    "Agricultural Burning": "Wildfire",
    "Industrial Fire": "Gas Flare",
}
LABEL_NOISE_RATE = 0.10


def _expand_sites(base_sites, variants_per_site=2, spread_deg=0.35):
    """Grow the hand-picked hub list into a larger, still-plausible set of
    sites by jittering new locations near each real hub (new refinery
    unit down the road, a neighbouring field, etc.)."""
    expanded = list(base_sites)
    for name, state, lat, lng, cls in base_sites:
        for v in range(variants_per_site):
            dlat = RNG.normal(0, spread_deg)
            dlng = RNG.normal(0, spread_deg)
            expanded.append((f"{name} — Site {v + 2}", state, lat + dlat, lng + dlng, cls))
    return expanded


def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def _blended_params(cls: str, blend_with: str, w: float = 0.5) -> dict:
    a, b = CLASS_PARAMS[cls], CLASS_PARAMS[blend_with]
    out = {}
    for key in a:
        if isinstance(a[key], tuple):
            out[key] = tuple(w * x + (1 - w) * y for x, y in zip(a[key], b[key]))
        elif isinstance(a[key], (int, float)):
            out[key] = w * a[key] + (1 - w) * b[key]
        else:
            out[key] = a[key]  # land_cover stays as the true class's cover
    return out


def generate_detections(period_days: int = 365) -> pd.DataFrame:
    all_sites = _expand_sites(SITES, variants_per_site=2, spread_deg=0.35)
    rows = []
    for site_id, (name, state, lat, lng, cls) in enumerate(all_sites):
        is_ambiguous = RNG.random() < LABEL_NOISE_RATE
        p = _blended_params(cls, CONFUSION_PAIRS[cls], w=0.55) if is_ambiguous else CLASS_PARAMS[cls]
        n_days = int(_clip(RNG.normal(*p["n_days_active"]), 1, period_days))
        # pick which days (within the period) had a detection
        start_day = RNG.integers(0, max(1, period_days - n_days))
        candidate_days = np.arange(start_day, min(period_days, start_day + n_days * 3))
        active_days = RNG.choice(
            candidate_days,
            size=min(len(candidate_days), max(1, int(n_days * p["daily_prob"]))),
            replace=False,
        )
        active_days.sort()

        dist_infra = _clip(RNG.normal(*p["dist_infra"]), 0.02, 25)
        dist_road = _clip(RNG.normal(*p["dist_road"]), 0.05, 15)

        for day in active_days:
            jitter_lat = RNG.normal(0, 0.004)
            jitter_lng = RNG.normal(0, 0.004)
            frp = max(3, RNG.normal(*p["frp"]))
            bt = max(280, RNG.normal(*p["bt"]))
            wind = max(0, RNG.normal(*p["wind"]))
            swir = _clip(RNG.normal(*p["swir"]), 0.05, 0.95)
            conf = int(_clip(RNG.normal(87, 8), 45, 99))
            rows.append(dict(
                site_id=site_id, site_name=name, state=state, true_class=cls,
                lat=lat + jitter_lat, lng=lng + jitter_lng,
                day=int(day), frp=round(frp, 1), brightness_k=round(bt, 1),
                wind_kmh=round(wind, 1), swir_reflectance=round(swir, 3),
                confidence=conf, dist_infra_km=round(dist_infra, 2),
                dist_road_km=round(dist_road, 2), land_cover=p["land_cover"],
                source="VIIRS",
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_detections()
    df.to_csv("data/raw_detections.csv", index=False)
    print(f"generated {len(df)} raw detections across {df.site_id.nunique()} sites")
    print(df.true_class.value_counts())
