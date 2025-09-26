ONE_HOUR = 60*60
# Set up gaps
# =================== CASE 1 ==============================
gap_definitions = {
    "planned": {
        "antenna repointing": {"rate_per_year": 26, "duration_hr": 3.3},
    },
    "unplanned": {
        "PAAM": {"rate_per_year": 1095, "duration_hr": 0.027777},
    }
}

# Set up taper information

taper_defs = {
    "planned": {
        "antenna repointing": {"lobe_lengths_hr": ONE_HOUR},
    "unplanned": {
        "PAAM": {"lobe_lengths_hr": 30/ONE_HOUR},
    }
}
}


