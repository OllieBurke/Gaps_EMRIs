ONE_HOUR = 60*60
# Set up gaps
# =================== CASE 1 ==============================
# gap_definitions = {
#     "planned": {
#         "antenna repointing": {"rate_per_year": 26, "duration_hr": 3.3},
#     },
#     "unplanned": {
#         "PAAM": {"rate_per_year": 1095, "duration_hr": 0.027777},
#     }
# }

# # Set up taper information

# taper_defs = {
#     "planned": {
#         "antenna repointing": {"lobe_lengths_hr": ONE_HOUR}
#     },
#     "unplanned": {
#         "PAAM": {"lobe_lengths_hr": 30/ONE_HOUR},
#     }
# }

# include_planned=True
# include_unplanned=True
# planned_seed = 1234
# unplanned_seed = 4321


# =================== CASE 2 ==============================
gap_definitions = {
    "planned": {
        "antenna repointing": {"rate_per_year": 10, "duration_hr": 10*24},
    },
    "unplanned": {
        "PAAM": {"rate_per_year": 1095, "duration_hr": 0.027777},
    }
}

# Set up taper information

taper_defs = {
    "planned": {
        "antenna repointing": {"lobe_lengths_hr": ONE_HOUR}
    },
    "unplanned": {
        "PAAM": {"lobe_lengths_hr": 30/ONE_HOUR},
    }
}

include_planned=True
include_unplanned=None
planned_seed = 1234
unplanned_seed = 4321


# =================== Full Shamalama ==============================
# gap_definitions = {
#     "planned": {
#         "antenna repointing": {"rate_per_year": 26, "duration_hr": 3.3},
#         "TM stray potential": {"rate_per_year": 2, "duration_hr": 24},
#         "TTL calibration": {"rate_per_year": 4, "duration_hr": 48},
#         #"Aliens": {"rate_per_year": 6, "duration_hr": 30*24}
#     },
#     "unplanned": {
#         "platform safe mode": {"rate_per_year": 3, "duration_hr": 60},
#         "payload safe mode": {"rate_per_year": 4, "duration_hr": 66},
#         "QPD loss micrometeoroid": {"rate_per_year": 5, "duration_hr": 24},
#         "HR GRS loss micrometeoroid": {"rate_per_year": 19, "duration_hr": 24},
#         "WR GRS loss micrometeoroid": {"rate_per_year": 6, "duration_hr": 24},
#     }
# }

# # Set up taper information

# taper_defs = {
#     "planned": {
#         "antenna repointing": {"lobe_lengths_hr": ONE_HOUR},
#         "TM stray potential": {"lobe_lengths_hr": 0.0},
#         "TTL calibration": {"lobe_lengths_hr": 0.0},
#     },
#     "unplanned": {
#         "platform safe mode": {"lobe_lengths_hr": 30/ONE_HOUR},
#         "payload safe mode": {"lobe_lengths_hr": 30/ONE_HOUR},
#         "QPD loss micrometeoroid": {"lobe_lengths_hr": 30/ONE_HOUR},
#         "HR GRS loss micrometeoroid": {"lobe_lengths_hr": 30/ONE_HOUR},
#         "WR GRS loss micrometeoroid": {"lobe_lengths_hr": 30/ONE_HOUR},
#     }
# }

# include_planned=True
# include_unplanned=True
# planned_seed = 1234
# unplanned_seed = 4321