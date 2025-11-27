experiments = [
    {
        "path": "/data/results/exp01_strong_cloudmask_regional_S2/",
        "percentile": {0.1, 0.5},
        "preprocessing": "strong_cloudmask",
        "clustering_resolution": 20,
        "method": "regional",
        "sensor": "S2",
    },
    {
        "path": "/data/results/exp02_low_cloudmask_local_MODIS/",
        "percentile": 0.05,
        "preprocessing": "low_cloudmask",
        "clustering_resolution": 30,
        "method": "local",
        "sensor": "MODIS",
    },
]
