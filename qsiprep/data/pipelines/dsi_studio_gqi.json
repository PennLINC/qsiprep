{
  "name": "dsistudio_pipeline",
  "space": "T1w",
  "atlases": ["schaefer100", "schaefer200"],
  "nodes": [
    {
      "name": "dsistudio_gqi",
      "software": "DSI Studio",
      "action": "reconstruction",
      "input": "qsiprep",
      "output_suffix": "gqi",
      "parameters": {"method": "gqi"}
    },
    {
      "name": "scalar_export",
      "software": "DSI Studio",
      "action": "export",
      "input": "dsistudio_gqi",
      "output_suffix": "gqiscalar"
    },
    {
      "name": "streamline_connectivity",
      "software": "DSI Studio",
      "action": "connectivity",
      "input": "dsistudio_gqi",
      "output": "",
      "parameters": {
        "turning_angle": 35,
        "method": 0,
        "smoothing": 0.0,
        "step_size": 1.0,
        "min_length": 10,
        "max_length": 250,
        "seed_plan": 0,
        "interpolation": 0,
        "initial_dir": 2,
        "fiber_count": 5000000,
        "connectivity_value": "count,ncount,mean_length,gfa",
        "connectivity_type": "pass,end",
        "output_trk": "no_trk"
      }
    },
    {
      "name": "controlability",
      "input": "streamline_connectivity",
      "action": "controllability",
      "output_suffix": "control"
    }
  ]
}
