from copy import deepcopy


def make_sloppy(spec):
    fast_options = {
        ("Dipy", "3dSHORE_reconstruction"): {"extrapolate_scheme": "ABCD"},
        ("Dipy", "MAPMRI_reconstruction"): {
            "extrapolate_scheme": "ABCD",
            "anisotropic_scaling": False,
            "laplacian_weighting": 0.2,
        },
        ("DSI Studio", "connectivity"): {"fiber_count": 5000},
        ("DSI Studio", "tractography"): {"fiber_count": 5000},
        ("DSI Studio", "autotrack"): {
            "track_id": "Association_ArcuateFasciculusL,Association_ArcuateFasciculusR",
            "tolerance": "30,40",
            "track_voxel_ratio": 0.8,
        },
        ("MRTrix3", "tractography"): {
            "tckgen": {
                "select": 1000,
                "seed": 5000,
                "backtrack": "DELETE",
                "n_samples": "DELETE",
                "n_trials": "DELETE",
                "algorithm": "SD_Stream",
            }
        },
        ("MRTrix3", "connectivity"): {"tck2connectome": {"search_radius": "DELETE"}},
        ("MRTrix3", "global_tractography"): {"niters": 10000},
        ("pyAFQ", "pyafq_tractometry"): {
            "mapping_definition": 'AffMap(affine_kwargs={"level_iters": [10, 10, 10]})',
            "bundle_info": '["SLF_L", "ARC_L", "CST_L", "CST_R"]',
            "n_seeds": 10000,
            "random_seeds": True,
            "export": "all_bundles_figure",
        },
    }
    sloppy_spec = deepcopy(spec)
    sloppy_nodes = []

    for node in sloppy_spec.get("nodes", []):
        key = (node.get("software", ""), node.get("action", ""))
        if key not in fast_options:
            sloppy_nodes.append(node)
        else:
            new_options = fast_options[key]
            node_params = node.get("parameters", {})
            sloppy_params = update_params(node_params, new_options)
            node["parameters"] = sloppy_params
            sloppy_nodes.append(node)
    sloppy_spec["nodes"] = sloppy_nodes
    return sloppy_spec


def update_params(node_params, params_to_update, elem_name=""):
    for k in params_to_update:
        if k in node_params:
            elem_name = elem_name + "." + k if elem_name else k
            if isinstance(params_to_update[k], dict):
                update_params(node_params[k], params_to_update[k], elem_name)
            else:
                print(elem_name, ": [", node_params[k], "->", params_to_update[k], "]")
                if params_to_update[k] == "DELETE":
                    del node_params[k]
                else:
                    node_params[k] = params_to_update[k]
    return node_params
