import numpy as np


def load_map(filename, xy_res, margin):
    """Load a map from a text file and create an occupancy grid.

    Creates an occupancy grid map where a node is considered blocked if it lies
    within 'margin' distance of an obstacle.

    Parameters
    ----------
    filename : str
        Path to the map text file.
    xy_res : float
        Grid resolution in x and y.
    margin : float
        Distance margin for obstacle inflation.

    Returns
    -------
    dict
        Map data with keys:
        - 'nodenumber': 1D array of node IDs (0-based)
        - 'blockflag': 1D array of occupancy flags (1=blocked, 0=free)
        - 'boundary': 1D array [x_min, y_min, x_max, y_max]
        - 'segment': 1D array [mx, my] grid dimensions
        - 'resolution': 1D array [xy_res, xy_res]
        - 'margin': float
        - 'basicdata': Nx7 array of raw boundary/obstacle data
        - 'obstacle_vertices': list of obstacle vertex arrays
        - 'block': Nx7 array with cell ranges and colors (or empty)
    """
    ## Parse the map file
    labels = []
    values = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            label = parts[0]
            nums = [float(x) for x in parts[1:]]
            ## Pad to 7 columns (boundary lines have 4 values, blocks have 7)
            while len(nums) < 7:
                nums.append(0.0)
            labels.append(label)
            values.append(nums)

    if len(values) == 0:
        ## Empty map file — create minimal map
        basicdata = np.zeros((1, 7))
        leftcorner_bound = basicdata[0, 0:2]
        rightcorner_bound = basicdata[0, 2:4]
        m = np.ceil((rightcorner_bound - leftcorner_bound) / np.array([xy_res, xy_res])).astype(int)
        mx, my = int(m[0]), int(m[1])
        num_nodes = mx * my if mx * my > 0 else 0
        return {
            'nodenumber': np.arange(num_nodes),
            'blockflag': np.zeros(num_nodes, dtype=int),
            'boundary': basicdata[0, 0:4],
            'segment': m,
            'resolution': np.array([xy_res, xy_res]),
            'margin': margin,
            'basicdata': basicdata,
            'obstacle_vertices': [],
            'block': np.array([]),
        }

    ## Reorder so boundary is row 0
    rawdata = np.array(values)
    boundary_idx = None
    for i, label in enumerate(labels):
        if label == 'boundary':
            boundary_idx = i
            break

    if boundary_idx is not None:
        basicdata = np.zeros_like(rawdata)
        basicdata[0, :] = rawdata[boundary_idx, :]
        row_idx = 1
        for i in range(len(labels)):
            if i != boundary_idx:
                basicdata[row_idx, :] = rawdata[i, :]
                row_idx += 1
    else:
        basicdata = rawdata.copy()

    row = basicdata.shape[0]

    leftcorner_bound = basicdata[0, 0:2]
    rightcorner_bound = basicdata[0, 2:4]

    ## Grid dimensions
    m = np.ceil((rightcorner_bound - leftcorner_bound) / np.array([xy_res, xy_res])).astype(int)
    mx, my = int(m[0]), int(m[1])
    num_nodes = mx * my

    ## Node numbers (0-based) and block flags
    node = np.arange(num_nodes)
    flag = np.zeros(num_nodes, dtype=int)

    obstacle_vertices = []
    block_info = np.array([])

    if row >= 2:
        ## Inflate obstacles by margin
        leftcorner_block = basicdata[1:row, 0:2] - margin
        rightcorner_block = basicdata[1:row, 2:4] + margin

        ## Convert obstacle corners to cell indices (0-based)
        n_upright_block = np.floor((rightcorner_block - leftcorner_bound) / np.array([xy_res, xy_res])).astype(int)
        n_lowleft_block = np.floor((leftcorner_block - leftcorner_bound) / np.array([xy_res, xy_res])).astype(int)

        ## Clamp to grid bounds (0-based)
        num_blocks = n_upright_block.shape[0]
        for i in range(num_blocks):
            n_upright_block[i, 0] = min(n_upright_block[i, 0], mx - 1)
            n_upright_block[i, 1] = min(n_upright_block[i, 1], my - 1)
            n_lowleft_block[i, 0] = max(n_lowleft_block[i, 0], 0)
            n_lowleft_block[i, 1] = max(n_lowleft_block[i, 1], 0)

        ## Mark occupied cells in blockflag
        for i in range(num_blocks):
            for cx in range(n_lowleft_block[i, 0], n_upright_block[i, 0] + 1):
                for cy in range(n_lowleft_block[i, 1], n_upright_block[i, 1] + 1):
                    nodeposition = cx + cy * mx
                    if 0 <= nodeposition < num_nodes:
                        flag[nodeposition] = 1

        ## Build block info (cell ranges + colors)
        block_info = np.zeros((num_blocks, 7))
        block_info[:, 0:2] = n_lowleft_block
        block_info[:, 2:4] = n_upright_block
        block_info[:, 4:7] = basicdata[1:row, 4:7]

        ## Build obstacle vertices
        for i in range(num_blocks):
            verts = np.array([
                [leftcorner_block[i, 0], leftcorner_block[i, 1]],
                [leftcorner_block[i, 0], rightcorner_block[i, 1]],
                [rightcorner_block[i, 0], leftcorner_block[i, 1]],
                [rightcorner_block[i, 0], rightcorner_block[i, 1]],
            ])
            obstacle_vertices.append(verts)

    return {
        'nodenumber': node,
        'blockflag': flag,
        'boundary': basicdata[0, 0:4],
        'segment': m,
        'resolution': np.array([xy_res, xy_res]),
        'margin': margin,
        'basicdata': basicdata,
        'obstacle_vertices': obstacle_vertices,
        'block': block_info,
    }
