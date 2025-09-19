import numpy as np
from utils import Disabled

PREF_TASK_LIST = {
    "ant": [
        "n", "s", "e", "w", 
        "ne", "nw", "se", "sw",
        "new_n", "new_s", "new_e", "new_w",
        "range", "range_diamond", 
        "hole", "hole2", 
        "1_hole_2_new_n", "1_range_2_new_n", "1_ne_2_new_n", "1_range_2_range", 
        "not_roll", 
    ],
    "half_cheetah": ["l", "r", "range", "not_flip"],
    "dmc_quadruped": [
        "n", "s", "e", "w",
        "ne", "nw", "se", "sw",
        "new_n", "new_s", "new_e", "new_w",
        "range", "hole", "hole2"
    ],
    "dmc_humanoid": [
        "n", "s", "e", "w",
        "ne", "nw", "se", "sw",
        "new_n", "new_s", "new_e", "new_w",
        "range", "hole", "hole2"
    ],
    "quadruped": [
        "n", "s", "e", "w",
        "ne", "nw", "se", "sw",
        "new_n", "new_s", "new_e", "new_w",
        "range", "hole", "hole2"
    ],
    "humanoid": [
        "n", "s", "e", "w",
        "ne", "nw", "se", "sw",
        "new_n", "new_s", "new_e", "new_w",
        "range", "hole", "hole2"
    ],
    "CSafe": ["hazard"]
}


SAFE_CENTER_ANT = np.array([0, 5])
SAFE_R_ANT = 20
SAFE_CENTER_CHEETAH = np.array([5, ])
SAFE_R_CHEETAH = 20
SAFE_CENTER_DMC = np.array([0, 2])  # 可根据实际环境调整
SAFE_R_DMC = 8

HOLE_CENTERS_ANT = {
    0: [
        np.array([15, 15]),
        np.array([15, -15]),
        np.array([-15, 15]),
        np.array([-15, -15])
    ],
    1: [
        np.array([20, 0]),
        np.array([-20, 0]),
        np.array([0, 20]),
        np.array([0, -20])
    ]
}
HOLE_R_ANT = 12
HOLE_CENTERS_DMC = {
    0: [
        np.array([6, 6]),
        np.array([6, -6]),
        np.array([-6, 6]),
        np.array([-6, -6])
    ],
    1: [
        np.array([8, 0]),
        np.array([-8, 0]),
        np.array([0, 8]),
        np.array([0, -8])
    ]
}
HOLE_R_DMC = 4

DIAMOND_CENTER = np.array([0, 5])  # 菱形中心
DIAMOND_A = 25  # x轴方向的半轴长度
DIAMOND_B = 10  # y轴方向的半轴长度

# for CSafe tasks
HAZARD_IDX = -3
SAFE_THRESHOLD = 0.9


DIRECTION_TASKS = {
    "n": {"coord_indices": [1], "direction_muls": [1]},  # y > 0
    "s": {"coord_indices": [1], "direction_muls": [-1]},  # y < 0
    "e": {"coord_indices": [0], "direction_muls": [1]},  # x > 0
    "w": {"coord_indices": [0], "direction_muls": [-1]},  # x < 0
    "ne": {"coord_indices": [0, 1], "direction_muls": [1, 1]}, # x > 0 and y > 0
    "nw": {"coord_indices": [0, 1], "direction_muls": [-1, 1]},  # x < 0 and y > 0
    "se": {"coord_indices": [0, 1], "direction_muls": [1, -1]},  # x > 0 and y < 0
    "sw": {"coord_indices": [0, 1], "direction_muls": [-1, -1]},  # x < 0 and y < 0
    "new_n": {"coord_indices": [0, 1], "special_func": lambda x, y: y > np.abs(x)},  # y > |x|
    "new_s": {"coord_indices": [0, 1], "special_func": lambda x, y: y < -np.abs(x)},  # y < -|x|
    "new_e": {"coord_indices": [0, 1], "special_func": lambda x, y: x > np.abs(y)},  # x > |y|
    "new_w": {"coord_indices": [0, 1], "special_func": lambda x, y: x < -np.abs(y)},  # x < -|y|
}


class OracleStatePref:
    def __init__(self, env_name, pref_task):
        if env_name in ["ant_pref_goal", "ant_pref_goal_zs"]:
            env_name = 'ant'
        if env_name in ["half_cheetah_goal_notflip", "half_cheetah_goal_notflip_zs"]:
            env_name = 'half_cheetah'
        self.env_name = env_name
        self.pref_task = pref_task
        self.pref_func_dict = {
            "ant": {
                None: self._gene_get_state_pref_2d_direction("n"),
                "n": self._gene_get_state_pref_2d_direction("n"),
                "s": self._gene_get_state_pref_2d_direction("s"),
                "e": self._gene_get_state_pref_2d_direction("e"),
                "w": self._gene_get_state_pref_2d_direction("w"),
                "ne": self._gene_get_state_pref_2d_direction("ne"),
                "nw": self._gene_get_state_pref_2d_direction("nw"),
                "se": self._gene_get_state_pref_2d_direction("se"),
                "sw": self._gene_get_state_pref_2d_direction("sw"),
                "new_n": self._gene_get_state_pref_2d_direction("new_n"),
                "new_s": self._gene_get_state_pref_2d_direction("new_s"),
                "new_e": self._gene_get_state_pref_2d_direction("new_e"),
                "new_w": self._gene_get_state_pref_2d_direction("new_w"),
                "range": self._gene_get_state_pref_range("ant"),
                "range_diamond": self._get_state_pref_ant_range_diamond,
                "hole": self._gene_get_state_pref_ant_hole(0, "ant"),
                "hole2": self._gene_get_state_pref_ant_hole(1, "ant"),
                "not_roll": self._get_state_pref_ant_not_roll,
                "1_hole_2_new_n": self._gene_get_state_pref_constraint_intent(
                    self._gene_get_state_pref_ant_hole(0, "ant"),
                    self._gene_get_state_pref_2d_direction("new_n")
                ),
                "1_range_2_new_n": self._gene_get_state_pref_constraint_intent(
                    self._gene_get_state_pref_range("ant"),
                    self._gene_get_state_pref_2d_direction("new_n")
                ),
                "1_ne_2_new_n": self._gene_get_state_pref_constraint_intent(
                    self._gene_get_state_pref_2d_direction("ne"),
                    self._gene_get_state_pref_2d_direction("new_n")
                ),
                "1_range_2_range": self._gene_get_state_pref_constraint_intent(
                    self._gene_get_state_pref_range("ant"),
                    self._gene_get_state_pref_range("ant")
                ),
            },
            'half_cheetah': {
                None: self._gene_get_state_pref_half_cheetah_direction("l"),
                "l": self._gene_get_state_pref_half_cheetah_direction("l"),
                "r": self._gene_get_state_pref_half_cheetah_direction("r"),
                "range": self._get_state_pref_half_cheetah_range,
                "not_flip": self._get_state_pref_half_cheetah_not_flip, 
            },
            'dmc_quadruped': {
                None: self._gene_get_state_pref_2d_direction("n"),
                "n": self._gene_get_state_pref_2d_direction("n"),
                "s": self._gene_get_state_pref_2d_direction("s"),
                "e": self._gene_get_state_pref_2d_direction("e"),
                "w": self._gene_get_state_pref_2d_direction("w"),
                "ne": self._gene_get_state_pref_2d_direction("ne"),
                "nw": self._gene_get_state_pref_2d_direction("nw"),
                "se": self._gene_get_state_pref_2d_direction("se"),
                "sw": self._gene_get_state_pref_2d_direction("sw"),
                "new_n": self._gene_get_state_pref_2d_direction("new_n"),
                "new_s": self._gene_get_state_pref_2d_direction("new_s"),
                "new_e": self._gene_get_state_pref_2d_direction("new_e"),
                "new_w": self._gene_get_state_pref_2d_direction("new_w"),
                "range": self._gene_get_state_pref_range("dmc"),
                "hole": self._gene_get_state_pref_ant_hole(0, "dmc"),
                "hole2": self._gene_get_state_pref_ant_hole(1, "dmc"),
            }, 
            'dmc_humanoid': {
                None: self._gene_get_state_pref_2d_direction("n"),
                "n": self._gene_get_state_pref_2d_direction("n"),
                "s": self._gene_get_state_pref_2d_direction("s"),
                "e": self._gene_get_state_pref_2d_direction("e"),
                "w": self._gene_get_state_pref_2d_direction("w"),
                "ne": self._gene_get_state_pref_2d_direction("ne"),
                "nw": self._gene_get_state_pref_2d_direction("nw"),
                "se": self._gene_get_state_pref_2d_direction("se"),
                "sw": self._gene_get_state_pref_2d_direction("sw"),
                "new_n": self._gene_get_state_pref_2d_direction("new_n"),
                "new_s": self._gene_get_state_pref_2d_direction("new_s"),
                "new_e": self._gene_get_state_pref_2d_direction("new_e"),
                "new_w": self._gene_get_state_pref_2d_direction("new_w"),
                "range": self._gene_get_state_pref_range("dmc"),
                "hole": self._gene_get_state_pref_ant_hole(0, "dmc"),
                "hole2": self._gene_get_state_pref_ant_hole(1, "dmc"),
            }, 
            "maze": {
                None: self._gene_get_state_pref_maze_direction("n"),
                "n": self._gene_get_state_pref_maze_direction("n"),
                "s": self._gene_get_state_pref_maze_direction("s"),
                "e": self._gene_get_state_pref_maze_direction("e"),
                "w": self._gene_get_state_pref_maze_direction("w"),
            },
            "CSafe": {
                None: self._get_pref_csafe_hazard,
                "hazard": self._get_pref_csafe_hazard,
            }
        }

        if env_name in self.pref_func_dict:
            self.get_state_pref = self.pref_func_dict[env_name][pref_task]
        elif env_name.startswith("CSafe"):
            self.get_state_pref = self.pref_func_dict["CSafe"][pref_task]
        else:
            raise Exception(f"oracle pref for {env_name} is not implemented")

    def get_state_pref(self, states):
        raise NotImplementedError()

    @Disabled
    def _old_gene_get_state_pref_ant_direction(self, direction):
        direction_idx = ["n", "s", "e", "w"].index(direction)
        coord_idx = [1, 1, 0, 0][direction_idx]
        direction_mul = [1, -1, 1, -1][direction_idx]

        def _get_state_pref_ant_direction(states):
            # foundation model guided skill discovery method, FoG.
            # len(states) = 8, states is list, states[0].shape = (200, 29)
            prefs = np.zeros((len(states), len(states[0])))  # (8, 200)
            for n_batch in range(len(states)):
                coords = states[n_batch][:, coord_idx]
                right_movement = np.where(coords * direction_mul >= 0, 1, 0)
                prefs[n_batch, :] = right_movement
            return prefs
        return _get_state_pref_ant_direction

    def _gene_get_state_pref_2d_direction(self, direction):
        if direction in DIRECTION_TASKS:
            task_config = DIRECTION_TASKS[direction]
            
            # 如果有特殊函数，使用特殊函数
            if "special_func" in task_config:
                def _get_state_pref_direction(states):
                    prefs = np.zeros((len(states), len(states[0])))
                    for n_batch in range(len(states)):
                        xs = states[n_batch][:, 0]
                        ys = states[n_batch][:, 1]
                        prefs[n_batch, :] = task_config["special_func"](xs, ys).astype(np.float32)
                    return prefs
                return _get_state_pref_direction
            
            # 否则使用通用的坐标和方向乘法
            coord_indices = task_config["coord_indices"]
            direction_muls = task_config["direction_muls"]
            
            def _get_state_pref_direction(states):
                prefs = np.zeros((len(states), len(states[0])))
                for n_batch in range(len(states)):
                    condition = np.ones(len(states[n_batch]), dtype=bool)
                    for coord_idx, direction_mul in zip(coord_indices, direction_muls):
                        coords = states[n_batch][:, coord_idx]
                        condition &= (coords * direction_mul >= 0)
                    prefs[n_batch, :] = condition.astype(np.float32)
                return prefs
            return _get_state_pref_direction
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def _gene_get_state_pref_constraint_intent(self, constraint_func, intent_func):
        def _get_state_pref_constraint_intent(states):
            constraint_prefs = constraint_func(states)
            intent_prefs = intent_func(states)
            # 如果 constraint_prefs 为 0，则应该是 0
            # 如果 constraint_prefs 为 1，则对于 intent_prefs 为 1 的取 2，否则取 1
            prefs = constraint_prefs + intent_prefs
            prefs = np.where(constraint_prefs == 0, 0, prefs)
            prefs = np.where(prefs >= 2, 2, prefs)
            return prefs
        return _get_state_pref_constraint_intent

    def _gene_get_state_pref_range(self, task):
        # restrict the max range to reach, which is an ellipse
        # states: list[traj]
        safe_center = SAFE_CENTER_ANT if task == 'ant' else SAFE_CENTER_DMC
        safe_r = SAFE_R_ANT if task == 'ant' else SAFE_R_DMC

        def _get_state_pref_range(states):
            prefs = np.zeros((len(states), len(states[0])))
            for n_batch in range(len(states)):
                loc = states[n_batch][:, :2] # (x,y)
                dists_sq = np.sum((loc - safe_center[None, :])**2, axis=1)
                in_circle = (dists_sq <= safe_r**2).astype(np.float32)
                prefs[n_batch, :] = in_circle
            return prefs
        return _get_state_pref_range

    def _get_state_pref_ant_range_diamond(self, states):
        # restrict the max range to reach, which is a diamond
        prefs = np.zeros((len(states), len(states[0])))
        for n_batch in range(len(states)):
            loc = states[n_batch][:, :2]  # (x,y)
            # 计算相对于菱形中心的坐标
            rel_loc = loc - DIAMOND_CENTER[None, :]
            # 检查是否在菱形内: |x|/a + |y|/b <= 1
            in_diamond = (np.abs(rel_loc[:, 0]) / DIAMOND_A + 
                          np.abs(rel_loc[:, 1]) / DIAMOND_B <= 1)
            prefs[n_batch, :] = in_diamond.astype(np.float32)
        return prefs

    def _gene_get_state_pref_ant_hole(self, hole_task_idx, task):
        # punish the agent for fall into a hole, which is an ellipse
        hole_centers = HOLE_CENTERS_ANT if task == 'ant' else HOLE_CENTERS_DMC
        hole_r = HOLE_R_ANT if task == 'ant' else HOLE_R_DMC

        def _get_state_pref_ant_hole(states):
            prefs = np.ones((len(states), len(states[0])))  # 初始化为1（安全）
            for n_batch in range(len(states)):
                loc = states[n_batch][:, :2]  # (x,y)
                for center in hole_centers[hole_task_idx]:
                    dists_sq = np.sum((loc - center[None, :])**2, axis=1)
                    in_hole = (dists_sq <= hole_r**2)
                    prefs[n_batch, in_hole] = 0
            return prefs
        return _get_state_pref_ant_hole

    def _gene_get_state_pref_half_cheetah_direction(self, direction):
        direction_idx = ["l", "r"].index(direction)
        direction_mul = [1, -1][direction_idx]

        def _get_state_pref_half_cheetah_direction(states):
            # foundation model guided skill discovery method, FoG.
            prefs = np.zeros((len(states), len(states[0])))
            for n_batch in range(len(states)):
                coords = states[n_batch][:, 0]
                right_movement = np.where(coords * direction_mul >= 0, 1, 0)
                prefs[n_batch, :] = right_movement
            return prefs
        return _get_state_pref_half_cheetah_direction

    def _get_state_pref_half_cheetah_range(self, states):
        # restrict the max range to reach, which is an ellipse
        prefs = np.zeros((len(states), len(states[0])))
        for n_batch in range(len(states)):
            loc = states[n_batch][:, :1] # (x, )
            dists_sq = np.sum((loc - SAFE_CENTER_CHEETAH[None, :])**2, axis=1)
            in_circle = (dists_sq <= SAFE_R_CHEETAH**2).astype(np.float32)
            prefs[n_batch, :] = in_circle
        return prefs

    def _get_state_pref_ant_not_roll(self, states):
        prefs = np.ones((len(states), len(states[0])))
        for n_batch in range(len(states)):
            batch_states = states[n_batch]
            # 提取四元数，计算机体Z轴在世界坐标系中的Z分量
            w, x, y, z = batch_states[:, 3], batch_states[:, 4], batch_states[:, 5], batch_states[:, 6]
            up_z = 1 - 2 * (x**2 + y**2)
            prefs[n_batch, :] = np.where(up_z < 0, 0, 1).astype(np.float32)
        return prefs

    def _get_state_pref_half_cheetah_not_flip(self, states):
        pitch_idx = 2
        prefs = np.ones((len(states), len(states[0])))
        for n_batch in range(len(states)):
            pitch = states[n_batch][:, pitch_idx]
            prefs[n_batch, :] = np.where(np.abs(pitch) > 1.57, 0, 1).astype(np.float32)
        return prefs

    @Disabled
    def _gene_get_state_pref_dmc_direction_pixel(self, direction):
        direction_idx = ["n", "s", "e", "w"].index(direction)
        coord_idx = [0, 0, 1, 1][direction_idx]
        direction_mul = [-1, 1, 1, -1][direction_idx]

        def _get_state_pref_dmc_direction(states):
            prefs = np.zeros((len(states), len(states[0])))
            for n_batch in range(len(states)):  # len(states) = 8
                # states[n_batch].shape = (batch_size, frame_stacks, 64, 64, 3) for dmc tasks
                # 希望得到 64*64 图像的边缘一圈像素，并取平均值，得到 colors (shape 为 (batch_size, 3))
                batch_size = states[n_batch].shape[0]
                pixel_states = states[n_batch].reshape(batch_size, -1, 64, 64, 3)
                pixel_states = np.mean(pixel_states, axis=1)
                top_edge = pixel_states[:, 20, :, :]  # (200, 64, 3)
                bottom_edge = pixel_states[:, -15, :, :]
                left_edge = pixel_states[:, :, 20, :]
                right_edge = pixel_states[:, :, -20, :]
                all_edges = np.concatenate((top_edge, bottom_edge, left_edge, right_edge), axis=1)
                colors = np.mean(all_edges, axis=1)  # (batch_size, 3)

                # pixel_wrappers.py:
                # env.physics.model.tex_rgb[cur_s:cur_s + 3] = [int(x / height * 255), int(y / width * 255), 128]
                channel = np.where((colors[:, coord_idx] - colors[:, 2]) * direction_mul + 10 >= 0, 1, 0)
                # 考察平均颜色中的红色 / 绿色与蓝色的差值, +10 是为了更 robust
                prefs[n_batch, :] = channel
            return prefs
        return _get_state_pref_dmc_direction

    def _gene_get_state_pref_maze_direction(self, direction):
        direction_idx = ["n", "s", "e", "w"].index(direction)
        coord_idx = [1, 1, 0, 0][direction_idx]
        direction_mul = [1, -1, 1, -1][direction_idx]

        def _get_state_pref_maze_direction(states):
            # foundation model guided skill discovery method, FoG.
            prefs = np.zeros((len(states), len(states[0])))
            for n_batch in range(len(states)):
                coords = states[n_batch][:, coord_idx]
                right_movement = np.where(coords * direction_mul >= 0, 1, 0)
                prefs[n_batch, :] = right_movement
            return prefs
        return _get_state_pref_maze_direction
    
    def _get_pref_csafe_hazard(self, ori_states):
        prefs = np.zeros((len(ori_states), len(ori_states[0])))
        for n_batch in range(len(ori_states)):
            in_hazard_raw = ori_states[n_batch][:, HAZARD_IDX]
            in_hazard = np.where(in_hazard_raw > 1e-5, 1, 0)
            prefs[n_batch, :] = in_hazard
        return prefs

    def get_coordinates_quality(self, coordinates):
        ''' should return 0 (bad), 1 (neutral), or 2 (good) '''
        if self.env_name in ['ant', 'dmc_quadruped', 'dmc_humanoid']:
            safe_center = SAFE_CENTER_ANT if self.env_name == 'ant' else SAFE_CENTER_DMC
            safe_r = SAFE_R_ANT if self.env_name == 'ant' else SAFE_R_DMC
            hole_centers = HOLE_CENTERS_ANT if self.env_name == 'ant' else HOLE_CENTERS_DMC
            hole_r = HOLE_R_ANT if self.env_name == 'ant' else HOLE_R_DMC

            if self.pref_task in DIRECTION_TASKS:
                task_config = DIRECTION_TASKS[self.pref_task]
                if "special_func" in task_config:
                    xs = coordinates[:, 0]
                    ys = coordinates[:, 1]
                    quality = task_config["special_func"](xs, ys).astype(np.float32)
                else:
                    coord_indices = task_config["coord_indices"]
                    direction_muls = task_config["direction_muls"]
                    condition = np.ones(len(coordinates), dtype=bool)
                    for coord_idx, direction_mul in zip(coord_indices, direction_muls):
                        coords = coordinates[:, coord_idx]
                        condition &= (coords * direction_mul >= 0)
                    quality = condition.astype(np.float32)
            elif self.pref_task == 'range':
                dists_sq = np.sum((coordinates - safe_center[None, :])**2, axis=1)
                quality = (dists_sq <= safe_r**2).astype(np.float32)
            elif self.pref_task == 'range_diamond':  # 添加菱形区域处理逻辑
                rel_coords = coordinates - DIAMOND_CENTER[None, :]
                quality = (np.abs(rel_coords[:, 0]) / DIAMOND_A + 
                           np.abs(rel_coords[:, 1]) / DIAMOND_B <= 1).astype(np.float32)
            elif self.pref_task in ['hole', 'hole2']:
                hold_task_idx = 0 if self.pref_task == 'hole' else 1  # hole2
                quality = np.ones(len(coordinates))
                for center in hole_centers[hold_task_idx]:
                    dists_sq = np.sum((coordinates - center[None, :])**2, axis=1)
                    in_hole = (dists_sq <= hole_r**2)
                    quality[in_hole] = 0
            elif self.pref_task == '1_hole_2_new_n':
                # 先判断是否在 hole 里
                hold_task_idx = 0
                quality = np.ones(len(coordinates))
                for center in hole_centers[hold_task_idx]:
                    dists_sq = np.sum((coordinates - center[None, :])**2, axis=1)
                    in_hole = (dists_sq <= hole_r**2)
                    quality[in_hole] = 0
                # 然后判断是否在 new_n 区域
                xs = coordinates[:, 0]
                ys = coordinates[:, 1]
                in_new_n = DIRECTION_TASKS["new_n"]["special_func"](xs, ys)
                quality = np.where(quality == 1, np.where(in_new_n, 2, 1), quality)
            elif self.pref_task == '1_range_2_new_n':
                # 先判断是否在 range 里
                dists_sq = np.sum((coordinates - safe_center[None, :])**2, axis=1)
                in_range = (dists_sq <= safe_r**2)
                quality = np.where(in_range, 1, 0)
                # 然后判断是否在 new_n 区域
                xs = coordinates[:, 0]
                ys = coordinates[:, 1]
                in_new_n = DIRECTION_TASKS["new_n"]["special_func"](xs, ys)
                quality = np.where(quality == 1, np.where(in_new_n, 2, 1), quality)
            elif self.pref_task == '1_ne_2_new_n':
                # 先判断是否在 ne 区域
                xs = coordinates[:, 0]
                ys = coordinates[:, 1]
                in_ne = (xs >= 0) & (ys >= 0)
                quality = np.where(in_ne, 1, 0)
                # 然后判断是否在 new_n 区域
                in_new_n = DIRECTION_TASKS["new_n"]["special_func"](xs, ys)
                quality = np.where(quality == 1, np.where(in_new_n, 2, 1), quality)
            elif self.pref_task == '1_range_2_range':
                # 判断是否在 range 里
                dists_sq = np.sum((coordinates - safe_center[None, :])**2, axis=1)
                in_range = (dists_sq <= safe_r**2)
                quality = np.where(in_range, 2, 0)
            else:
                raise NotImplementedError(f"Unknown pref_task: {self.pref_task} for ant")

        elif self.env_name == 'half_cheetah':
            if self.pref_task in ['l', 'r']:
                direction_idx = ["l", "r"].index(self.pref_task)
                direction_mul = [1, -1][direction_idx]
                quality = np.where(coordinates[:, 0] * direction_mul >= 0, 1, 0)
            elif self.pref_task == 'range':
                dists_sq = np.sum((coordinates - SAFE_CENTER_CHEETAH[None, :])**2, axis=1)
                quality = (dists_sq <= SAFE_R_CHEETAH**2).astype(np.float32)
            else:
                raise NotImplementedError(f"Unknown pref_task: {self.pref_task} for half_cheetah")

        else:
            raise NotImplementedError(f"Unknown or unimplemented env_name: {self.env_name}")
        return quality

    def get_csafe_coordinates_quality(self, uniq_coords_idx, ori_obs):
        uniq_ori_obs = ori_obs[uniq_coords_idx]
        if self.pref_task == "hazard":
            quality = np.where(uniq_ori_obs[HAZARD_IDX] > 0, 1, 0)
        return quality

    def get_state_safe_quality(self, states, all_coords):
        """
        根据状态判断质量（0: 坏, 1: 中性, 2: 好）
        输入: states - 状态列表，每个元素是一个轨迹的状态数组
        输出: 质量数组，形状与states相同
        """
        if self.env_name == 'ant' and self.pref_task == 'not_roll':
            w, x, y, z = states[:, 3], states[:, 4], states[:, 5], states[:, 6]
            up_z = 1 - 2 * (x**2 + y**2)
            # 如果up_z小于0，表示机体翻转（背部着地），质量为0
            # quality = (up_z >= 0).astype(np.float32)
            quality_neutral_index = up_z >= 0.0  # 质量为2的条件
            quality_bad_index = up_z < 0.0  # 质量为0的条件

        elif self.env_name == 'half_cheetah' and self.pref_task == 'not_flip':
            pitch_idx = 2
            pitch = states[:, pitch_idx]
            # 如果俯仰角的绝对值超过90度（约1.57弧度），则认为翻转，质量为0
            quality_neutral_index = np.abs(pitch) <= 1.57
            quality_bad_index = np.abs(pitch) > 1.57

        neutral_coords = all_coords[quality_neutral_index]
        bad_coords = all_coords[quality_bad_index]
        unique_coords = np.unique(np.floor(all_coords), axis=0)
        uniq_neutral_coords = np.unique(np.floor(neutral_coords), axis=0)
        uniq_bad_coords = np.unique(np.floor(bad_coords), axis=0)

        # 计算每个唯一坐标点的中性比例
        uniq_neutral_ratios = []

        for coord in unique_coords:
            # 找到该坐标点对应的所有样本
            mask = (np.floor(all_coords) == coord).all(axis=1)
            total_count = np.sum(mask)

            if total_count > 0:
                # 计算该坐标点中中性状态的比例
                neutral_count = np.sum(quality_neutral_index[mask])
                neutral_ratio = neutral_count / total_count
                uniq_neutral_ratios.append(neutral_ratio)
            else:
                uniq_neutral_ratios.append(0.0)

        uniq_neutral_ratios = np.array(uniq_neutral_ratios)

        # 根据阈值分类坐标点
        uniq_ratio_neutral_coords = unique_coords[uniq_neutral_ratios >= SAFE_THRESHOLD]
        uniq_ratio_bad_coords = unique_coords[uniq_neutral_ratios < SAFE_THRESHOLD]

        return unique_coords, uniq_neutral_coords, uniq_bad_coords, \
            uniq_ratio_neutral_coords, uniq_ratio_bad_coords

    def calc_eval_metrics(self, trajectories):
        '''
        Use this function in _evaluate_policy() of metra.py
        Please refer to calc_eval_metrics() in 
          - mujoco/mujoco_utils.py
          - custom_dmc_tasks/wrappers.py
        '''
        eval_metrics = {}

        if self.env_name == 'ant':
            coord_dims = [0, 1]  # x, y coordinates
        elif self.env_name == 'half_cheetah':
            coord_dims = [0]
        elif self.env_name in ['dmc_quadruped', 'dmc_humanoid'] or "CSafe" in self.env_name:
            coord_dims = 2  # x, y coordinates
        else:
            coord_dims = 1

        if "CSafe" not in self.env_name:
            coords = []
            for traj in trajectories:
                if self.env_name in ['ant', 'half_cheetah']:
                    traj1 = traj['env_infos']['coordinates'][:, coord_dims]
                    traj2 = traj['env_infos']['next_coordinates'][-1:, coord_dims]
                elif self.env_name in ['dmc_quadruped', 'dmc_humanoid']:
                    traj1 = traj['env_infos']['coordinates']
                    traj2 = traj['env_infos']['next_coordinates'][-1:, :]
                coords.append(traj1)
                coords.append(traj2)
            coords = np.concatenate(coords, axis=0)
        else:
            coords = []
            ori_obss = []
            for traj in trajectories:
                traj1 = traj['env_infos']['coordinates']
                coords.append(traj1)
                ori_obss.append(traj['env_infos']['ori_obs'])
            coords = np.concatenate(coords, axis=0)
            ori_obss = np.concatenate(ori_obss, axis=0)

        if self.pref_task in ['not_roll', 'not_flip']:
            all_states = []
            for traj in trajectories:
                states = traj['env_infos']['ori_obs']
                next_states = traj['env_infos']['next_ori_obs'][-1:, :]
                traj1 = traj['env_infos']['coordinates'][:, coord_dims]  # self.env_name in ['ant', 'half_cheetah']
                traj2 = traj['env_infos']['next_coordinates'][-1:, coord_dims]
                all_states.append(states)
                all_states.append(next_states)
            all_states = np.concatenate(all_states, axis=0)
            unique_coords, uniq_neutral_coords, uniq_bad_coords, uniq_ratio_neutral_coords, uniq_ratio_bad_coords = \
                self.get_state_safe_quality(all_states, all_coords=coords)
            
            eval_metrics.update({
                'MjNumGoodCoords': 0.0,
                'MjNumNeutralCoords': len(uniq_neutral_coords),
                'MjNumBadCoords': len(uniq_bad_coords),
                'MjGoodCoordsRatio': 0.0,
                'MjNeutralCoordsRatio': len(uniq_neutral_coords) / len(unique_coords),
                'MjBadCoordsRatio': len(uniq_bad_coords) / len(unique_coords),
                # real safe coverage / safe ratio
                'MjNumNeutralCoords_real': len(uniq_ratio_neutral_coords),
                'MjNumBadCoords_real': len(uniq_ratio_bad_coords),
                'MjNeutralCoordsRatio_real': len(uniq_ratio_neutral_coords) / len(unique_coords),
                'MjBadCoordsRatio_real': len(uniq_ratio_bad_coords) / len(unique_coords),
            })
            return eval_metrics

        elif not "CSafe" in self.env_name:
            uniq_coords = np.unique(np.floor(coords), axis=0)
            coords_quality = self.get_coordinates_quality(uniq_coords)
        else:
            # NOTE: the range agent can reach is small in CSafe tasks, so we increase the precision of grid division
            uniq_coords, uniq_coords_idx = np.unique(np.floor(coords * 100), axis=0, return_index=True)
            coords_quality = self.get_csafe_coordinates_quality(uniq_coords_idx, ori_obss)

        eval_metrics.update({
            # 'MjNumTrajs': len(trajectories),
            # 'MjAvgTrajLen': len(coords) / len(trajectories) - 1,
            # 'MjNumCoords': len(coords),
            'MjNumGoodCoords': np.sum(coords_quality == 2),
            'MjNumNeutralCoords': np.sum(coords_quality == 1),
            'MjNumBadCoords': np.sum(coords_quality == 0),
            'MjGoodCoordsRatio': np.sum(coords_quality == 2) / len(coords_quality),
            'MjNeutralCoordsRatio': np.sum(coords_quality == 1) / len(coords_quality),
            'MjBadCoordsRatio': np.sum(coords_quality == 0) / len(coords_quality),
        })

        return eval_metrics


