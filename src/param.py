"""集中调参入口。

这里放策略、走位、踢球和避障的可调参数。数值迁移自当前代码文件,
以后优先在本文件调整参数。
"""

from __future__ import annotations

import math


# ======================================================================
# 踢球力度
# ======================================================================

# Player.kick() 会把力度夹到这个范围内。如果超过范围，因为控制接口不支持，可能会摔倒。
KICK_POWER_MIN = 1.0
KICK_POWER_MAX = 10.0

# 普通比赛踢球力度。
KICK_POWER_DEFAULT = 5.0
KICK_POWER_BACKFIELD = 5.0
KICK_POWER_OUR_KICKOFF = 5.0


# ======================================================================
# Player 走位控制
# ======================================================================

ARRIVE_DIST = 0.15             # 到达目标点阈值 (m)
OMNI_DIST = 1.0                # 距目标小于该值时使用全向近距控制
TURN_THRESHOLD = 0.5           # 远距离行走时,角差超过该值先原地转向 (rad)
MAX_LINEAR = 2.0               # 恢复官方基线指令上限；不增加冲刺能力
MAX_ANGULAR = 2.0              # 指令转向速度上限 (rad/s)
LINEAR_GAIN = 1.5              # 平移 P 增益
ANGULAR_GAIN = 2.0             # 转向 P 增益


# ======================================================================
# Player 踢球 / 射门规划
# ======================================================================

KICK_ENTER_M = 2.0             # 距球小于该值进入踢球状态
KICK_EXIT_M = 2.5              # 踢球中距球大于该值才退出踢球状态
CHASE_BEHIND_M = 0.35          # 追球时站到球后方的距离


# ======================================================================
# Player 技术动作参数：守门 / 支援 
# ======================================================================

GUARD_FACE_BALL = True
GUARD_THREAT_ENTER_X = -1.0
GUARD_THREAT_EXIT_X = -0.7

SUPPORT_DIST_M = 3.6

# ======================================================================
# Normal 阶段策略
# ======================================================================

ATTACKER_KEEP_DIST_MARGIN_M = 0.3  # 防止 Attacker 选择产生震荡

FALLEN_COST = 10.0  # 摔倒球员的距离惩罚值(米)


# ======================================================================
# 开球 / 定位球策略
# ======================================================================

# 开球
KICKOFF_STAGE_M = 2.0
KICKOFF_FRONT_MARGIN = 0.1
KICKOFF_LATERAL_TOL = 0.35

CENTER_LEAVE_DIST_M = 0.15 # 球离开中心点多少距离，认为球已经动了

OPP_SET_WALL_DIST_M = 2.0 # 对方开球时，在多少距离之后阻挡


# ======================================================================
# 站位 / 避让
# ======================================================================

OPPONENT_RESTART_AVOID_M = 1.9
CIRCLE_MARGIN_M = 0.3

# 合规工程余量，不是裁判阈值。全部定位球采用新版 G-03 的 0.5 m 条件。
FIELD_SAFE_MARGIN_M = 0.4
HALF_SAFE_MARGIN_M = 0.3
RESTART_SHOT_DISTANCE_M = 0.85
RESTART_PUSH_POWER = 1.0
RESTART_PUSH_LENGTH_M = 1.2
RESTART_GOAL_MARGIN_M = 0.5
WAIT_BALL_DISTANCE_M = 3.6
RULE_TARGET_MARGIN_M = 0.25
RULE_LOOKAHEAD_S = 0.6
RULE_ESCAPE_SPEED_MPS = 0.6
RULE_ANGLE_SAMPLES = 72
INACTIVITY_RECOVERY_S = 4.0
INACTIVITY_PROGRESS_M = 0.25
INACTIVITY_BALL_DISTANCE_M = 3.2
KICKOFF_TAKER_LINE_MARGIN_M = 0.1
GET_UP_RETRY_S = 2.0           # 只重试官方状态仍允许恢复的起身，不重复提交运行中任务


# ======================================================================
# 踢球目标几何
# ======================================================================

GOAL_TARGET_DEPTH_M = 0.25              # 踢球目标点在球门内的深度 (m)

# ======================================================================
# 障碍物几何参数
# ======================================================================

BALL_OBSTACLE_RADIUS = 0.5              # 球的障碍半径 (m)；用于避障计算
OPPONENT_RADIUS = 0.55                  # 对手机器人半径 (m)
TEAMMATE_RADIUS = 0.48                  # 队友机器人半径 (m)
SAFETY_MARGIN = 0.22                    # 通用安全余量 (m)

GOAL_DEPTH = 0.6                        # 球门深度 (m)；用于球门障碍建模
POST_RADIUS = 0.18                      # 门柱半径 (m)
NET_RADIUS = 0.20                       # 球网半径 (m)
NET_STEP = 0.35                         # 球网离散化步长 (m)

START_IGNORE = 0.0                      # 起点忽略半径 (m)；起点附近障碍不参与规划
TARGET_IGNORE = 0.0                     # 终点忽略半径 (m)；终点附近障碍不参与规划

# ======================================================================
# 全局路径规划器 (A* Grid Planner)
# ======================================================================

USE_GLOBAL_PATH_PLANNER = True          # 是否启用全局规划器；False 时回退到局部 VFH
GLOBAL_GRID_RESOLUTION_M = 0.35         # 栅格分辨率 (m/格)；越小路径越精细，但计算量越大
GLOBAL_FIELD_MARGIN_M = 0.25            # 场地边界外扩余量 (m)；确保边界附近路径可行
GLOBAL_OBSTACLE_MARGIN_M = 0.10         # 障碍物膨胀余量 (m)；在障碍半径上额外扩大多少
GLOBAL_PATH_LOOKAHEAD_M = 0.9           # 从规划路径中提取前方多远的路径点 (m)

# ======================================================================
# 局部路径规划器 (VFH Direction Scan)
# ======================================================================

PLAN_LOOKAHEAD = 1.2                    # 前方探测射线长度 (m)；规划器"看"的范围
PLAN_CLEARANCE = 0.35                   # 最小安全余量 (m)；候选方向前方必须大于此值
PLAN_STEP = math.radians(15)            # 候选方向扫描步长 (rad)；越小方向越精细
PLAN_MAX_OFFSET = math.radians(100)     # 最大偏离目标方向的角度 (rad)；扫描范围 ± 该值

# ======================================================================
# 可视化
# ======================================================================

KICK_TARGET_MARK_SIZE_M = 0.18

# ======================================================================
# 球路观测与预判（初值是估计，运行时用无碰撞的官方位置样本校准）
# ======================================================================
BALL_VELOCITY_WINDOW_S = 0.18
BALL_SAMPLE_MIN_DT_S = 0.025
BALL_SAMPLE_MAX_GAP_S = 0.4
BALL_MAX_OBSERVED_SPEED_MPS = 14.0
BALL_TELEPORT_MARGIN_M = 0.45
BALL_STOP_SPEED_MPS = 0.12
BALL_VELOCITY_FILTER = 0.55
BALL_DECEL_INITIAL_MPS2 = 0.55
BALL_DECEL_MIN_MPS2 = 0.08
BALL_DECEL_MAX_MPS2 = 3.0
BALL_DECEL_FILTER = 0.08
BALL_FREE_ROLL_CLEARANCE_M = 1.0
BALL_ROLL_MIN_SPEED_MPS = 0.5
BALL_ROLL_MAX_TURN_RAD = math.radians(12)
BALL_PREDICTION_MAX_S = 2.0
BALL_PREDICTION_STEP_S = 0.1
BALL_PREDICTION_BOUNDARY_M = 0.08
BALL_WALK_SPEED_MPS = 0.9  # 走路速度估计，不假设可冲刺
BALL_INTERCEPT_REACH_M = 0.45
BALL_CALIBRATION_LOG_S = 10.0
BALL_KICK_SAMPLE_WINDOW_S = 3.0
BALL_KICK_SAMPLE_MIN_SPEED_MPS = 0.8
BALL_KICK_DIRECTION_TOL_RAD = math.radians(25)
BALL_KICK_SPEED_INITIAL_MPS = 0.6
BALL_KICK_SPEED_PER_POWER = 0.9

# 三人制战术：一人处理球，一人接应/封线，一人保护身后。
TRIANGLE_SPACING_M = 5.2
TRIANGLE_WIDTH_M = 3.3
ATTACK_SWEEPER_X = -3.8
DEFEND_KEEPER_LINE_M = 0.5
DEFEND_KEEPER_Y_MARGIN_M = 0.25
DEFEND_COVER_DEPTH_M = 2.2
DEFEND_COVER_SEPARATION_M = 1.4
POSSESSION_REACH_M = 1.4
POSSESSION_ADVANTAGE_M = 0.35
POSSESSION_HOLD_S = 0.8
COUNTERPRESS_S = 3.0   # 仅战术转换，绝不用于放开裁判限制
TACTIC_ROLE_KEEP_S = 0.55
TACTIC_INTERCEPT_KEEP_S = 0.45
TACTIC_BLOCKED_COST_S = 2.0
SHOT_MAX_DISTANCE_M = 8.0
SHOT_CLOSE_DISTANCE_M = 2.5
SHOT_LANE_CLEARANCE_M = 0.48
SHOT_TARGET_POST_MARGIN_M = 0.35
SHOT_POWER = 8.5
ADVANCE_POWER = 5.0
TACTIC_AUDIT_INTERVAL_S = 2.0
PASS_MIN_DISTANCE_M = 2.2
PASS_MAX_DISTANCE_M = 8.0
PASS_LANE_CLEARANCE_M = 0.7
PASS_ARRIVAL_SPEED_MPS = 0.6
PASS_POWER_MIN = 1.0
PASS_POWER_MAX = 3.0
PASS_PRESSURE_M = 1.3
PASS_PROGRESS_MIN_M = 0.6
PASS_PLAN_HOLD_S = 2.5
PASS_FLIGHT_HOLD_S = 2.0
PASS_LAUNCH_DISTANCE_M = 0.55
PASS_LAUNCH_SPEED_MPS = 0.65
PASS_LAUNCH_DIRECTION_COS = 0.75
PASS_RUN_AHEAD_M = 2.0
PASS_RUN_HOLD_S = 3.0
PASS_LEAD_TIME_S = 0.45
PASS_LEAD_MAX_M = 0.7
LEAD_DEFEND_GOAL_MARGIN = 2
BALL_FAST_INTERCEPT_MPS = 0.8
BALL_KICK_ENGAGE_M = 2.0
SHOT_CENTER_DISTANCE_M = 4.0
KEEPER_CLEAR_REACH_M = 1.25
KEEPER_SWEEP_REACH_M = 1.4
KEEPER_FAST_SHOT_MPS = 0.8
KEEPER_INCOMING_VX_MPS = -0.2
KEEPER_FOOT_REACH_M = 0.4
KEEPER_GOALSIDE_MARGIN_M = 0.2
KICKOFF_SIDE_ANGLE_RAD = math.radians(45)
RESTART_SUPPORT_DEPTH_M = 2.5
TACTIC_CLEARANCE_Y_M = 3.0
TACTIC_TARGET_REFRESH_M = 0.5
TACTIC_MIN_ACTION_HOLD_S = 0.35
RESTART_SAFE_PASS_POWER = 1.8

# 寻路无进展时改道；这些计时不用于比赛放行。
NAV_STALL_S = 1.4
NAV_PROGRESS_M = 0.18
NAV_DETOUR_LENGTH_M = 1.1
NAV_DETOUR_HOLD_S = 1.8
NAV_DETOUR_ARRIVE_M = 0.25
NAV_ROLE_ROTATE_S = 1.6
NAV_TARGET_CHANGE_M = 2.0
NAV_ESCAPE_ANGLES = tuple(math.radians(a) for a in (60,-60,90,-90,120,-120,180))
TACTIC_RECOVERY_OFFSET_M = 1.2

# 重开回位、返场与门线救险
READY_RETURN_X_M = -0.65
READY_RETURN_LANE_M = 2.5
READY_TEAM_SEPARATION_M = 0.85
GOAL_DANGER_DEPTH_M = 1.5
GOAL_DANGER_WIDTH_MARGIN_M = 0.6
GOAL_LINE_KICK_FIELD_MARGIN_M = 0.03
GOAL_CLEAR_SIDE_M = 2.8
GOAL_CLEAR_FORWARD_M = 4.0
ACTIVE_REPOSITION_M = 0.75
ACTIVE_REPOSITION_MIN_BALL_M = 0.65

# 2026-11-18：比赛收益优先。净胜至少两球才收缩；落后全队前压。
READY_PRESS_X_M = -0.8
READY_PRESS_WING_M = 2.3
READY_BASE_X_M = -5.8
ATTACK_BASE_DEPTH_M = 4.5
ATTACK_BASE_LIMIT_X_M = 1.0
CHASE_BASE_DEPTH_M = 3.5
DEFEND_TRANSITION_X_M = 1.5
MARK_GOALSIDE_M = 0.85
MARK_SWITCH_COST_M = 0.5
PRESS_GOALSIDE_M = 0.55
PRESS_CONTEST_M = 1.4
TOUCHLINE_ZONE_M = 0.8
LIVE_BALL_FIELD_MARGIN_M = 0.03
TOUCHLINE_ROBOT_EXCURSION_M = 0.85  # SDK实测球后接近到边线外约0.53m，留完成动作的空间
BOUNDARY_RETURN_DEPTH_M = 2.0
RECEIVE_REACH_M = 0.65
RECEIVE_SPEED_MPS = 0.7
RECEIVE_LOOKAHEAD_S = 0.45
PASS_MIN_FLIGHT_S = 0.35
PASS_RECEIVE_GRACE_S = 1.2
PASS_READY_TOL_M = 0.45
CONTROL_HOLD_S = 0.22
CONTROL_REACH_M = 0.9
CONTROL_ADVANTAGE_M = 0.15
CONTROL_LOSS_S = 1.2
EVENT_CLEARANCE_TRAVEL_M = 1.5
EVENT_CLEARANCE_MARGIN_M = 0.4
EVENT_WINDOW_S = 6.0
EVENT_BOUNDARY_RECOVER_M = 1.0
EVENT_HISTORY_LIMIT = 200
REWARD_RECEIVE = 0.2
REWARD_LOSS = -0.2
REWARD_INTERCEPT = 0.3
REWARD_DISPOSSESSED = -0.3
REWARD_BOUNDARY_SAVE = 0.3
REWARD_CLEARANCE = 0.6
REWARD_PENALTY = -0.5
REWARD_GOAL = 1.0
GOAL_RESCUE_BACK_MARGIN_M = 0.1
GOAL_RESCUE_POST_MARGIN_M = 0.25
