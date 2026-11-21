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
OPP_RESTART_SCREEN_M = 2.35  # 站在官方避让圈外封堵，而非全部退至禁区。
OPP_RESTART_COVER_DEPTH_M = 3.2
OPP_RESTART_COVER_WIDTH_M = 1.7
OPP_KICKOFF_FRONT_X_M = -2.3  # 中路纵深两层，READY即落位，避免中圈两翼被直球穿过。
OPP_KICKOFF_BACK_X_M = -3.8
OPP_KICKOFF_SCREEN_DEPTH_M = 1.5
OPP_KICKOFF_PRESS_S = 4.0
OPP_KICKOFF_FAST_BALL_MPS = 1.0
OPP_RESTART_PRESS_S = 2.0  # 只在官方解除重启限制后计战术压迫窗口。
OPP_RESTART_PRESS_DEPTH_M = 1.2
OPP_RESTART_PRESS_WIDTH_M = 1.3
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
ACTIVE_STALL_S = 1.6  # 活球无位移时快速换路，不以远离球为默认处理。
KICK_TASK_STALL_S = 4.0  # SDK接近/调整出脚需要连续控制，不能沿用普通行走的短脱困窗口。
INACTIVITY_PROGRESS_M = 0.25
INACTIVITY_BALL_DISTANCE_M = 3.2
KICKOFF_TAKER_LINE_MARGIN_M = 0.1
GET_UP_RETRY_S = 3.0           # 启动失败的重试间隔；getting_up期间绝不重复提交
MODE_REQUEST_RETRY_S = 0.75
RECOVERY_SOFT_START_S = 0.8
RECOVERY_KICK_SETTLE_S = 0.25
RECOVERY_WALK_LIMIT_MPS = 0.5

# 指令整形：工程初值，非SDK的稳定性保证；暂停/罚下立即清零，不做缓停。
WALK_SIDE_LIMIT_MPS = 0.85
WALK_BACK_LIMIT_MPS = 0.8
WALK_NEAR_ROBOT_M = 1.0
WALK_NEAR_ROBOT_LIMIT_MPS = 0.6
WALK_TURN_LIMIT_RADPS = 1.2
WALK_TURN_SLOWDOWN = 0.4
WALK_ACCEL_MPS2 = 2.5
WALK_YAW_ACCEL_RADPS2 = 4.0
WALK_COMMAND_MAX_DT_S = 0.1
WALK_COMMAND_RESET_GAP_S = 0.4


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
TRIANGLE_WIDTH_M = 2.0  # 接应靠近球路，减少长横传暴露时间。
ATTACK_SWEEPER_X = -3.8
DEFEND_KEEPER_LINE_M = 0.5
DEFEND_KEEPER_Y_MARGIN_M = 0.25
DEFEND_COVER_DEPTH_M = 2.2
DEFEND_COVER_SEPARATION_M = 1.4
HOME_HALF_TRIGGER_X_M = 0.0  # 活球进入己方半场，立即覆盖前插/反抢等待分支。
HOME_COVER_FIELD_FRACTION = 0.25  # 从己方门线向前全场长度的1/4。
HOME_COVER_Y_LIMIT_M = 1.4  # 第二道保护守中路，随球侧移动但不追到边线。
POSSESSION_REACH_M = 1.4
POSSESSION_ADVANTAGE_M = 0.35
POSSESSION_HOLD_S = 0.8
COUNTERPRESS_S = 0.8   # 最长反抢窗口，仅战术转换，绝不用于放开裁判限制
COUNTERPRESS_REACH_M = 1.8
COUNTERPRESS_CONTACT_M = 0.75
COUNTERPRESS_BYPASS_M = 0.8
LOOSE_RACE_ADVANTAGE_S = 0.15  # 预计先到球即可出击，不要求先进入近球控制半径。
LOOSE_OPPONENT_CONTACT_M = 1.0
LOOSE_KICK_CONTACT_M = 0.9
LOOSE_CLEAR_MIN_X_M = 1.5
RETREAT_PREDICT_S = 0.45
RETREAT_SCREEN_M = 1.0
RETREAT_KEEPER_SPACE_M = 1.0
RETREAT_TURN_FIRST_RAD = math.radians(65)
RETREAT_FACE_TRAVEL_M = 0.8
TACTIC_ROLE_KEEP_S = 0.55
TACTIC_INTERCEPT_KEEP_S = 0.45
TACTIC_BLOCKED_COST_S = 2.0
SHOT_MAX_DISTANCE_M = 8.0
SHOT_CENTRE_DISTANCE_M = 4.0  # 远射优先中央，近门才用更细的空侧瞄准。
SHOT_CLOSE_DISTANCE_M = 2.5
SHOT_LANE_CLEARANCE_M = 0.48
SHOT_TARGET_POST_MARGIN_M = 0.35
SHOT_POWER = 8.5
ADVANCE_POWER = 5.0
TACTIC_AUDIT_INTERVAL_S = 2.0
PASS_MIN_DISTANCE_M = 1.6  # 允许小范围短传，避免等待队友拉开过远。
PASS_MAX_DISTANCE_M = 8.0
PASS_LANE_CLEARANCE_M = 0.7
PASS_ARRIVAL_SPEED_MPS = 0.6
PASS_POWER_MIN = 1.0
PASS_POWER_MAX = 3.0
PASS_PRESSURE_M = 1.3
PASS_PROGRESS_MIN_M = 0.6
PASS_PLAN_HOLD_S = 4.0  # 缩短未出脚计划上限；球离脚后仍按球速估接球窗口。
PASS_FLIGHT_HOLD_S = 2.0
PASS_LAUNCH_DISTANCE_M = 0.25
PASS_LAUNCH_SPEED_MPS = 0.35
PASS_LAUNCH_DIRECTION_COS = 0.75
PASS_RUN_AHEAD_M = 2.0
PASS_RUN_HOLD_S = 3.0
PASS_LEAD_TIME_S = 0.45
PASS_LEAD_MAX_M = 0.7
LEAD_DEFEND_GOAL_MARGIN = 2
BALL_FAST_INTERCEPT_MPS = 0.8
BALL_KICK_ENGAGE_M = 2.0
SHOT_AIM_HOLD_S = 1.2
SHOT_AIM_SWITCH_CLEARANCE_M = 0.2
SHOT_AIM_RESET_M = 1.0
KEEPER_IDLE_SHUFFLE_S = 5.5  # 门将只在门线附近短横移，不参加普通向前脱困。
KEEPER_IDLE_SHUFFLE_M = 0.35
KEEPER_CLEAR_REACH_M = 1.25
KEEPER_SWEEP_REACH_M = 1.4
KEEPER_CONTEST_REACH_M = 2.6  # 近门慢球且门将明显更近时，主动出击解围。
KEEPER_CONTEST_ADVANTAGE_M = 0.45
KEEPER_ADVANCE_DEPTH_M = 0.8
KEEPER_ADVANCE_BALL_GAP_M = 0.55
KEEPER_FAST_SHOT_MPS = 0.8
KEEPER_INCOMING_VX_MPS = -0.2
KEEPER_FOOT_REACH_M = 0.4
KEEPER_GOALSIDE_MARGIN_M = 0.2
KICKOFF_SIDE_ANGLE_RAD = math.radians(45)
KICKOFF_PASS_ANGLE_RAD = math.radians(45)  # 低力度向前斜推，主罚连续跟进，避免横传长时间暴露。
KICKOFF_RECEIVER_X_M = -0.8  # 给行走惯性留半场余量，避免接应者在中线附近越界。
KICKOFF_RECEIVER_Y_M = 2.05
KICKOFF_RECEIVER_READY_M = 0.4
KICKOFF_RECEIVER_WAIT_S = 0.0  # 主罚立即接近，不在10秒开球窗口内等待远处队友。
KICKOFF_FOLLOW_S = 4.0
KICKOFF_AUDIT_S = 0.25
KICKOFF_EVIDENCE_S = 8.0  # 取证窗口独立于4秒接应保持，不用于裁判放行。
KICKOFF_TAKER_READY_X_M = -0.45
KICKOFF_TAKER_READY_Y_M = 0.4
PRESS_SUPPORT_LOOKAHEAD_S = 0.6
PRESS_SUPPORT_DEPTH_M = 0.8
PRESS_SUPPORT_WIDTH_M = 1.6
BALL_TURN_LATENCY_S_PER_RAD = 0.25  # 到球估计考虑转身；不假定机器人能瞬时变向。
PASS_FLIGHT_ESTIMATE_SPEED_MPS = 1.5
PASS_INTERCEPT_FRACTIONS = (0.25,0.5,0.75,1.0)
PASS_RACE_MARGIN_S = 0.15
RESTART_SUPPORT_DEPTH_M = 2.5
TACTIC_CLEARANCE_Y_M = 3.0
TACTIC_TARGET_REFRESH_M = 0.5
TACTIC_MIN_ACTION_HOLD_S = 0.35
RESTART_SAFE_PASS_POWER = 1.8

# 寻路无进展时改道；这些计时不用于比赛放行。
NAV_STALL_S = 0.8
NAV_PROGRESS_M = 0.18
NAV_DETOUR_LENGTH_M = 1.1
NAV_DETOUR_HOLD_S = 0.9
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
GOAL_CLEAR_AIM_HOLD_S = 0.9
GOAL_CLEAR_AIM_RESET_M = 0.8
GOAL_CLEAR_SWITCH_ADVANTAGE_M = 0.25
GOAL_CLEAR_BLOCKED_M = 0.35
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
TACKLE_ENGAGE_M = 2.0
TACKLE_REENGAGE_M = 2.5
TACKLE_GOALSIDE_MARGIN_M = 0.35
TACKLE_SWITCH_ADVANTAGE_M = 0.6  # 接近同一球时保留处理者，队友明显更近才接替。
CORNER_LANE_RADIUS_M = 0.85
CORNER_BALL_CLEAR_M = 1.3
CORNER_YIELD_STEP_M = 1.6
CORNER_SUPPORT_INSET_M = 2.4
TOUCHLINE_ZONE_M = 0.8
LIVE_BALL_FIELD_MARGIN_M = 0.03
TOUCHLINE_ROBOT_EXCURSION_M = 0.85  # SDK实测球后接近到边线外约0.53m，留完成动作的空间
BOUNDARY_RETURN_DEPTH_M = 2.0
BOUNDARY_INWARD_M = 2.0
BOUNDARY_RECEIVE_INSET_M = 1.0
BOUNDARY_ARRIVAL_SPEED_MPS = 0.25
RESTART_INWARD_MIN_SIN = 0.5
PASS_RECEIVER_SPACE_M = 0.8
PASS_ABORT_LANE_M = 0.4
PASS_OVERSHOOT_M = 0.8
PASS_OFFLINE_M = 1.2
RECEIVE_CUSHION_OFFSET_M = 0.25
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
