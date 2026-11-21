"""Player:单个机器人的控制 handle —— 用户可直接编辑这个类。

平台原语(set_velocity / kick / release_kick / request_mode / get_up)委托给注入的
``_backend``(framework 层);状态 property(pose / mode / is_fallen / penalty)从
``self.context`` 或 backend 读。

走位行为(walk_to / face_to / ensure_ready)也是 Player 方法:它们本质是"对本
球员下命令的动词",且未来加迟滞/避障需要的跨帧状态可直接挂 ``self``。纯坐标计算
(dist / angle_to / 球门坐标等)才放 utils/geom。

实例跨整场比赛存活;``self.context`` 每帧被框架覆写。用户想加自己的拐棍方法直接
在这里加。
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from .framework.types import Context, Penalty, Pose2D, GameState, SetPlay, KICKING_TEAM_NONE
from .framework import debugdraw

from .param import *
from .utils.geom import (
    angle_to,
    clamp,
    dist,
    normalize_angle,
    opponent_goal,
    own_goal,
    own_goal_area_center,
)
from .utils.obstacles import collect_obstacles, Obstacle
from .utils.path_planner import plan_global_path
from .utils.motion_control import WalkCommand
from .utils.rules import legal_target, safe_velocity, safe_push_direction, segment_clear, goal_danger, live_ball_access

if TYPE_CHECKING:
    from .framework.config import SoccerConfig


__all__ = ["Player"]


_log = logging.getLogger(__name__)

def _heading_clearance(
    px: float, py: float, heading: float, obstacles: list,
) -> float:
    """用于局部避障。沿 (px,py) 朝 ``heading`` 的 lookahead 射线,到最近障碍的余量(dist - radius)。

    只考虑**前方**(投影 t>0)的障碍;身后 / 侧后的障碍不挡这个方向,跳过——否则贴近
    任何障碍(哪怕正后方)都会把所有方向的余量拉低,误判"全堵"。
    无障碍返回 inf;越大越空,负值表示会撞。
    """
    ux, uy = math.cos(heading), math.sin(heading)
    min_clear = math.inf
    for obs in obstacles:
        t = (obs.x - px) * ux + (obs.y - py) * uy
        if t <= 0.0:
            continue                      # 障碍在身后/侧后,不挡此方向
        if t > PLAN_LOOKAHEAD:
            t = PLAN_LOOKAHEAD
        nx, ny = px + ux * t, py + uy * t
        clear = math.hypot(obs.x - nx, obs.y - ny) - obs.radius
        if clear < min_clear:
            min_clear = clear
    return min_clear


def _behind_ball(
    ball_x: float, ball_y: float, aim: tuple[float, float], offset: float,
) -> tuple[float, float]:
    """用于计算站位。球在 球→``aim`` 连线上的"后方"点:从球沿【背离 aim】方向退 ``offset`` 米。

    追球走位目标用它:球落在机器人与 ``aim`` 之间,到位时天然对准 aim 方向。
    aim 与球重合(退化)时,方向不定,直接返回球位。
    """
    dx, dy = aim[0] - ball_x, aim[1] - ball_y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (ball_x, ball_y)
    ux, uy = dx / d, dy / d              # 球指向 aim 的单位向量
    return (ball_x - ux * offset, ball_y - uy * offset)   # 背离 aim 退 offset


class Player:
    """单个球员的 handle。可以在这里加入新的技术动作。
    """

    def __init__(
        self,
        player_id: int,
        config: "SoccerConfig",
        _backend: object | None,
    ) -> None:
        self.id: int = player_id
        self.config: "SoccerConfig" = config
        self._backend = _backend           # 机器人控制接口的包装，通常不用改
        self.context: Context | None = None

        # 当前高层动作名(仅供可视化/调试)。策略分派层每帧写入;有子状态的动作
        # (如 guard)在方法内部细化。可视化 pass 读它标注文字,见 main.py。
        self.action: str = "init"

        # SDK 缓存字段,框架后台会自动更新
        self._mode: str | None = None
        self._fall_down_state: str | None = None

        # 避障绕行侧记忆(跨帧;None=当前无绕行)
        self._avoid_side: float | None = None

        # 踢球迟滞状态(跨帧)
        self._kicking: bool = False
        self.rule_discs: list[Obstacle] = []
        self.rule_own_half = False
        self.rule_can_kick = False
        self.rule_shot_origin = None
        self.kickoff_pass_direction = None
        self.ball_access = False
        self._activity_anchor = None
        self._activity_since = None
        self._activity_ball = None
        self._goal_idle_pose = self._goal_idle_since = self._goal_shuffle = None
        self._clear_aim = self._clear_origin = None
        self._clear_until = 0.0
        self._retreating = False
        self._rescue_target = None
        self._recovering = False
        self._get_up_requested_at = None
        self._recovery_state = None
        self._recovery_seen_busy = False
        self._recovery_soft_until = 0.0
        self._recovery_kick_until = 0.0
        self._mode_request = None
        self._walk_command = WalkCommand()
        self.readiness_action = 'switching_mode'
        self.ball_motion = None
        self.evaluation = None
        self._nav_anchor = None
        self._nav_since = None
        self._nav_goal = None
        self._nav_detour = None
        self._nav_detour_until = 0.0
        self._nav_attempt = 0
        self.routing_blocked_until = 0.0

        # block/guard 的跨帧迟滞状态
        self._block_pressing: bool = False
        self._guard_threatened: bool = False
        # 门线撞球进门迟滞状态
        self._goal_line_push: bool = False
        # support 卡住后临时转 attack 的跨帧状态
        self._support_last_pos: tuple[float, float] | None = None
        self._support_stationary_since: float | None = None
        self._support_last_update_at: float | None = None

    # ------------------------------------------------------------------
    # 状态读取
    # ------------------------------------------------------------------

    @property
    def is_kicking(self) -> bool:
        """当前是否处于踢球状态。"""
        return self._kicking

    @property
    def pose(self) -> Pose2D | None:
        ctx = self.context
        if ctx is None:
            return None
        robot = ctx.teammates.get(self.id)
        return None if robot is None else robot.pose

    @property
    def mode(self) -> str | None:
        if self._backend is not None:
            return self._backend.mode
        return self._mode

    @property
    def is_fallen(self) -> bool:
        return self.fall_down_state not in (None, "normal")

    @property
    def fall_down_state(self) -> str | None:
        if self._backend is not None:
            return getattr(self._backend, "fall_down_state", None)
        return self._fall_down_state

    @property
    def penalty(self) -> Penalty:
        ctx = self.context
        if ctx is None or ctx.game is None:
            return Penalty.NONE
        state = ctx.game.get_player_state(self.config.team_id, self.id)
        return Penalty.NONE if state is None else state.penalty

    @property
    def is_penalized(self) -> bool:
        return self.penalty != Penalty.NONE

    # ------------------------------------------------------------------
    # 底盘控制
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        if not self._motion_allowed() or self.pose is None or self.is_fallen:
            self.release_kick()
            vx = vy = vyaw = 0.0
            self._walk_command.reset()
        else:
            pose = self.pose
            limit = RECOVERY_WALK_LIMIT_MPS if self.context.now < self._recovery_soft_until else MAX_LINEAR
            nearby = [r for i,r in self.context.teammates.items() if i != self.id and r.pose is not None]
            nearby.extend(r for r in self.context.opponents.values() if r.pose is not None)
            if any(dist(pose.x,pose.y,r.pose.x,r.pose.y) < WALK_NEAR_ROBOT_M for r in nearby):
                limit = min(limit,WALK_NEAR_ROBOT_LIMIT_MPS)
            vx, vy, vyaw = self._walk_command.shape(
                pose,self.context.now,self.context.dt,vx,vy,vyaw,limit)
            c, s = math.cos(pose.theta), math.sin(pose.theta)
            if vx or vy:
                # 最后再检查规则几何，平滑不能把旧方向带进禁圈。
                wx, wy = safe_velocity(self.context, pose, vx*c-vy*s, vx*s+vy*c,
                                       self.rule_discs, self.rule_own_half,ball_access=self.ball_access)
                vx, vy = wx*c+wy*s, -wx*s+wy*c
            self._walk_command.record(pose,self.context.now,vx,vy,vyaw)
        if self._backend is None:
            _log.debug(
                "player %d set_velocity vx=%.3f vy=%.3f vyaw=%.3f (no backend)",
                self.id, vx, vy, vyaw,
            )
            return
        self._backend.set_velocity(vx, vy, vyaw)

    def _motion_allowed(self) -> bool:
        g = self.context.game if self.context is not None else None
        return (g is not None and not g.stopped and not self.is_penalized
                and g.state in (GameState.READY, GameState.PLAYING))

    def stop(self) -> None:
        self.release_kick()
        self.set_velocity(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 踢球
    # ------------------------------------------------------------------

    def kick(
        self,
        kick_direction: float | None = None,
        power: float = KICK_POWER_DEFAULT,
    ) -> None:
        pose = self.pose
        g = self.context.game if self.context is not None else None
        if (not self._motion_allowed() or g.state != GameState.PLAYING
                or not self.rule_can_kick or self.is_fallen or self._recovering or pose is None
                or self.context.now < self._recovery_kick_until
                or (g.kicking_team not in (self.context.team_id, KICKING_TEAM_NONE)
                    and (g.set_play != SetPlay.NONE or g.secondary_time > 0))):
            self.stop()
            return

        ball = self.context.ball if self.context is not None else None
        if ball is None:
            self.stop()
            return

        # 活球处理者可用边线外短绕行空间；官方禁圈仍然检查。
        # 主罚接近球不套用普通走位的0.3m半场余量；下面单独检查主罚中线余量。
        legal_pose = legal_target(self.context, (pose.x,pose.y), self.rule_discs, False,
                                  ball_access=live_ball_access(self.context))
        # 门线上仍是活球。救险只缩小边界工程余量，绝不绕过暂停/罚下/禁圈。
        rescue = goal_danger(self.context) and not self.rule_own_half
        in_field = (abs(pose.x) <= self.context.field.length/2-GOAL_LINE_KICK_FIELD_MARGIN_M
                    and abs(pose.y) <= self.context.field.width/2-GOAL_LINE_KICK_FIELD_MARGIN_M)
        goal_pocket = (-self.context.field.length/2-GOAL_DEPTH+GOAL_RESCUE_BACK_MARGIN_M <= pose.x
                       <= -self.context.field.length/2+FIELD_SAFE_MARGIN_M
                       and abs(pose.y)<self.context.field.goal_width/2-GOAL_RESCUE_POST_MARGIN_M)
        pose_clear = all(dist(pose.x,pose.y,c.x,c.y) >= c.radius+RULE_TARGET_MARGIN_M for c in self.rule_discs)
        if any(math.hypot(ball.x-c.x,ball.y-c.y) < c.radius for c in self.rule_discs):
            self.stop()
            return
        if legal_pose != (pose.x,pose.y) and not (rescue and (in_field or goal_pocket) and pose_clear):
            # SDK接近后可能把人带过边界余量；主动走回内场，不能永远只取消任务。
            recovery = legal_target(self.context,(pose.x,pose.y),self.rule_discs,self.rule_own_half)
            if recovery is not None:
                self.action = 'kick:return_infield'
                self.walk_to(recovery,ball_access=True)
            else:
                self.stop()
            return
        if self.rule_own_half and (ball.x > KICKOFF_TAKER_LINE_MARGIN_M
                                  or pose.x > -KICKOFF_TAKER_LINE_MARGIN_M):
            self.walk_to((min(pose.x,-HALF_SAFE_MARGIN_M),pose.y))
            return

        if rescue:
            target = self.clearance_target()
            kick_direction = angle_to(ball.x,ball.y,*target)
            power = SHOT_POWER

        restricted = self.rule_shot_origin is not None and dist(
            ball.x, ball.y, *self.rule_shot_origin) <= RESTART_SHOT_DISTANCE_M
        if restricted:
            kick_direction = safe_push_direction(self.context)
            if self.kickoff_pass_direction is not None:
                kick_direction = safe_push_direction(self.context,self.kickoff_pass_direction)
            elif self.rule_own_half:
                kick_direction = KICKOFF_SIDE_ANGLE_RAD
            power = RESTART_PUSH_POWER
            if kick_direction is None:
                self.stop()
                return

        if kick_direction is None:
            kick_plan = self.plan_kick()
            if kick_plan is None:
                self.stop()
                return
            kick_direction, power = kick_plan

        if self._backend is None:
            _log.debug(
                "player %d kick ball=(%.3f, %.3f) dir=%.3f (no backend)",
                self.id, ball.x, ball.y, kick_direction,
            )
            return
        # 场地坐标 → 体坐标(用当前 pose)
        dx = ball.x - pose.x
        dy = ball.y - pose.y
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        ball_x_body = dx * cos_t + dy * sin_t
        ball_y_body = -dx * sin_t + dy * cos_t
        kick_direction = normalize_angle(kick_direction)
        direction_body = normalize_angle(kick_direction - pose.theta)
        power_clamped = max(KICK_POWER_MIN, min(KICK_POWER_MAX, power))
        if self.ball_motion is not None:
            self.ball_motion.observe_kick_request(self.context,self.id,power_clamped,kick_direction)
        self._kicking = True
        self._walk_command.reset()  # SDK独占底盘时不沿用先前行走速度
        self._backend.kick(direction_body, power_clamped, ball_x_body, ball_y_body)
        if self.evaluation is not None and dist(pose.x,pose.y,ball.x,ball.y) <= BALL_KICK_ENGAGE_M:
            self.evaluation.record_kick(self.context,self.id,
                (ball.x+math.cos(kick_direction),ball.y+math.sin(kick_direction)))

    def plan_kick(self) -> tuple[float, float] | None:
        """计算踢球方向和力度。

        从当前球位踢向对方球门中心；实际执行仍经过 kick 的合规闸门。
        返回 ``(kick_direction, kick_power)``;球或上下文不可用时返回 None。
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return None

        kick_target = opponent_goal(ctx)
        kick_direction = angle_to(ball.x, ball.y, *kick_target)
        kick_target = self._goal_target_for_direction(kick_direction)
        kick_power = (
            KICK_POWER_BACKFIELD if self._in_backfield()
            else KICK_POWER_DEFAULT
        )

        self._draw_kick_target(kick_target)
        return kick_direction, kick_power

    def _goal_target_for_direction(
        self, kick_direction: float,
    ) -> tuple[float, float]:
        """把射门方向投到对方门线上,用于可视化踢球目标。"""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return (0.0, 0.0)

        dx = math.cos(kick_direction)
        if dx <= 1e-6:
            return opponent_goal(ctx)
        goal_x = ctx.field.length / 2.0
        t = max(0.0, (goal_x - ball.x) / dx)
        return (goal_x, ball.y + math.sin(kick_direction) * t)

    def _in_backfield(self) -> bool:
        """球在我方后场时,默认踢球加大力度。"""
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return False
        # own_penalty_edge_x = -ctx.field.length / 2.0 + ctx.field.penalty_area_length
        # return ball.x < own_penalty_edge_x
        return ball.x < 0

    def _draw_kick_target(self, target: tuple[float, float]) -> None:
        """以 X 标出 plan_kick 选择的踢球目标。"""
        from .framework import debugdraw

        x, y = target
        s = KICK_TARGET_MARK_SIZE_M
        debugdraw.line(
            [(x - s, y - s), (x + s, y + s)],
            rgb=(1.0, 0.0, 1.0), ns="kick_target",
        )
        debugdraw.line(
            [(x - s, y + s), (x + s, y - s)],
            rgb=(1.0, 0.0, 1.0), ns="kick_target",
        )

    def release_kick(self) -> None:
        if self._kicking:
            self._walk_command.reset()
        self._kicking = False  # 清除踢球迟滞标志(取消方块显示)
        if self._backend is None:
            _log.debug("player %d release_kick (no backend)", self.id)
            return
        self._backend.release_kick()

    def kick_can_score(self, kick_direction: float) -> bool:
        """判断从当前球位按 ``kick_direction`` 踢,直线轨迹是否能进对方球门。
        """
        ctx = self.context
        ball = ctx.ball if ctx is not None else None
        if ctx is None or ball is None:
            return False

        goal_x = ctx.field.length / 2.0
        dx = math.cos(kick_direction)
        dy = math.sin(kick_direction)
        if dx <= 1e-6:
            return False

        half_goal = ctx.field.goal_width / 2.0
        if half_goal <= 0.0:
            return False
        if ball.x >= goal_x:
            y_at_goal = ball.y
        else:
            t = (goal_x - ball.x) / dx
            if t < 0.0:
                return False
            y_at_goal = ball.y + dy * t
        return -half_goal <= y_at_goal <= half_goal

    # ------------------------------------------------------------------
    # 慢操作（同步接口异步调取，使得不阻塞）
    # ------------------------------------------------------------------

    def request_mode(self, mode: str) -> None:
        if not self._motion_allowed():
            self.stop()
            return
        if self._backend is None:
            _log.debug("player %d request_mode -> %s (no backend)", self.id, mode)
            return
        now = self.context.now
        if (self._mode_request is not None and self._mode_request[0] == mode
                and 0 <= now-self._mode_request[1] < MODE_REQUEST_RETRY_S):
            return
        self._mode_request = (mode,now)
        self._backend.request_mode(mode)

    def get_up(self) -> None:
        if not self._motion_allowed():
            self.stop()
            return
        if self._backend is None:
            _log.debug("player %d get_up (no backend)", self.id)
            return
        self._backend.get_up()

    # ------------------------------------------------------------------
    # 走位
    # ------------------------------------------------------------------

    def ensure_ready(self) -> bool:
        """允许运动时异步起身/切模式；起身会移动，停止阶段禁止新启动。

        返回本帧是否可执行动作(True=已就绪)。
        """
        if not self._motion_allowed():
            self.stop()
            return False
        state = self.fall_down_state
        previous = self._recovery_state
        if state != previous:
            _log.info('RECOVERY player=%d state=%s previous=%s mode=%s action=%s last_world=%s',
                      self.id,state,previous,self.mode,self.action,self._walk_command.world)
            self._recovery_state = state
        self.readiness_action = 'switching_mode'
        if self._recovering and state == "normal":
            self._recovering = False
            self._get_up_requested_at = None
            self._recovery_seen_busy = False
            self._recovery_soft_until = self.context.now+RECOVERY_SOFT_START_S
            self._recovery_kick_until = self.context.now+RECOVERY_KICK_SETTLE_S
            self.reset_navigation()
        if state == 'getting_up':
            self._recovering = True
            self._recovery_seen_busy = True
        if self.is_fallen or self._recovering:
            self.stop()
            self.readiness_action = 'recover:'+str(state or 'unknown')
            # get_up 是异步任务，后台“调用返回”不等于机器人已站起。
            # 只在 SDK 的 recoverable 明确允许时发起；等待期间不抢发 walk。
            recoverable = getattr(self._backend,"fall_down_recoverable",False)
            now = self.context.now
            # 起身中的缓存可能短暂为空；再次明确报告可恢复的倒地也允许重试。
            explicit_retry = state in ('falling','fallen') and self._recovery_seen_busy and recoverable
            if explicit_retry:
                self._recovery_seen_busy = False
            if (recoverable and state in ('falling','fallen') and not self._recovery_seen_busy
                    and (self._get_up_requested_at is None
                                or now-self._get_up_requested_at >= GET_UP_RETRY_S)):
                self._recovering = True
                self._get_up_requested_at = now
                self.get_up()
            return False
        if state is None and self.mode != 'walk':
            # SDK刚派发异步任务可能清空缓存，未知不等于已站稳。
            self.stop()
            self.readiness_action = 'recover:unknown'
            return False
        if self.mode != "walk":
            self.stop()
            self.request_mode("walk")
            return False
        return True

    def face_to(self, target_theta: float) -> None:
        """原地转向到目标朝向。"""
        if self.pose is None:
            self.stop()
            return
        err = normalize_angle(target_theta - self.pose.theta)
        self.set_velocity(0.0, 0.0, self._angular(err))

    def reset_navigation(self):
        """阶段变化/返场清除旧路线和卡住记忆，当帧按新局势决策。"""
        self._nav_anchor = self._nav_since = self._nav_goal = self._nav_detour = None
        self._nav_detour_until = self.routing_blocked_until = 0.0
        self._activity_anchor = self._activity_since = None
        self._activity_ball = None
        self._goal_idle_pose = self._goal_idle_since = self._goal_shuffle = None
        self._clear_aim = self._clear_origin = None
        self._clear_until = 0.0
        self._retreating = False
        self._rescue_target = None
        self._walk_command.reset()

    def _navigation_target(self,target,obstacles):
        pose,now = self.pose,self.context.now
        point = (pose.x,pose.y)
        if (self._nav_goal is None or dist(*target,*self._nav_goal) > NAV_TARGET_CHANGE_M
                or self._nav_since is None or now < self._nav_since):
            self._nav_anchor,self._nav_since,self._nav_goal = point,now,target
            self._nav_detour = None
        if dist(*point,*self._nav_anchor) >= NAV_PROGRESS_M:
            self._nav_anchor,self._nav_since = point,now
        if self._nav_detour is not None:
            if (now < self._nav_detour_until and dist(*point,*self._nav_detour) > NAV_DETOUR_ARRIVE_M
                    and segment_clear(point,self._nav_detour,obstacles)):
                return self._nav_detour
            self._nav_detour = None
        if now-self._nav_since < NAV_STALL_S or dist(*point,*target) < ARRIVE_DIST:
            return target
        self.routing_blocked_until = now+NAV_ROLE_ROTATE_S
        self._nav_attempt += 1
        heading = angle_to(*point,*target)
        offsets = NAV_ESCAPE_ANGLES if self._nav_attempt % 2 else tuple(-a for a in NAV_ESCAPE_ANGLES)
        for offset in offsets:
            a = heading+offset
            candidate = legal_target(self.context,(pose.x+NAV_DETOUR_LENGTH_M*math.cos(a),
                                     pose.y+NAV_DETOUR_LENGTH_M*math.sin(a)),
                                     self.rule_discs,self.rule_own_half,ball_access=self.ball_access)
            if (candidate is not None and dist(*point,*candidate) > NAV_DETOUR_ARRIVE_M
                    and segment_clear(point,candidate,obstacles,GLOBAL_OBSTACLE_MARGIN_M)):
                self._nav_detour,self._nav_detour_until = candidate,now+NAV_DETOUR_HOLD_S
                self._nav_since = now
                _log.info('NAV reroute player=%d attempt=%d target=(%.2f,%.2f)',self.id,self._nav_attempt,*candidate)
                return candidate
        self._nav_since = now
        return target

    def walk_to(
        self,
        target: tuple[float, float],
        *,
        face: float | None = None,
        avoid_ball: bool = False,
        avoid_robots: bool = False,
        arrive_dist: float = ARRIVE_DIST,
        ball_access: bool = False,
    ) -> bool:
        """走向目标点。返回是否已到达。

        避障(局部规划器,简化版 VFH):``avoid_ball`` / ``avoid_robots`` 开启时,把
        球 / 机器人 / 球门结构收集成圆形障碍,在机器人周围扫一圈候选方向,选"最朝
        目标 + 前方 ``PLAN_LOOKAHEAD`` 内不撞障碍"的方向前进。能处理多障碍、凹形
        结构和对称冲突,比单障碍绕行 / 势场排斥稳。

        全向行走，边走边转；``face`` 指定希望保持的朝向。
        """
        self.release_kick() # 踢球状态下，走位会被覆盖，先取消踢球状态
        self.ball_access = ball_access and live_ball_access(self.context)
        pose = self.pose
        if pose is None or not self._motion_allowed() or self.is_fallen:
            self.stop()
            return False

        target = legal_target(self.context, target, self.rule_discs, self.rule_own_half,
                              ball_access=self.ball_access)
        if target is None:
            self.stop()
            return False
        obstacles = collect_obstacles(self.context,self.id,ball=avoid_ball,robots=avoid_robots,goals=True)
        obstacles.extend(self.rule_discs)
        target = self._navigation_target(target,obstacles)
        # 重摆球后已在禁圈内：先径向撤离，不等转身或规划。
        if any(dist(pose.x,pose.y,c.x,c.y) < c.radius for c in self.rule_discs):
            # 撤离也必须考虑门柱/球网，不能径直退进球门结构后卡住。
            escape_obstacles = [*self.rule_discs,*collect_obstacles(
                self.context,self.id,ball=False,robots=False,goals=True)]
            wx,wy = safe_velocity(self.context,pose,target[0]-pose.x,target[1]-pose.y,
                                  escape_obstacles,self.rule_own_half)
            c,s = math.cos(pose.theta),math.sin(pose.theta)
            self.set_velocity(wx*c+wy*s,-wx*s+wy*c,0.0)
            return False

        tx, ty = target
        dx = tx - pose.x
        dy = ty - pose.y
        distance = math.hypot(dx, dy)

        if distance < arrive_dist:
            # 已到达:按需转到目标朝向
            if face is not None:
                err = normalize_angle(face - pose.theta)
                if abs(err) > 0.1:
                    self.set_velocity(0.0, 0.0, self._angular(err))
                else:
                    self.stop()
            else:
                self.stop()
            return True

        # 规划:默认全局 A*,找不到路时退回旧局部规划。
        goal_dir = math.atan2(dy, dx)
        planned_path: list[tuple[float, float]] | None = None
        waypoint: tuple[float, float] | None = None
        if not segment_clear((pose.x,pose.y),(tx,ty),obstacles,GLOBAL_OBSTACLE_MARGIN_M):
            if USE_GLOBAL_PATH_PLANNER:
                planned_path = plan_global_path(
                    self.context,
                    (pose.x, pose.y),
                    (tx, ty),
                    obstacles,
                    own_half=self.rule_own_half,
                    ball_access=self.ball_access,
                )
                if planned_path is not None:
                    waypoint = self._path_waypoint(pose, planned_path, obstacles)
                    heading = angle_to(pose.x, pose.y, waypoint[0], waypoint[1])
                else:
                    heading = self._plan_heading(pose, goal_dir, obstacles)
            else:
                heading = self._plan_heading(pose, goal_dir, obstacles)
        else:
            heading = goal_dir

        # 可视化:目标点(绿)、到目标连线(灰)、规划朝向(黄箭头)、
        # 前方探测射线(青,长度=lookahead;规划器"看"的范围,无 path/途径点概念)
        from .framework import debugdraw
        debugdraw.point(tx, ty, rgb=(0.0, 1.0, 0.0), scale=0.15, ns="target")
        debugdraw.line([(pose.x, pose.y), (tx, ty)], rgb=(0.4, 0.4, 0.4), ns="to_target")
        if planned_path is not None and len(planned_path) >= 2:
            debugdraw.line(planned_path, rgb=(0.2, 0.8, 1.0), ns="global_path")
        if waypoint is not None:
            debugdraw.point(
                waypoint[0], waypoint[1],
                rgb=(0.2, 0.8, 1.0), scale=0.12, ns="global_waypoint",
            )
        debugdraw.arrow(
            pose.x, pose.y,
            pose.x + math.cos(heading) * 0.6, pose.y + math.sin(heading) * 0.6,
            rgb=(1.0, 1.0, 0.0), ns="heading",
        )
        debugdraw.line(
            [(pose.x, pose.y),
             (pose.x + math.cos(heading) * PLAN_LOOKAHEAD,
              pose.y + math.sin(heading) * PLAN_LOOKAHEAD)],
            rgb=(0.0, 0.8, 0.8), ns="lookahead",
        )

        # 全向走位，追球时不原地等待转身。
        # 近距离:全向行走,沿 heading 平移,同时转向 face
        travel = min(distance,dist(pose.x,pose.y,*waypoint)) if waypoint is not None else distance
        wdx, wdy = math.cos(heading) * travel, math.sin(heading) * travel
        cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
        vx = LINEAR_GAIN * (wdx * cos_t + wdy * sin_t)
        vy = LINEAR_GAIN * (-wdx * sin_t + wdy * cos_t)
        speed = math.hypot(vx, vy)
        if speed > MAX_LINEAR:
            vx *= MAX_LINEAR / speed
            vy *= MAX_LINEAR / speed
        if face is None:
            face = heading  # 追远球时转向路线，避免长期以受限侧移/倒退接近。
        vyaw = (
            self._angular(normalize_angle(face - pose.theta))
            if face is not None else 0.0
        )
        self.set_velocity(vx, vy, vyaw)
        return False

    def _path_waypoint(
        self, pose: Pose2D, path: list[tuple[float, float]], obstacles=(),
    ) -> tuple[float, float]:
        """Pick a short lookahead waypoint from a planned global path."""
        # 只跳到当前可直达点，避免反复追逐身后的栅格中心或切穿障碍。
        visible = [p for p in path if segment_clear((pose.x,pose.y),p,obstacles,GLOBAL_OBSTACLE_MARGIN_M)]
        for point in reversed(visible):
            if dist(pose.x,pose.y,*point) > ARRIVE_DIST:
                return point
        return path[-1] if path else (pose.x,pose.y)

    def _plan_heading(
        self, pose: Pose2D, goal_dir: float, obstacles: list,
    ) -> float:
        """扫候选方向,选最朝目标且前方够空的方向;都不够空时返回最空的那个。

        候选按偏离目标方向 ``|offset|`` 从小到大试;先试哪一侧由 player_id 奇偶决定,
        用来打破两人同侧避让的对称。
        """
        sign_first = 1.0 if self.id % 2 == 0 else -1.0
        best_h = goal_dir
        best_clear = -math.inf

        offsets = [0.0]
        k = 1
        while k * PLAN_STEP <= PLAN_MAX_OFFSET + 1e-9:
            offsets.append(sign_first * k * PLAN_STEP)
            offsets.append(-sign_first * k * PLAN_STEP)
            k += 1

        for off in offsets:
            h = goal_dir + off
            clear = _heading_clearance(pose.x, pose.y, h, obstacles)
            if clear >= PLAN_CLEARANCE:
                return h
            if clear > best_clear:
                best_clear, best_h = clear, h
        return best_h

    # ------------------------------------------------------------------
    # 高层动作(策略在 main.py 里直接调这些)
    # ------------------------------------------------------------------

    def clearance_target(self):
        ball = self.context.ball
        side = 1.0 if ball.y >= 0 else -1.0
        forward = self.context.field.length/2-FIELD_SAFE_MARGIN_M
        targets = [legal_target(self.context,(forward,y)) for y in
                   (0.0,side*GOAL_CLEAR_SIDE_M,-side*GOAL_CLEAR_SIDE_M)]
        robots = [r for r in self.context.opponents.values() if r.pose is not None]
        robots += [r for pid,r in self.context.teammates.items() if pid != self.id and r.pose is not None]
        def clearance(target):
            dx,dy = target[0]-ball.x,target[1]-ball.y
            length2 = dx*dx+dy*dy
            space = math.inf
            for robot in robots:
                u = ((robot.pose.x-ball.x)*dx+(robot.pose.y-ball.y)*dy)/length2 if length2 else 0.0
                if 0.0 <= u <= 1.0:
                    space = min(space,dist(robot.pose.x,robot.pose.y,ball.x+u*dx,ball.y+u*dy))
            return space
        best = max(targets,key=clearance)
        previous = self._clear_aim
        if (previous is not None and self._clear_origin is not None
                and dist(ball.x,ball.y,*self._clear_origin) < GOAL_CLEAR_AIM_RESET_M
                and self.context.now < self._clear_until
                and clearance(previous) >= GOAL_CLEAR_BLOCKED_M
                and clearance(previous)+GOAL_CLEAR_SWITCH_ADVANTAGE_M >= clearance(best)):
            return previous
        self._clear_aim,self._clear_origin = best,(ball.x,ball.y)
        self._clear_until = self.context.now+GOAL_CLEAR_AIM_HOLD_S
        return best

    def play_ball(self,intercept,target,power):
        ball,pose = self.context.ball,self.pose
        if ball is None or pose is None:
            self.stop()
            return
        d = dist(pose.x,pose.y,ball.x,ball.y)
        speed = self.ball_motion.speed if self.ball_motion is not None else 0.0
        if speed>RECEIVE_SPEED_MPS and d>(KICK_EXIT_M if self.is_kicking else BALL_KICK_ENGAGE_M):
            # 远处快球先拦截；近身与已启动任务保持 SDK 连续控制。
            self.walk_to(intercept or (ball.x,ball.y),avoid_robots=True,ball_access=True)
            return
        if d <= BALL_KICK_ENGAGE_M or (self.is_kicking and d <= KICK_EXIT_M):
            self.kick(angle_to(ball.x,ball.y,*target),power)
        else:
            approach = intercept or (ball.x,ball.y)
            if goal_danger(self.context):
                # 不为绕到球后站位而把目标放进球网；从场内侧接近活球。
                approach = (max(approach[0],-self.context.field.length/2+FIELD_SAFE_MARGIN_M),approach[1])
            else:
                approach = _behind_ball(*approach,target,CHASE_BEHIND_M)
            self.walk_to(approach,avoid_robots=True,ball_access=True)

    def defend_goal(self,target, *, allow_clear=True):
        """门前封堵不避开球；来球近身立即交给持续解围任务。"""
        ball,pose = self.context.ball,self.pose
        if ball is None or pose is None:
            self.stop()
            return
        reach = dist(pose.x,pose.y,ball.x,ball.y)
        motion = self.ball_motion
        incoming = (motion is not None and motion.speed > KEEPER_FAST_SHOT_MPS
                    and motion.velocity[0] < KEEPER_INCOMING_VX_MPS
                    and ball.x > pose.x+KEEPER_GOALSIDE_MARGIN_M)
        if allow_clear and reach <= KEEPER_CLEAR_REACH_M:
            self._goal_idle_pose = self._goal_idle_since = self._goal_shuffle = None
            self.kick(angle_to(ball.x,ball.y,*self.clearance_target()),SHOT_POWER)
        else:
            if (self._goal_idle_pose is None or self._goal_idle_since is None
                    or dist(pose.x,pose.y,*self._goal_idle_pose) >= INACTIVITY_PROGRESS_M
                    or reach > INACTIVITY_BALL_DISTANCE_M or incoming):
                self._goal_idle_pose,self._goal_idle_since = (pose.x,pose.y),self.context.now
                self._goal_shuffle = None
            if self.context.now-self._goal_idle_since >= KEEPER_IDLE_SHUFFLE_S:
                if self._goal_shuffle is None:
                    limit = self.context.field.goal_width/2-DEFEND_KEEPER_Y_MARGIN_M
                    side = -1.0 if target[1]>0 else 1.0
                    self._goal_shuffle = (target[0],max(-limit,min(limit,
                                           target[1]+side*KEEPER_IDLE_SHUFFLE_M)))
                target = self._goal_shuffle
            face = (angle_to(pose.x,pose.y,ball.x,ball.y)
                    if dist(pose.x,pose.y,*target)<=RETREAT_FACE_TRAVEL_M else None)
            self.walk_to(target,face=face,
                         avoid_ball=False,avoid_robots=True,ball_access=True)

    def recover_position(self,target):
        """远距离回防转身朝路线前进，接近防区后再面向球。"""
        pose,ball = self.pose,self.context.ball
        if pose is None or target is None:
            self.stop()
            return
        heading = angle_to(pose.x,pose.y,*target)
        if dist(pose.x,pose.y,*target) > RETREAT_FACE_TRAVEL_M:
            err = normalize_angle(heading-pose.theta)
            if abs(err) > RETREAT_TURN_FIRST_RAD:
                self.release_kick()
                self.set_velocity(0.,0.,self._angular(err))
                return
            face = heading
        else:
            face = angle_to(pose.x,pose.y,ball.x,ball.y) if ball is not None else heading
        self.walk_to(target,face=face,avoid_ball=True,avoid_robots=True)

    def receive_ball(self,target, *, incoming=True):
        """准备接应时避球，来球后迎接；确认控制后交回战术层选择传/射。"""
        ball,pose = self.context.ball,self.pose
        if ball is None or pose is None:
            self.stop()
            return
        # 慢球到脚边时不立刻低力度踢向远门：等待战术层确认控制后重新选择传/射。
        # 来球期间只迎接，避免接球动作本身把球再次踢离、控制估计永远无法稳定。
        self.walk_to(target,face=angle_to(pose.x,pose.y,ball.x,ball.y),
                     avoid_ball=not incoming,avoid_robots=True,ball_access=incoming)


    def block_path_projection(
        self, opponent_id: int,
    ) -> tuple[float, float, float, float] | None:
        """自己到 对手→球 线段的垂足:返回 (x, y, 垂距, 线段参数 t)。"""
        ctx = self.context
        pose = self.pose
        ball = ctx.ball if ctx is not None else None
        opponent = ctx.opponents.get(opponent_id) if ctx is not None else None
        if ctx is None or pose is None or ball is None or opponent is None:
            return None
        if opponent.pose is None:
            return None

        ax, ay = opponent.pose.x, opponent.pose.y
        bx, by = ball.x, ball.y
        vx, vy = bx - ax, by - ay
        length2 = vx * vx + vy * vy
        if length2 < 1e-6:
            return None

        raw_t = ((pose.x - ax) * vx + (pose.y - ay) * vy) / length2
        t = clamp(raw_t, 0.0, 1.0)
        tx = ax + vx * t
        ty = ay + vy * t
        return tx, ty, dist(pose.x, pose.y, tx, ty), raw_t

    def attack(self, kick_target: tuple[float, float] | None = None) -> None:
        """追球射门。踢向 ``kick_target``(默认对方球门中心)。

        踢球迟滞:距球 ≤ ENTER 进入踢球;进入后距球 > EXIT 才退出(EXIT > ENTER),
        防边界抖动。行走追球时避机器人，SDK 踢球阶段由 kick 统一限制。追球目标设在【球后方】(球→球门连线
        上、背离球门退 ``CHASE_BEHIND_M``),到位时天然对准射门方向。
        """
        ball = self.context.ball if self.context is not None else None
        if ball is None or self.pose is None:
            self.stop()
            return
        if kick_target is None:
            kick_target = opponent_goal(self.context)

        d = dist(self.pose.x, self.pose.y, ball.x, ball.y)
        self._kicking = d <= (KICK_EXIT_M if self._kicking else KICK_ENTER_M)
        if self._kicking:
            self.kick(angle_to(ball.x,ball.y,*kick_target), KICK_POWER_DEFAULT)
        else:
            self.release_kick()
            self.walk_to(
                _behind_ball(ball.x, ball.y, kick_target, CHASE_BEHIND_M),
                avoid_robots=True,
            )

    def guard(self) -> None:
        """守门:站小禁区中央待命

        无 pose(未就位)时退回站小禁区中央。
        """
        home = own_goal_area_center(self.context) if self.context is not None else None
        if home is None or self.pose is None:
            self.action = "guard:stop"
            self.stop()
            return

        ball = self.context.ball if self.context is not None else None
        home = (max(home[0],-self.context.field.length/2
                    +self.context.field.goal_area_length+FIELD_SAFE_MARGIN_M),home[1])

        # 待命朝向:朝球(便于快速反应);球不可见则朝对方门方向(默认 0)。
        face = 0.0
        if GUARD_FACE_BALL and ball is not None:
            face = angle_to(
                self.pose.x, self.pose.y, ball.x, ball.y,
            )

        self.action = "guard:home"
        debugdraw.point(
            home[0], home[1], rgb=(0.0, 0.6, 1.0), scale=0.2, ns="guard_home",
        )
        self.wait_away(home)

    def support(self) -> None:
        """支援:站在 球→己方门中心 连线上距球 ``SUPPORT_DIST_M`` 处补防。

        站位点在球与己方门之间的封堵线上。
        """
        ctx = self.context
        if ctx is None or ctx.ball is None:
            self.stop()
            return

        ball = self.context.ball
        gx, gy = own_goal(ctx)
        bx, by = (ball.x, ball.y)
        dx, dy = gx - bx, gy - by
        d = math.hypot(dx, dy)
        if d < 1e-6:
            ux, uy = -1.0, 0.0
        else:
            ux, uy = dx / d, dy / d
        along = min(SUPPORT_DIST_M, d)
        tx = bx + ux * along
        ty = by + uy * along

        # 不越过己方底线(x >= 门线 + 0.3),横向夹进场内
        half_l = ctx.field.length / 2.0
        half_w = ctx.field.width / 2.0 - 0.3
        tx = clamp(tx, -half_l + 0.3, half_l)
        ty = clamp(ty, -half_w, half_w)
        self.wait_away((tx, ty))

    def wait_away(self, target) -> None:
        """合法等待点离球 3 m 以外；不是用微小抖动规避无活动。"""
        ball = self.context.ball if self.context is not None else None
        if ball is None:
            self.stop()
            return
        discs = [*self.rule_discs, Obstacle(ball.x,ball.y,WAIT_BALL_DISTANCE_M)]
        target = legal_target(self.context,target,discs,self.rule_own_half)
        self.rule_discs = discs  # 本帧整段行走也避让，下一帧由状态机重置
        self.move_to_position(target)

    def recover_inactivity(self, *, emergency=False) -> bool:
        """仅用本地时钟防卡住；不用于球权开放、重启或返场判断。"""
        ctx,pose = self.context,self.pose
        ball = ctx.ball if ctx is not None else None
        if not self._motion_allowed() or ctx.game.state != GameState.PLAYING or pose is None or ball is None:
            self._activity_anchor = self._activity_since = None
            self._retreating = False
            self._rescue_target = None
            return False
        near = dist(pose.x,pose.y,ball.x,ball.y) <= INACTIVITY_BALL_DISTANCE_M
        if not near:
            self._activity_anchor = (pose.x,pose.y)
            self._activity_since = ctx.now
            self._retreating = False
            self._rescue_target = None
            return False
        if (self._activity_anchor is None or self._activity_since is None
                or ctx.now < self._activity_since
                or dist(pose.x,pose.y,*self._activity_anchor) >= INACTIVITY_PROGRESS_M
                or (emergency and self._activity_ball is not None
                    and dist(ball.x,ball.y,*self._activity_ball) >= INACTIVITY_PROGRESS_M)):
            self._activity_anchor = (pose.x,pose.y)
            self._activity_since = ctx.now
            self._activity_ball = (ball.x,ball.y)
            if emergency:
                # 已完成真实换位即可重新参与救险，不一直退到球外 3.6 m。
                self._retreating = False
                self._rescue_target = None
        stall_limit = (KICK_TASK_STALL_S if emergency and self.is_kicking
                       else ACTIVE_STALL_S if emergency else INACTIVITY_RECOVERY_S)
        if ctx.now-self._activity_since >= stall_limit:
            if not self._retreating:
                _log.info("player %d inactivity retreat: pose=(%.2f,%.2f) ball=(%.2f,%.2f)",
                          self.id,pose.x,pose.y,ball.x,ball.y)
            self._retreating = True
        if self._retreating:
            if emergency:
                if self._rescue_target is None:
                    obstacles = collect_obstacles(ctx,self.id,ball=False,robots=True,goals=True)
                    candidates = []
                    for i in range(RULE_ANGLE_SAMPLES):
                        a = math.tau*i/RULE_ANGLE_SAMPLES
                        t = legal_target(ctx,(pose.x+ACTIVE_REPOSITION_M*math.cos(a),
                                              pose.y+ACTIVE_REPOSITION_M*math.sin(a)),self.rule_discs)
                        if (t is not None and dist(pose.x,pose.y,*t) >= ACTIVE_REPOSITION_M/2
                                and dist(ball.x,ball.y,*t) >= ACTIVE_REPOSITION_MIN_BALL_M):
                            clear = segment_clear((pose.x,pose.y),t,obstacles,GLOBAL_OBSTACLE_MARGIN_M)
                            approach = _behind_ball(ball.x,ball.y,opponent_goal(ctx),CHASE_BEHIND_M)
                            candidates.append((clear,-dist(*approach,*t),t))
                    if candidates:
                        self._rescue_target = max(candidates,key=lambda c:(c[0],c[1]))[2]
                self.action = "goal:reposition" if goal_danger(ctx) else "unstick:reposition"
                self.routing_blocked_until = ctx.now+NAV_ROLE_ROTATE_S
                if self._rescue_target is not None:
                    self.walk_to(self._rescue_target,avoid_robots=True)
                else:
                    self.stop()
                return True
            self.action = "unstick:retreat"
            self.wait_away((pose.x,pose.y))
        return self._retreating

    def take_kickoff(self, kick_target: tuple[float, float] | None = None) -> None:
        """我方开球/重开:未就位则绕到球后待命(避球不碰),就位后接近并踢。"""
        ball = self.context.ball if self.context is not None else None
        if ball is None or self.pose is None:
            self.stop()
            return
        if kick_target is None:
            kick_target = opponent_goal(self.context)
        kick_dir = angle_to(ball.x, ball.y, *kick_target)
        cos_k, sin_k = math.cos(kick_dir), math.sin(kick_dir)
        rel_x, rel_y = self.pose.x - ball.x, self.pose.y - ball.y
        behind = rel_x * cos_k + rel_y * sin_k          # <0 在球后方(己方侧)
        lateral = abs(-rel_x * sin_k + rel_y * cos_k)
        if behind > KICKOFF_FRONT_MARGIN or lateral > KICKOFF_LATERAL_TOL:
            stage = (
                ball.x - cos_k * KICKOFF_STAGE_M,
                ball.y - sin_k * KICKOFF_STAGE_M,
            )
            self.release_kick()
            self.walk_to(stage, face=kick_dir, avoid_ball=True)
        else:
            self.attack(kick_target)

    def move_to_position(self, target: tuple[float, float] | None) -> None:
        """走到站位点(支援/防守/避让),面向球,避球+避机器人。"""
        if target is None:
            self.stop()
            return
        face = None
        ball = self.context.ball if self.context is not None else None
        if ball is not None and self.pose is not None:
            face = angle_to(self.pose.x, self.pose.y, ball.x, ball.y)
            if dist(self.pose.x,self.pose.y,*target) > RETREAT_FACE_TRAVEL_M:
                face = None  # 远途朝路线行走，接近支援点后再面向球。
        self.release_kick()
        self.walk_to(target, face=face, avoid_ball=True, avoid_robots=True)

    @staticmethod
    def _angular(err: float) -> float:
        return clamp(ANGULAR_GAIN * err, -MAX_ANGULAR, MAX_ANGULAR)
