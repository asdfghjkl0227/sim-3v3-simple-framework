"""SoccerSim 策略入口 —— 比赛策略主逻辑都在这里,改打法就改这个文件。

结构(由浅入深):
- main.py(本文件):比赛策略。play() 按 Phase 状态机分派到 _act_*;各 _act_* 选出
  attacker(离球最近)并直接调 player 动作。
- player.py:Player 控制 handle + 高层动作(attack / take_kickoff /
  move_to_position / walk_to);想加拐棍/技术动作直接改它。
- utils/:走位/几何/避障工具(opponent_goal / dist / angle_to ...)。
- framework/:平台管线,用户不改。

改打法主要改本文件:Phase 状态机、各 _act_* 行为、站位公式。
"""

from __future__ import annotations

import logging
import math
from enum import Enum

from booster_agent_framework import AgentBase

from .framework.agent import SoccerAgentMixin
from .framework.types import KICKING_TEAM_NONE, Context, GameState, SetPlay
from .param import *
from .player import Player
from .utils import dist, opponent_goal, own_goal
from .utils.geom import angle_to
from .utils.obstacles import Obstacle
from .utils.ball_motion import BallMotion
from .utils.rules import legal_target, goal_danger
from .tactics import TeamTactics
from .evaluation import MatchEvaluation


_log = logging.getLogger(__name__)


# ======================================================================
# Phase 状态机 —— 比赛阶段分类
# ======================================================================


class Phase(Enum):
    """比赛阶段。顶层状态机,决定当前是正常拼抢/开球/定位球/准备/停止。"""
    NORMAL = "normal"              # PLAYING 正常拼抢
    OUR_KICKOFF = "our_kickoff"    # 我方开球(SET+PLAYING 初期,take_kickoff)
    OPP_KICKOFF = "opp_kickoff"    # 对方开球(避让)
    OUR_SET_PLAY = "our_set_play"  # 我方定位球(任意球/角球/球门球)
    OPP_SET_PLAY = "opp_set_play"  # 对方定位球(避让)
    READY = "ready"                # READY 走位
    STOPPED = "stopped"            # SET(非开球重开) / INITIAL / FINISHED / stopped


def get_phase(context: Context) -> Phase:
    """根据裁判机状态判断当前比赛阶段。"""
    g = context.game
    if g is None or g.stopped:
        return Phase.STOPPED

    state = g.state

    # READY:走 ready 位
    if state == GameState.READY:
        return Phase.READY

    # PLAYING:正常拼抢 or 开球/定位球执行中
    if state == GameState.PLAYING and not g.stopped:
        if g.set_play != SetPlay.NONE and g.kicking_team == KICKING_TEAM_NONE:
            return Phase.STOPPED  # 未知主罚方不冒然抢球
        # 定位球:set_play != NONE,kicking_team 指示哪方
        if g.set_play != SetPlay.NONE and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_SET_PLAY
            else:
                return Phase.OPP_SET_PLAY

        # 开球:secondary_time > 0(倒计时窗口),kicking_team 指示哪方
        if g.secondary_time > 0 and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_KICKOFF
            else:
                return Phase.OPP_KICKOFF

        # 正常拼抢
        return Phase.NORMAL

    # SET / INITIAL / FINISHED / stopped:站定
    return Phase.STOPPED

def get_set_play_type(context: Context) -> SetPlay:
    """当前生效的定位球类型;无定位球(或无裁判机数据)时返回 ``SetPlay.NONE``。

    直接读裁判机的 ``set_play`` 字段,不区分是哪方主罚 —— 哪方由 :func:`get_phase`
    (OUR_SET_PLAY / OPP_SET_PLAY)判定。这里只回答"是什么类型的定位球"。

    共 7 种可能返回值(见 framework.types.SetPlay):
    - ``NONE``:无定位球(正常比赛/开球等)
    - ``DIRECT_FREE_KICK``:直接任意球(枚举名称不豁免本策略的 G-03 闸门)
    - ``INDIRECT_FREE_KICK``:间接任意球(须先触碰他人才能进球)
    - ``PENALTY_KICK``:点球
    - ``THROW_IN``:界外球(踢入)
    - ``GOAL_KICK``:球门球
    - ``CORNER_KICK``:角球
    """
    g = context.game
    if g is None:
        return SetPlay.NONE
    return g.set_play


# ======================================================================
# Agent 入口
# ======================================================================


class SoccerSimAgent(SoccerAgentMixin, AgentBase):
    """3v3 SoccerSim agent。"""

    player_class = Player

    def init_store(self, store) -> None:
        _log.info("init_store called")
        store.prev_phase = None       # 上一帧 phase,用于检测 phase 跳变(边沿)
        store.cur_phase = None
        store.kickoff_taker = None    # 锁定的开球主罚球员 id(每次进入开球时重选)
        store.normal_attacker = None
        store.restart_key = None
        store.restart_origin = None
        store.shot_origin = None
        store.ball_motion = BallMotion()
        store.tactics = TeamTactics()
        store.evaluation = MatchEvaluation()
        store.tactics.evaluation = store.evaluation
        store.penalized_ids = set()
        store.active_ids = set()
        store.ready_targets = {}
        store.last_audit = None

    @staticmethod
    def play(context: Context, players: list[Player], store) -> None:
        phase = get_phase(context)
        store.prev_phase = store.cur_phase
        store.cur_phase = phase
        store.evaluation.update(context)
        store.ball_motion.update(context)
        _update_restart(context, store)
        _configure_rules(context, players, store)
        penalized_ids = {p.id for p in players if p.is_penalized}
        returned_ids = store.penalized_ids-penalized_ids
        store.penalized_ids = penalized_ids
        phase_changed = store.prev_phase != phase
        if phase_changed and phase != Phase.NORMAL:
            store.tactics.reset()
            store.ready_targets = {}
        for p in players:
            if phase_changed or p.id in returned_ids:
                p.release_kick()
                p.reset_navigation()
        if returned_ids:
            # 不继承罚下前的持球/传球意图；保留现有后卫，返场者立即补空位。
            store.tactics.pass_plan = None
            store.normal_attacker = None

        # 画可视化(每帧)
        _analyze_and_draw(context, players, store)

        # 当前 phase 以 label 画在场外。
        from .framework import debugdraw
        g = context.game
        game_state = g.state.value if g is not None else "none"
        set_play = g.set_play.value if g is not None else "none"
        secondary_time = g.secondary_time if g is not None else 0.0
        debugdraw.text(
            0.0, context.field.width / 2.0 + 0.2,
            f"phase={phase.value} state={game_state} set={set_play} secondary={secondary_time:.1f}",
            rgb=(1.0, 1.0, 0.0), ns="phase",
        )

        # 活性自理 + 过滤出本帧可行动的球员。
        # 先停止/罚下闸门，再允许起身或切模式；未就绪球员不参与角色分配。
        active: list[Player] = []
        for p in players:
            if phase == Phase.STOPPED or p.is_penalized:
                p.action = "penalized" if p.is_penalized else "stopped"
                p.stop()
                p._activity_anchor = p._activity_since = None
                p._retreating = False
            elif p.pose is None:
                p.action = "no_pose"       # 自己位置未知:不参与分派(下游按 pose 已知处理)
                p.stop()
            elif not p.ensure_ready():
                p.action = "fallen" if p.is_fallen else "switching_mode"
            elif p.recover_inactivity(emergency=phase in
                    (Phase.NORMAL,Phase.OUR_KICKOFF,Phase.OUR_SET_PLAY)):
                pass
            else:
                active.append(p)
        new_ids = {p.id for p in active}
        if new_ids != store.active_ids:
            for p in active:
                if p.id not in store.active_ids:
                    p.reset_navigation()
            store.tactics.pass_plan = None
            store.tactics.attacker_id = None
        store.active_ids = new_ids

        if (phase in (Phase.OUR_KICKOFF,Phase.OUR_SET_PLAY) and active
                and store.kickoff_taker not in new_ids):
            # 原主罚正在起身或换位时，让可行动队友接替，不能所有人继续等他。
            store.kickoff_taker = _select_closest_attacker(context,active).id
            _configure_rules(context,players,store)

        # 按 phase 对整队分派一次(角色分配等全队计算只在 _act_* 里算一次)。
        if phase == Phase.NORMAL:
            _act_normal(context, active, store)
        elif phase == Phase.OUR_KICKOFF:
            _clear_normal_sticky(store)
            _act_our_kickoff(context, active, store)
        elif phase == Phase.OPP_KICKOFF:
            _clear_normal_sticky(store)
            _act_opp_kickoff(context, active)
        elif phase == Phase.OUR_SET_PLAY:
            _clear_normal_sticky(store)
            _act_our_set_play(context, active, store)
        elif phase == Phase.OPP_SET_PLAY:
            _clear_normal_sticky(store)
            _act_opp_set_play(context, active, store)
        elif phase == Phase.READY:
            _clear_normal_sticky(store)
            _act_ready(context, active, store)
        elif phase == Phase.STOPPED:
            _clear_normal_sticky(store)
            for p in active:
                p.action = "stopped"
                p.stop()

        # 队员可视化统一在最后画一遍:覆盖所有球员(含判罚/未就绪/STOPPED),
        # 修复 SET 等状态下红球/标签消失的问题。
        for p in players:
            _draw_teammate_marker(p)
        if store.last_audit is None or context.now-store.last_audit >= TACTIC_AUDIT_INTERVAL_S:
            store.last_audit = context.now
            rows = []
            for p in players:
                pose = p.pose
                if pose is not None:
                    rows.append('%d:(%.2f,%.2f) %s k=%s mode=%s fall=%s recovery=%s' %
                                (p.id,pose.x,pose.y,p.action,p.is_kicking,
                                 p.mode,p.fall_down_state,p._recovering))
            ball = context.ball
            _log.info('AUDIT phase=%s ball=%s speed=%.2f players=%s',phase.value,
                      None if ball is None else (round(ball.x,2),round(ball.y,2)),
                      store.ball_motion.speed,'; '.join(rows))


def _clear_normal_sticky(store) -> None:
    store.normal_attacker = None


def _update_restart(context, store):
    """保留摆球点跨越球权开放；不把计时结束当作 G-03 豁免。"""
    g = context.game
    if g is None:
        return
    if g.state in (GameState.INITIAL, GameState.READY, GameState.FINISHED):
        store.restart_key = store.restart_origin = store.shot_origin = None
        store.kickoff_taker = None
        return
    restart = store.cur_phase in (Phase.OUR_KICKOFF, Phase.OPP_KICKOFF,
                                  Phase.OUR_SET_PLAY, Phase.OPP_SET_PLAY)
    if not restart:
        store.restart_key = None
        return
    key = (g.kicking_team, g.set_play)
    if key != store.restart_key or store.prev_phase == Phase.STOPPED:
        store.restart_key = key
        store.restart_origin = None
        store.shot_origin = None
        store.kickoff_taker = None
    if store.restart_origin is None and context.ball is not None:
        store.restart_origin = (context.ball.x,context.ball.y)
        # 任何一方重启都保守遵守同一射门距离条件，不猜测谁已触球。
        store.shot_origin = store.restart_origin


def _configure_rules(context, players, store):
    phase = store.cur_phase
    ours = phase in (Phase.OUR_KICKOFF,Phase.OUR_SET_PLAY)
    available = [p for p in players if p.pose is not None and not p.is_penalized
                 and not p.is_fallen and p.mode == "walk"]
    if ours and available and store.kickoff_taker not in {p.id for p in available}:
        store.kickoff_taker = _select_closest_attacker(context,available).id
    for p in players:
        p.ball_access = False
        p.rule_discs = []
        p.rule_own_half = phase in (Phase.READY,Phase.OUR_KICKOFF,Phase.OPP_KICKOFF)
        p.rule_can_kick = phase == Phase.NORMAL or (
            ours and p.id == store.kickoff_taker and store.restart_origin is not None)
        p.rule_shot_origin = store.shot_origin
        p.ball_motion = store.ball_motion
        p.evaluation = store.evaluation
        if phase in (Phase.OPP_KICKOFF,Phase.OPP_SET_PLAY):
            if store.restart_origin is not None:
                ox,oy = store.restart_origin
                p.rule_discs.append(Obstacle(ox,oy,OPPONENT_RESTART_AVOID_M))
            if context.ball is not None:
                p.rule_discs.append(Obstacle(context.ball.x,context.ball.y,OPPONENT_RESTART_AVOID_M))
        if (phase == Phase.OPP_KICKOFF
                or (phase == Phase.OUR_KICKOFF and p.id != store.kickoff_taker)
                or phase == Phase.READY):
            # READY 全员圈外，避免队员不可用导致中圈人数变化。
            p.rule_discs.append(Obstacle(0.0,0.0,context.field.circle_radius+CIRCLE_MARGIN_M))


def _player_dist_to_ball(context: Context, p: Player) -> float:
    """球员到球当前位置的距离。"""
    ball = context.ball
    pose = p.pose
    return (
        dist(pose.x, pose.y, ball.x, ball.y) + _fallen_time_cost(p)
        if ball is not None and pose is not None else math.inf
    )


def _fallen_time_cost(p: Player) -> float:
    return FALLEN_COST if p.is_fallen else 0.0


def _select_closest_attacker(
    context: Context,
    players: list[Player],
    preferred_id: int | None = None,
) -> Player:
    """选到球距离最小的球员。

    ``players`` 非空、已就绪、pose 已知。normal 与开球共用。
    """
    ranked = [(p, _player_dist_to_ball(context, p)) for p in players]
    best, best_dist = min(ranked, key=lambda item: item[1])
    preferred = next((item for item in ranked if item[0].id == preferred_id), None)
    if (
        preferred is not None
        and preferred[1] <= best_dist + ATTACKER_KEEP_DIST_MARGIN_M
    ):
        return preferred[0]
    return best


def _act_normal(context: Context, players: list[Player], store) -> None:
    """NORMAL:距球最近者 attack,剩下人里离己方门最近者 guard,其余 support。

    ``players`` 是本帧可行动球员(已就绪、pose 已知),这里直接挑角色并执行。
    """
    if not players:
        return

    if context.ball is None:
        for p in players:
            p.stop()
        return

    intents = store.tactics.plan(context,players,store.ball_motion)
    for p in players:
        intent = intents.get(p.id)
        if intent is None:
            p.stop()
            continue
        p.action = intent.role
        if intent.receive:
            p.receive_ball(intent.target)
        elif intent.kick_target is not None:
            p.play_ball(intent.target,intent.kick_target,intent.power)
        elif intent.role in ('cover:goal','mark:goal_cover','protect:goal'):
            p.defend_goal(intent.target)
        else:
            p.move_to_position(intent.target)


def _act_our_kickoff(context: Context, players: list[Player], store) -> None:
    _act_our_restart(context, players, store)


def _act_our_restart(context, players, store):
    if context.ball is None:
        for p in players:
            p.stop()
        return
    support = [p for p in players if p.id != store.kickoff_taker]
    keeper = min(support,key=lambda p:dist(p.pose.x,p.pose.y,-context.field.length/2,0.)) if support else None
    for p in players:
        if p.id == store.kickoff_taker:
            p.action = "restart:push"
            # SDK 自主接近定位球，避免先走局部接近点再反复取消任务。
            p.kick(angle_to(context.ball.x,context.ball.y,
                            context.field.length/2,0.),KICK_POWER_OUR_KICKOFF)
        else:
            p.action = "restart:wait"
            if p is keeper:
                target = (-context.field.length/2+DEFEND_KEEPER_LINE_M,0.)
            elif store.cur_phase == Phase.OUR_KICKOFF:
                target = (READY_PRESS_X_M,(p.id % 2 * 2-1)*WAIT_BALL_DISTANCE_M)
            else:
                side = -1.0 if context.ball.y >= 0 else 1.0
                target = (context.ball.x-RESTART_SUPPORT_DEPTH_M,
                          context.ball.y+side*WAIT_BALL_DISTANCE_M)
            p.wait_away(target)


def _act_opp_kickoff(context: Context, players: list[Player]) -> None:
    _wait_restart(context, players)


def _act_our_set_play(context: Context, players: list[Player], store) -> None:
    _act_our_restart(context, players, store)


def _act_opp_set_play(context: Context, players: list[Player], store) -> None:
    _wait_restart(context, players)


def _wait_restart(context, players):
    keeper = min(players,key=lambda p:dist(p.pose.x,p.pose.y,-context.field.length/2,0.)) if players else None
    for p in players:
        p.action = "opponent_restart:retreat"
        target = (-context.field.length/2+DEFEND_KEEPER_LINE_M,0.) if p is keeper else (
            -context.field.length/2+context.field.penalty_area_length,
            (p.id % 2*2-1)*context.field.width/3)
        p.wait_away(target)


def _act_ready(context: Context, players: list[Player], store) -> None:
    # 阵型压到中场；分配时让最近者负责后点，另两人占中线两侧。
    if players and not store.ready_targets:
        from itertools import permutations
        targets = [(READY_BASE_X_M,0.),(READY_PRESS_X_M,-READY_PRESS_WING_M),
                   (READY_PRESS_X_M,READY_PRESS_WING_M)][:len(players)]
        order = sorted(players,key=lambda p:p.id)
        positions = {p.id:pose for p in order if (pose := p.pose) is not None}
        assignment = min(permutations(targets),key=lambda ts:sum(
            dist(positions[p.id].x,positions[p.id].y,*t)
            for p,t in zip(order,ts) if p.id in positions))
        store.ready_targets = {p.id:t for p,t in zip(order,assignment)}
    for p in sorted(players,key=lambda p:p.id):
        if p.pose is None:
            p.stop()
            continue
        if p.id not in store.ready_targets:
            pose = p.pose
            target = (min(pose.x,READY_RETURN_X_M),pose.y)
            if pose.x > -HALF_SAFE_MARGIN_M and abs(pose.y) < READY_RETURN_LANE_M:
                side = 1.0 if pose.y > 0 else -1.0
                target = (READY_RETURN_X_M,side*READY_RETURN_LANE_M)
            occupied = [Obstacle(x,y,READY_TEAM_SEPARATION_M)
                        for pid,(x,y) in store.ready_targets.items() if pid != p.id]
            target = legal_target(context,target,[*p.rule_discs,*occupied],True)
            if target is not None:
                store.ready_targets[p.id] = target
        p.action = "ready:midfield"
        target = store.ready_targets.get(p.id)
        if target is None:
            p.stop()
        else:
            # 优先从界外/对方半场/中圈撤到最近合法点，再压到中场阵型。
            home = legal_target(context,(p.pose.x,p.pose.y),p.rule_discs,True)
            if home is not None and dist(p.pose.x,p.pose.y,*home)>ARRIVE_DIST:
                p.action = 'ready:legal_first'
                p.walk_to(home,face=0.0,avoid_robots=True)
            else:
                p.walk_to(target,face=0.0,avoid_robots=True)


# ======================================================================
# 战场可视化 —— 显示球位置 + 球员到球的距离,画到 ROS 可视化
# ======================================================================

def _draw_teammate_marker(p: Player) -> None:
    """我方队员可视化:红色。踢球中→方块,否则→球体。

    每帧对所有球员统一调用(不受 phase/判罚/就绪影响)。标签两行:
    - 上:编号 + 当前高层动作(``p.action``),踢球中追加 ``[KICK]``。
    - 通过形状(方块 vs 球体)再次区分是否进入 kick 状态。
    """
    from .framework import debugdraw

    if p.pose is None:
        return
    red = (1.0, 0.2, 0.2)
    if p.is_kicking:
        debugdraw.cube(p.pose.x, p.pose.y, rgb=red, scale=0.38, ns="teammate")
    else:
        debugdraw.point(p.pose.x, p.pose.y, rgb=red, scale=0.3, ns="teammate")
    kick_tag = " [KICK]" if p.is_kicking else ""
    label = f"{p.id}:{p.action}{kick_tag}"
    debugdraw.text(p.pose.x, p.pose.y, label, rgb=(1.0, 0.9, 0.6), ns="teammate_id")


def _analyze_and_draw(context: Context, players: list[Player], store) -> None:
    """每帧:计算球员到球的距离,画可视化。

    不再依赖 analysis 模块;距离改为基于球当前位置。
    """
    from .framework import debugdraw

    ball = context.ball

    # 球不可见:无可视化
    if ball is None:
        return

    # 1. 画球当前位置(绿色点)
    debugdraw.point(ball.x, ball.y, rgb=(0.0, 1.0, 0.0), scale=0.2, ns="ball_current")

    # 2. 球员到球的距离:我方(红标签)+ 敌方(蓝标签)
    for p in players:
        if p.pose is None:
            continue
        d = dist(p.pose.x, p.pose.y, ball.x, ball.y)
        debugdraw.text(
            p.pose.x + 0.3, p.pose.y - 0.3, f"{d:.1f}m",
            rgb=(1.0, 0.6, 0.6), ns="dist_ours",
        )
    for r in context.opponents.values():
        if r.pose is None:
            continue
        d = dist(r.pose.x, r.pose.y, ball.x, ball.y)
        debugdraw.text(
            r.pose.x + 0.3, r.pose.y - 0.3, f"{d:.1f}m",
            rgb=(0.6, 0.6, 1.0), ns="dist_opp",
        )

