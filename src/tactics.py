"""三人制战术规划。输出意图，实际动作仍走 Player 的规则闸门。"""
from dataclasses import dataclass
import logging
import math
from itertools import permutations
from typing import TypedDict

from .param import *
from .evaluation import MatchEvaluation
from .utils.rules import legal_target, goal_danger

_log = logging.getLogger(__name__)


def distance(a,b):
    return math.hypot(a[0]-b[0],a[1]-b[1])


def point_of(player):
    return player.pose.x,player.pose.y


def lane_clearance(start,end,robots):
    dx,dy = end[0]-start[0],end[1]-start[1]
    length2 = dx*dx+dy*dy
    clearance = math.inf
    for robot in robots:
        if robot.pose is None:
            continue
        t = ((robot.pose.x-start[0])*dx+(robot.pose.y-start[1])*dy)/length2 if length2 else 0.0
        if 0.0 <= t <= 1.0:
            clearance = min(clearance,math.hypot(robot.pose.x-start[0]-t*dx,
                                               robot.pose.y-start[1]-t*dy))
    return clearance


@dataclass(frozen=True)
class Intent:
    role: str
    target: tuple[float,float]
    kick_target: tuple[float,float] | None = None
    power: float = SHOT_POWER
    receive: bool = False


class PassPlan(TypedDict):
    passer: int
    receiver: int
    target: tuple[float,float]
    power: float
    origin: tuple[float,float]
    launched: bool
    until: float
    boundary: bool


class TeamTactics:
    def __init__(self):
        self.mode = 'loose'
        self.possession = 'loose'
        self.possession_at = 0.0
        self.counterpress_until = 0.0
        self.transition_active = False
        self.kickoff_follow_until = 0.0
        self.kickoff_receiver_id = None
        self.kickoff_taker_id = None
        self.opponent_restart_press_until = 0.0
        self.opponent_kickoff_defense = False
        self.attacker_id = None
        self.attacker_since = 0.0
        self.anchor_id = None
        self.shot_aim = None
        self.shot_aim_origin = None
        self.shot_aim_until = 0.0
        self.pass_plan: PassPlan | None = None
        self.last_passer = None
        self.run_until = 0.0
        self.last_positions = {}
        self.velocities = {}
        self.last_update = None
        self._last_mode_log = None
        self.mark_assignments = {}
        self.evaluation: MatchEvaluation | None = None

    def reset(self):
        self.pass_plan = None
        self.attacker_id = None
        self.anchor_id = None
        self.shot_aim = None
        self.shot_aim_origin = None
        self.shot_aim_until = 0.0
        self.possession = 'loose'
        self.counterpress_until = 0.0
        self.transition_active = False
        self.kickoff_follow_until = 0.0
        self.kickoff_receiver_id = None
        self.kickoff_taker_id = None
        self.opponent_restart_press_until = 0.0
        self.opponent_kickoff_defense = False
        self.last_positions.clear()
        self.velocities.clear()
        self.last_update = None
        self.mark_assignments.clear()
        self.last_passer = None
        self.run_until = 0.0

    def _update_possession(self,context,players):
        ball = (context.ball.x,context.ball.y)
        our = min((distance(point_of(p),ball) for p in players),default=math.inf)
        their = min((distance((r.pose.x,r.pose.y),ball) for r in context.opponents.values()
                     if r.pose is not None),default=math.inf)
        candidate = 'loose'
        if our < POSSESSION_REACH_M and our+POSSESSION_ADVANTAGE_M < their:
            candidate = 'ours'
        elif their < POSSESSION_REACH_M and their+POSSESSION_ADVANTAGE_M < our:
            candidate = 'theirs'
        previous = self.possession
        if candidate != 'loose' or context.now-self.possession_at > POSSESSION_HOLD_S:
            self.possession = candidate
        if candidate != 'loose':
            self.possession_at = context.now
        if previous == 'ours' and self.possession == 'theirs':
            self.counterpress_until = context.now+COUNTERPRESS_S
            self.transition_active = True
            self.kickoff_follow_until = 0.0
            self.pass_plan = None
            self.last_passer = None
            self.run_until = 0.0
            # 失球取消传球计划，但保留原处理者候选，避免球权估计抖动打断近身抢球。
        elif candidate == 'ours':
            self.transition_active = False
            self.counterpress_until = 0.0

    def start_kickoff_followup(self,context,receiver_id=None,taker_id=None):
        """官方已放行后保留近球主罚的连续任务，队友明显更有利时才接替。"""
        self.kickoff_follow_until = context.now+KICKOFF_FOLLOW_S
        self.attacker_id = taker_id
        self.kickoff_receiver_id = receiver_id
        self.kickoff_taker_id = taker_id
        self.pass_plan = None

    def _loose_ball_race(self,context,players,motion,hold_keeper):
        """比较双方到运动球的预计时间，不把旧持球估计或回撤窗口当作争球禁令。"""
        if (self.possession == 'ours' and not self.transition_active) or goal_danger(context):
            return None
        if (context.now < self.kickoff_follow_until and not self.transition_active
                and self.possession != 'theirs'):
            return None
        if self.pass_plan is not None and self.possession != 'theirs' and not self.transition_active:
            return None
        ball = (context.ball.x,context.ball.y)
        opponents = [r for r in context.opponents.values() if r.pose is not None]
        if not opponents or min(distance(ball,(r.pose.x,r.pose.y)) for r in opponents) <= LOOSE_OPPONENT_CONTACT_M:
            return None  # 对手仍贴着球时，继续短反抢/封堵，而非误认为无人控制。
        their_time = min(motion.intercept(context,r.pose)[1] for r in opponents)
        options = []
        for p in players:
            if p.id == hold_keeper:
                continue
            point,cost = motion.intercept(context,p.pose)
            if point is None:
                continue
            if context.now < getattr(p,'routing_blocked_until',0.):
                cost += TACTIC_BLOCKED_COST_S
            options.append((cost,p,point))
        if not options:
            return None
        best = min(options,key=lambda item:item[0])
        preferred = next((item for item in options if item[1].id == self.attacker_id),None)
        if preferred is not None and preferred[0] <= best[0]+TACTIC_INTERCEPT_KEEP_S:
            best = preferred
        if best[0]+LOOSE_RACE_ADVANTAGE_S >= their_time:
            # 原人选稍慢时仍须允许真正能赢得争球的队友接替。
            best = min(options,key=lambda item:item[0])
        return (best[1],best[2]) if best[0]+LOOSE_RACE_ADVANTAGE_S < their_time else None

    def start_opponent_restart_pressure(self,context, *, kickoff=False):
        """仅由官方阶段切回NORMAL触发，不以球位移/本地时间猜测放行。"""
        self.opponent_kickoff_defense = kickoff
        self.opponent_restart_press_until = context.now+(OPP_KICKOFF_PRESS_S if kickoff else OPP_RESTART_PRESS_S)
        self.attacker_id = None
        self.pass_plan = None

    def _restart_pressure(self,context,players,keeper,motion):
        field = [p for p in players if p is not keeper]
        if not field:
            return None
        if self.opponent_kickoff_defense:
            return self._kickoff_central_defense(context,field,keeper,motion)
        presser = self._choose_tackler(context,field)
        ball = (context.ball.x,context.ball.y)
        result = {presser.id:self._tackle_intent(context,presser,motion),
                  keeper.id:Intent('cover:goal',self._keeper_target(context,motion))}
        for p in field:
            if p is presser:
                continue
            # 一人争球，另一人占持球者身后的另一侧，接替被突破者并封短传。
            side = -1.0 if presser.pose.y >= ball[1] else 1.0
            target = (ball[0]-OPP_RESTART_PRESS_DEPTH_M,
                      ball[1]+side*OPP_RESTART_PRESS_WIDTH_M)
            result[p.id] = Intent('defend:restart_cover',legal_target(context,target))
        return result

    def _kickoff_central_defense(self,context,field,keeper,motion):
        """前层争球、后层封中路；快球越过前层后让仍在球门侧者截击。"""
        ball = (context.ball.x,context.ball.y)
        eligible = [p for p in field if p.pose.x <= ball[0]+TACKLE_GOALSIDE_MARGIN_M]
        fast = (motion.speed >= OPP_KICKOFF_FAST_BALL_MPS
                and motion.velocity[0] < KEEPER_INCOMING_VX_MPS)
        candidates = eligible or field
        if fast:
            presser,intercept = self._choose_attacker(context,candidates,motion)
            # 接近球时恢复持续解围；远处快球先卡路线，不能慢慢绕到球后。
            intent = (Intent('defend:kickoff_intercept',intercept,receive=True)
                      if distance(point_of(presser),ball)>KEEPER_FOOT_REACH_M
                      else Intent('defend:kickoff_clear',ball,presser.clearance_target(),SHOT_POWER))
        else:
            presser = self._choose_tackler(context,candidates)
            intent = self._tackle_intent(context,presser,motion)
        result = {presser.id:intent,keeper.id:Intent('cover:goal',self._keeper_target(context,motion))}
        for p in field:
            if p is presser:
                continue
            goal = (-context.field.length/2,0.)
            length = max(distance(ball,goal),OPP_KICKOFF_SCREEN_DEPTH_M)
            x = max(goal[0]+KEEPER_ADVANCE_DEPTH_M+DEFEND_COVER_SEPARATION_M,
                    ball[0]+(goal[0]-ball[0])*OPP_KICKOFF_SCREEN_DEPTH_M/length)
            y = motion.crossing_y(x) if fast else None
            if y is None:
                y = ball[1]+(goal[1]-ball[1])*OPP_KICKOFF_SCREEN_DEPTH_M/length
            target = legal_target(context,(x,y))
            result[p.id] = Intent('defend:restart_cover',target,receive=fast)
        return result

    def _loose_clear_target(self,context):
        ball = (context.ball.x,context.ball.y)
        forward = min(context.field.length/2-FIELD_SAFE_MARGIN_M,
                      max(LOOSE_CLEAR_MIN_X_M,ball[0]+GOAL_CLEAR_FORWARD_M))
        side = 1.0 if ball[1] >= 0 else -1.0
        targets = [(forward,0.),(forward,side*TACTIC_CLEARANCE_Y_M),
                   (forward,-side*TACTIC_CLEARANCE_Y_M),
                   (ball[0]+BOUNDARY_RETURN_DEPTH_M,side*(context.field.width/2-FIELD_SAFE_MARGIN_M))]
        targets = [legal_target(context,t) for t in targets]
        # 通道优先，同样畅通时优先向对方半场；中路被封时允许向边线解围。
        return max(targets,key=lambda t:(lane_clearance(ball,t,context.opponents.values()),t[0]))

    @staticmethod
    def score_margin(context):
        our = context.game.get_team_state(context.team_id)
        opponents = [t for t in context.game.teams if t.team_number != context.team_id]
        if our is None or not opponents:
            return 0
        return our.score-max(t.score for t in opponents)

    @staticmethod
    def _protect_lead(context):
        return False  # 统一主动控球，不因比分收缩。

    def _choose_attacker(self,context,players,motion):
        costs = {}
        targets = {}
        for p in players:
            point,cost = motion.intercept(context,p.pose)
            targets[p.id] = point or (context.ball.x,context.ball.y)
            if context.now < getattr(p,'routing_blocked_until',0.0):
                cost += TACTIC_BLOCKED_COST_S
            costs[p.id] = cost
        best = min(players,key=lambda p:costs[p.id])
        preferred = next((p for p in players if p.id==self.attacker_id),None)
        if preferred is not None:
            ball = (context.ball.x,context.ball.y)
            handoff = (distance(point_of(best),ball) <= RECEIVE_REACH_M
                       and distance(point_of(best),ball)+TACKLE_SWITCH_ADVANTAGE_M
                           < distance(point_of(preferred),ball))
            if not handoff and (costs[preferred.id] <= costs[best.id]+TACTIC_INTERCEPT_KEEP_S or
                    (context.now-self.attacker_since < TACTIC_ROLE_KEEP_S and
                     context.now >= getattr(preferred,'routing_blocked_until',0.0))):
                best = preferred
        plan = self.pass_plan
        if plan is not None and self.possession != 'theirs':
            locked_id = plan['receiver'] if plan['launched'] else plan['passer']
            locked = next((p for p in players if p.id==locked_id),None)
            if locked is not None:
                # 球未离脚时锁传球人，离脚后锁接球人，避免SDK接近任务被互换角色打断。
                best = locked
        if best.id != self.attacker_id:
            self.attacker_since = context.now
            self.attacker_id = best.id
        return best,targets[best.id]

    def _update_teammate_velocity(self,context,players):
        dt = context.now-self.last_update if self.last_update is not None else 0.0
        for p in players:
            point = point_of(p)
            previous = self.last_positions.get(p.id)
            if previous is not None and BALL_SAMPLE_MIN_DT_S <= dt <= BALL_SAMPLE_MAX_GAP_S:
                vx,vy = (point[0]-previous[0])/dt,(point[1]-previous[1])/dt
                speed = math.hypot(vx,vy)
                if speed > BALL_WALK_SPEED_MPS:
                    vx,vy = vx*BALL_WALK_SPEED_MPS/speed,vy*BALL_WALK_SPEED_MPS/speed
                old = self.velocities.get(p.id,(0.0,0.0))
                self.velocities[p.id] = tuple((a+b)/2 for a,b in zip(old,(vx,vy)))
            self.last_positions[p.id] = point
        self.last_update = context.now

    def _update_pass(self,context,players,motion):
        plan = self.pass_plan
        if plan is None:
            return
        ids = {p.id for p in players}
        if context.now > plan['until'] or plan['receiver'] not in ids or plan['passer'] not in ids:
            self.pass_plan = None
            return
        ball = (context.ball.x,context.ball.y)
        if self.possession == 'theirs':
            self.pass_plan = None
            return
        travel = distance(ball,plan['origin'])
        dx,dy = plan['target'][0]-plan['origin'][0],plan['target'][1]-plan['origin'][1]
        length = math.hypot(dx,dy)
        if not plan['launched']:
            if lane_clearance(ball,plan['target'],context.opponents.values()) < PASS_ABORT_LANE_M:
                self.pass_plan = None
                return
        elif length:
            # 偏离/越过落点后立即重新争球，不为失效传球等待至固定超时。
            along = ((ball[0]-plan['origin'][0])*dx+(ball[1]-plan['origin'][1])*dy)/length
            lateral = abs((ball[0]-plan['origin'][0])*dy-(ball[1]-plan['origin'][1])*dx)/length
            if along > length+PASS_OVERSHOOT_M or lateral > PASS_OFFLINE_M:
                self.pass_plan = None
                return
        progress = ((ball[0]-plan['origin'][0])*dx+(ball[1]-plan['origin'][1])*dy)/max(length*travel,1e-6)
        if (not plan['launched'] and travel > PASS_LAUNCH_DISTANCE_M
                and motion.speed > PASS_LAUNCH_SPEED_MPS and progress > PASS_LAUNCH_DIRECTION_COS):
            # 只是传球路线的观测反馈，不用于 G-03 的官方触球判定。
            plan['launched'] = True
            # 慢球可滚动数秒；接球窗口取预计走球时间，不固定两秒超时。
            plan['until'] = context.now+max(PASS_FLIGHT_HOLD_S,
                length/max(PASS_ARRIVAL_SPEED_MPS,motion.speed)+PASS_RECEIVE_GRACE_S)
            if self.evaluation is not None:
                self.evaluation.start_pass(context,plan)
            self.last_passer = plan['passer']
            self.run_until = context.now+PASS_RUN_HOLD_S
            _log.info('TACTIC pass_motion passer=%d receiver=%d',plan['passer'],plan['receiver'])
        receiver = next((p for p in players if p.id==plan['receiver']),None)
        if (plan['launched'] and receiver is not None and self.evaluation is not None and
                self.evaluation.owner == ('ours',plan['receiver']) and
                self.evaluation.candidate == self.evaluation.owner and
                distance(point_of(receiver),ball)<=RECEIVE_REACH_M and motion.speed<=RECEIVE_SPEED_MPS):
            self.pass_plan = None

    def _goal_target(self,context):
        ball = (context.ball.x,context.ball.y)
        if distance(ball,(context.field.length/2,0.)) >= SHOT_CENTRE_DISTANCE_M:
            return (context.field.length/2,0.)
        y = context.field.goal_width/2-SHOT_TARGET_POST_MARGIN_M
        targets = [(context.field.length/2,gy) for gy in (0.0,-y,y)]
        clearance = {target:lane_clearance(ball,target,context.opponents.values()) for target in targets}
        best = max(targets,key=lambda target:clearance[target])
        previous = self.shot_aim
        if (previous in clearance and context.now < self.shot_aim_until
                and self.shot_aim_origin is not None
                and distance(ball,self.shot_aim_origin) < SHOT_AIM_RESET_M
                and clearance[previous]+SHOT_AIM_SWITCH_CLEARANCE_M >= clearance[best]):
            return previous
        if best != previous or context.now >= self.shot_aim_until:
            self.shot_aim,self.shot_aim_origin = best,ball
            self.shot_aim_until = context.now+SHOT_AIM_HOLD_S
        return best

    def _pass_option(self,context,handler,rest,motion, *, boundary=False):
        ball = (context.ball.x,context.ball.y)
        options = []
        for p in rest:
            if p.id == self.anchor_id:
                continue
            velocity = self.velocities.get(p.id,(0.0,0.0))
            flight = min(BALL_PREDICTION_MAX_S,distance(ball,point_of(p))/PASS_FLIGHT_ESTIMATE_SPEED_MPS)
            lead = (velocity[0]*flight,velocity[1]*flight)
            size = math.hypot(*lead)
            if size > PASS_LEAD_MAX_M:
                lead = tuple(v*PASS_LEAD_MAX_M/size for v in lead)
            target = legal_target(context,(p.pose.x+lead[0],p.pose.y+lead[1]))
            inset = context.field.width/2-BOUNDARY_RECEIVE_INSET_M
            target = (target[0],max(-inset,min(inset,target[1])))
            if distance(point_of(p),target) > PASS_READY_TOL_M:
                continue
            if boundary and ((ball[1]-target[1])*(1. if ball[1]>0 else -1.) < BOUNDARY_RECEIVE_INSET_M
                             or target[0] < ball[0]-FIELD_SAFE_MARGIN_M):
                continue
            receiver_space = min((distance(target,(r.pose.x,r.pose.y))
                                  for r in context.opponents.values() if r.pose is not None),default=math.inf)
            if receiver_space < PASS_RECEIVER_SPACE_M:
                continue
            length = distance(ball,target)
            if not PASS_MIN_DISTANCE_M <= length <= PASS_MAX_DISTANCE_M:
                continue
            clear = lane_clearance(ball,target,context.opponents.values())
            if clear < PASS_LANE_CLEARANCE_M:
                continue
            # 无把握时不在门前横传/回传，也不为了出界去瞄准对手身体。
            if target[0] < -context.field.length/2+context.field.penalty_area_length:
                continue
            forward = target[0]-ball[0]
            safety = min(clear,TRIANGLE_SPACING_M)
            # 用用户奖励比较成功接应与丢球/被断的期望收益，再计入推进价值。
            probability = min(1.0,clear/PASS_LANE_CLEARANCE_M/2)
            score = probability*REWARD_RECEIVE+(1-probability)*(REWARD_LOSS+REWARD_DISPOSSESSED)
            score += forward/SHOT_MAX_DISTANCE_M+safety/PASS_MAX_DISTANCE_M-length/PASS_MAX_DISTANCE_M
            if p.id == self.last_passer and target[0] <= ball[0]:
                score -= PASS_PROGRESS_MIN_M
            power = max(PASS_POWER_MIN,min(PASS_POWER_MAX,
                        motion.power_for_distance(length,PASS_ARRIVAL_SPEED_MPS)))
            # 比较球到通道中段的时间与对手赶到该点的时间，不只看静态垂距。
            speed = BALL_KICK_SPEED_INITIAL_MPS+BALL_KICK_SPEED_PER_POWER*power
            safe = True
            for fraction in PASS_INTERCEPT_FRACTIONS:
                travel = length*fraction
                residual = speed*speed-2*motion.deceleration*travel
                if residual < 0:
                    safe = False
                    break
                arrival = (speed-math.sqrt(residual))/motion.deceleration
                point = (ball[0]+(target[0]-ball[0])*fraction,
                         ball[1]+(target[1]-ball[1])*fraction)
                if any(max(0.,distance((r.pose.x,r.pose.y),point)-BALL_INTERCEPT_REACH_M)
                       /BALL_WALK_SPEED_MPS < arrival-PASS_RACE_MARGIN_S
                       for r in context.opponents.values() if r.pose is not None):
                    safe = False
                    break
            if not safe:
                continue
            options.append((score,p.id,target,power))
        return max(options,key=lambda item:item[0]) if options else None

    def _start_pass(self,context,handler,option, *, boundary=False):
        _,receiver,target,power = option
        self.pass_plan = {'passer':handler.id,'receiver':receiver,'target':target,'power':power,
                          'origin':(context.ball.x,context.ball.y),'launched':False,
                          'until':context.now+PASS_PLAN_HOLD_S,'boundary':boundary}
        _log.info('TACTIC pass_plan passer=%d receiver=%d target=%s power=%.2f boundary=%s',
                  handler.id,receiver,target,power,boundary)

    @staticmethod
    def _receive_target(context,receiver,motion):
        target,_ = motion.intercept(context,receiver.pose)
        target = target or (context.ball.x,context.ball.y)
        if motion.speed > RECEIVE_SPEED_MPS:
            # 站在来球落点的下游，面向来球，避免迎头冲向球或与传球者碰撞。
            target = tuple(a+v/motion.speed*RECEIVE_CUSHION_OFFSET_M
                           for a,v in zip(target,motion.velocity))
        return legal_target(context,target)

    def _ball_intent(self,context,handler,intercept,rest,motion,mode):
        ball = (context.ball.x,context.ball.y)
        f = context.field
        if ball[0]<=-f.length/2+f.penalty_area_length and abs(ball[1])<=f.penalty_area_width/2:
            self.pass_plan = None
            return Intent('penalty:clear',intercept,handler.clearance_target(),SHOT_POWER)
        if self.pass_plan is not None and not self.pass_plan['launched']:
            if self.pass_plan['passer'] == handler.id:
                role = 'boundary:pass' if self.pass_plan.get('boundary') else 'pass'
                return Intent(role,intercept,self.pass_plan['target'],self.pass_plan['power'])
            self.pass_plan = None
        if abs(ball[1]) > f.width/2-TOUCHLINE_ZONE_M:
            option = self._pass_option(context,handler,rest,motion,boundary=True)
            if option is not None:
                self._start_pass(context,handler,option,boundary=True)
                return Intent('boundary:pass',ball,option[2],option[3])
            # 没有可靠接应者时，只短斜送到同侧内场，不盲目横穿整片场地。
            side = 1.0 if ball[1]>=0 else -1.0
            y = side*max(0.,abs(ball[1])-BOUNDARY_INWARD_M)
            candidates = [legal_target(context,(ball[0]+step,y)) for step in
                          (BOUNDARY_RETURN_DEPTH_M,GOAL_CLEAR_FORWARD_M,FIELD_SAFE_MARGIN_M)]
            target = max(candidates,key=lambda t:lane_clearance(ball,t,context.opponents.values()))
            power = max(PASS_POWER_MIN,min(PASS_POWER_MAX,
                        motion.power_for_distance(distance(ball,target),BOUNDARY_ARRIVAL_SPEED_MPS)))
            return Intent('boundary:save',ball,target,power)
        goal = self._goal_target(context)
        if distance(point_of(handler),ball)<=BALL_KICK_ENGAGE_M or handler.is_kicking:
            self.pass_plan=None
            return Intent('shoot',intercept,goal,SHOT_POWER)
        option=self._pass_option(context,handler,rest,motion)
        if option is not None and self.possession=='ours':
            self._start_pass(context,handler,option)
            return Intent('pass',intercept,option[2],option[3])
        return Intent('advance',intercept,goal,SHOT_POWER)

    def _marking(self,context,players,handler,holder,compact):
        """一人盯一人，持球者的防守人占球与球门之间；最短总走位带迟滞。"""
        goal = (-context.field.length/2,0.)
        opponents = [r for r in context.opponents.values() if r.pose is not None
                     and abs(r.pose.x)<=context.field.length/2
                     and abs(r.pose.y)<=context.field.width/2]
        targets = {}
        for r in opponents:
            point = (r.pose.x,r.pose.y)
            length = max(distance(point,goal),MARK_GOALSIDE_M)
            target = (point[0]+(goal[0]-point[0])*MARK_GOALSIDE_M/length,
                      point[1]+(goal[1]-point[1])*MARK_GOALSIDE_M/length)
            if compact:
                target = (min(-DEFEND_COVER_DEPTH_M,target[0]),target[1])
            targets[r.player_id] = legal_target(context,target)
        assignments = {}
        if holder is not None and holder.player_id in targets:
            assignments[handler.id] = holder.player_id
        rest = [p for p in players if p.id not in assignments]
        ids = [i for i in targets if i not in assignments.values()]
        best = None
        for selected in permutations(ids,min(len(rest),len(ids))):
            cost = sum(distance(point_of(p),targets[i])+
                (MARK_SWITCH_COST_M if self.mark_assignments.get(p.id,i)!=i else 0.)
                for p,i in zip(rest,selected))
            if best is None or cost<best[0]:
                best = cost,selected
        if best is not None:
            assignments.update({p.id:i for p,i in zip(rest,best[1])})
        self.mark_assignments = assignments
        return {pid:Intent('mark:'+str(oid),targets[oid]) for pid,oid in assignments.items()}

    def _keeper_target(self,context,motion):
        ball = (context.ball.x,context.ball.y)
        goal_x = -context.field.length/2
        depth = DEFEND_KEEPER_LINE_M
        if ball[0] <= goal_x+context.field.penalty_area_length:
            if motion.speed <= KEEPER_FAST_SHOT_MPS:
                depth = max(depth,min(KEEPER_ADVANCE_DEPTH_M,
                                       ball[0]-goal_x-KEEPER_ADVANCE_BALL_GAP_M))
            else:
                # 已前移封角时，不因球突然加速而主动退让射门线路。
                anchor = context.teammates.get(self.anchor_id)
                if anchor is not None and anchor.pose is not None:
                    depth = max(depth,min(KEEPER_ADVANCE_DEPTH_M,anchor.pose.x-goal_x))
        x = goal_x+depth
        projected = motion.crossing_y(x)
        if projected is None:
            fraction = depth/max(ball[0]-goal_x,depth)
            projected = ball[1]*min(1.0,fraction)
        limit = context.field.goal_width/2-DEFEND_KEEPER_Y_MARGIN_M
        return x,max(-limit,min(limit,projected))

    def _cover_target(self,context,keeper):
        ball = (context.ball.x,context.ball.y)
        opponents = [r for r in context.opponents.values() if r.pose is not None]
        holder = min(opponents,key=lambda r:distance(ball,(r.pose.x,r.pose.y)),default=None)
        threats = [r for r in opponents if r is not holder]
        threat = min(threats,key=lambda r:r.pose.x,default=None)
        if threat is not None:
            target = ((ball[0]+threat.pose.x)/2-FIELD_SAFE_MARGIN_M,
                      (ball[1]+threat.pose.y)/2)
        else:
            target = (ball[0]-DEFEND_COVER_DEPTH_M,ball[1]/2)
        # 第二道保护在球与门之间，与门将横向错开。
        x = max(keeper[0]+DEFEND_COVER_SEPARATION_M,min(ball[0]-FIELD_SAFE_MARGIN_M,target[0]))
        side = 1.0 if ball[1] >= 0 else -1.0
        y = target[1]
        if abs(y-keeper[1]) < DEFEND_COVER_SEPARATION_M:
            y = keeper[1]+side*DEFEND_COVER_SEPARATION_M
        return legal_target(context,(x,y))

    def _tackle_intent(self,context,player,motion,role='press:tackle'):
        ball = (context.ball.x,context.ball.y)
        if distance(point_of(player),ball) <= TACKLE_ENGAGE_M or (player.is_kicking and distance(point_of(player),ball)<=KICK_EXIT_M):
            target = self._goal_target(context)
            return Intent(role,ball,target,SHOT_POWER)
        intercept,_ = motion.intercept(context,player.pose)
        return Intent('press:approach' if role=='press:tackle' else role,intercept or ball,receive=True)

    def _choose_tackler(self,context,players):
        """同一争球过程保持处理者；资格筛选在调用处，不能锁住已被突破的人。"""
        ball = (context.ball.x,context.ball.y)
        best = min(players,key=lambda p:distance(point_of(p),ball))
        previous = next((p for p in players if p.id == self.attacker_id),None)
        if (previous is not None and context.now >= getattr(previous,'routing_blocked_until',0.)
                and distance(point_of(previous),ball) <= distance(point_of(best),ball)+TACKLE_SWITCH_ADVANTAGE_M):
            best = previous
        if best.id != self.attacker_id:
            self.attacker_id,self.attacker_since = best.id,context.now
        return best

    def plan(self,context,players,motion):
        """统一控球压迫：不按比分/半场/反抢计时退守，只保留一个球门保护者。"""
        if not players or context.ball is None:
            return {}
        self._update_teammate_velocity(context,players)
        self._update_possession(context,players)
        self._update_pass(context,players,motion)
        self.mode = 'possession'
        ball = (context.ball.x,context.ball.y)
        goal = (-context.field.length/2,0.)
        keeper = next((p for p in players if p.id == self.anchor_id),None)
        if keeper is None:
            keeper = min(players,key=lambda p:distance(point_of(p),goal))
        self.anchor_id = keeper.id
        field = [p for p in players if p is not keeper]
        crossing = motion.crossing_y(goal[0]+DEFEND_KEEPER_LINE_M)
        incoming = (motion.speed>KEEPER_FAST_SHOT_MPS and motion.velocity[0]<KEEPER_INCOMING_VX_MPS
                    and crossing is not None and abs(crossing)<=context.field.goal_width/2+GOAL_DANGER_WIDTH_MARGIN_M)
        keeper_distance = distance(point_of(keeper),ball)
        sweep = (keeper_distance<=KEEPER_CLEAR_REACH_M or
                 (not incoming and ball[0]<=goal[0]+context.field.penalty_area_length
                  and keeper_distance<=KEEPER_CONTEST_REACH_M
                  and keeper_distance+KEEPER_CONTEST_ADVANTAGE_M<min(
                      (distance(point_of(p),ball) for p in field),default=math.inf)))
        candidates = players if sweep or not field else field
        if incoming and field and not sweep:
            goalside=[p for p in field if p.pose.x<=ball[0]+TACKLE_GOALSIDE_MARGIN_M]
            candidates=goalside or field
        if goal_danger(context):
            handler = self._choose_tackler(context,candidates)
            intercept = ball if motion.speed<=RECEIVE_SPEED_MPS else motion.intercept(context,handler.pose)[0] or ball
            self.pass_plan = None
        else:
            handler,intercept = self._choose_attacker(context,candidates,motion)
        if sweep and keeper_distance+KEEPER_CONTEST_ADVANTAGE_M<min(
                (distance(point_of(p),ball) for p in field),default=math.inf):
            handler=keeper
            self.attacker_id=keeper.id
            intercept=motion.intercept(context,keeper.pose)[0] or ball
        rest = [p for p in players if p is not handler]
        anchor = keeper if keeper in rest else min(rest,key=lambda p:distance(point_of(p),goal),default=None)
        if anchor is not None:self.anchor_id=anchor.id
        threatened = self.possession == 'theirs'
        holder = min((r for r in context.opponents.values() if r.pose is not None),
                     key=lambda r:distance(ball,(r.pose.x,r.pose.y)),default=None)
        self._marking(context,players,handler,holder,False)
        if goal_danger(context):
            action=Intent('goal:clear',intercept,handler.clearance_target(),SHOT_POWER)
        elif handler is keeper and ball[0]<=goal[0]+context.field.penalty_area_length:
            action=Intent('penalty:clear',intercept,handler.clearance_target(),SHOT_POWER)
        elif self.pass_plan is not None and self.pass_plan['launched'] and not threatened:
            action=Intent('receive:intercept',self._receive_target(context,handler,motion),receive=True)
        elif (context.now<self.kickoff_follow_until and distance(point_of(handler),ball)>RECEIVE_REACH_M):
            action=self._tackle_intent(context,handler,motion,role='kickoff:contest')
        elif threatened:
            action=self._tackle_intent(context,handler,motion)
        else:
            action=self._ball_intent(context,handler,intercept,rest,motion,self.mode)
        result={handler.id:action}
        if anchor is not None:
            result[anchor.id]=Intent('cover:goal',self._keeper_target(context,motion))
        for p in rest:
            if p is anchor:continue
            predicted=motion.predict(context,PRESS_SUPPORT_LOOKAHEAD_S) or ball
            side=-1. if handler.pose.y>=predicted[1] else 1.
            if goal_danger(context):
                target=self._cover_target(context,self._keeper_target(context,motion))
                role='press:outlet'
            elif threatened or incoming:
                # 跟上未来球位压迫出球点，取消固定四分之一线与反抢后全员回撤。
                target=(predicted[0]-PRESS_SUPPORT_DEPTH_M,predicted[1]+side*PRESS_SUPPORT_WIDTH_M)
                role='press:outlet'
            else:
                target=(predicted[0]+PASS_RUN_AHEAD_M,predicted[1]+side*TRIANGLE_WIDTH_M)
                role='triangle:wing'
            result[p.id]=Intent(role,legal_target(context,target))
        plan=self.pass_plan
        if plan is not None and not threatened and not goal_danger(context):
            receiver=next((p for p in players if p.id==plan['receiver']),None)
            if receiver is not None:
                result[receiver.id]=Intent('receive:intercept' if plan['launched'] else 'receive:ready',
                    self._receive_target(context,receiver,motion) if plan['launched'] else plan['target'],receive=True)
        return result
