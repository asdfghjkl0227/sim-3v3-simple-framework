"""三人制战术规划。输出意图，实际动作仍走 Player 的规则闸门。"""
from dataclasses import dataclass
import logging
import math
from itertools import permutations

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


class TeamTactics:
    def __init__(self):
        self.mode = 'loose'
        self.possession = 'loose'
        self.possession_at = 0.0
        self.counterpress_until = 0.0
        self.attacker_id = None
        self.attacker_since = 0.0
        self.anchor_id = None
        self.pass_plan = None
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
        self.possession = 'loose'
        self.counterpress_until = 0.0
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
            self.pass_plan = None

    @staticmethod
    def score_margin(context):
        our = context.game.get_team_state(context.team_id)
        opponents = [t for t in context.game.teams if t.team_number != context.team_id]
        if our is None or not opponents:
            return 0
        return our.score-max(t.score for t in opponents)

    @staticmethod
    def _protect_lead(context):
        return TeamTactics.score_margin(context) >= LEAD_DEFEND_GOAL_MARGIN

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
            if (costs[preferred.id] <= costs[best.id]+TACTIC_INTERCEPT_KEEP_S or
                    (context.now-self.attacker_since < TACTIC_ROLE_KEEP_S and
                     context.now >= getattr(preferred,'routing_blocked_until',0.0))):
                best = preferred
        plan = self.pass_plan
        if plan is not None and plan['launched'] and self.possession != 'theirs':
            receiver = next((p for p in players if p.id==plan['receiver']),None)
            if receiver is not None:
                # 传球窗口内只有指定接球者处理球，不能再让第三人也收到迎球指令。
                best = receiver
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
        travel = distance(ball,plan['origin'])
        dx,dy = plan['target'][0]-plan['origin'][0],plan['target'][1]-plan['origin'][1]
        length = math.hypot(dx,dy)
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

    @staticmethod
    def _goal_target(context):
        ball = (context.ball.x,context.ball.y)
        if distance(ball,(context.field.length/2,0.))>SHOT_CENTER_DISTANCE_M:
            return context.field.length/2,0.
        y = context.field.goal_width/2-SHOT_TARGET_POST_MARGIN_M
        targets = [(context.field.length/2,gy) for gy in (-y,0.0,y)]
        return max(targets,key=lambda target:lane_clearance(ball,target,context.opponents.values()))

    def _pass_option(self,context,handler,rest,motion):
        ball = (context.ball.x,context.ball.y)
        options = []
        for p in rest:
            if p.id == self.anchor_id:
                continue
            velocity = self.velocities.get(p.id,(0.0,0.0))
            lead = (velocity[0]*PASS_LEAD_TIME_S,velocity[1]*PASS_LEAD_TIME_S)
            size = math.hypot(*lead)
            if size > PASS_LEAD_MAX_M:
                lead = tuple(v*PASS_LEAD_MAX_M/size for v in lead)
            target = legal_target(context,(p.pose.x+lead[0],p.pose.y+lead[1]))
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
            options.append((score,p.id,target,power))
        return max(options,key=lambda item:item[0]) if options else None

    def _ball_intent(self,context,handler,intercept,rest,motion,mode):
        ball = (context.ball.x,context.ball.y)
        f = context.field
        if ball[0]<=-f.length/2+f.penalty_area_length and abs(ball[1])<=f.penalty_area_width/2:
            self.pass_plan = None
            return Intent('penalty:clear',intercept,handler.clearance_target(),SHOT_POWER)
        if abs(ball[1]) > f.width/2-TOUCHLINE_ZONE_M:
            self.pass_plan = None
            # 从球外侧接近，向中路短送；不能继续沿边线盲目向前踢。
            target = legal_target(context,(ball[0]+BOUNDARY_RETURN_DEPTH_M,0.))
            power = max(PASS_POWER_MIN,min(PASS_POWER_MAX,
                        motion.power_for_distance(distance(ball,target),PASS_ARRIVAL_SPEED_MPS)))
            return Intent('boundary:save',ball,target,power)
        if self.pass_plan is not None and not self.pass_plan['launched']:
            if self.pass_plan['passer'] == handler.id:
                return Intent('pass',intercept,self.pass_plan['target'],self.pass_plan['power'])
            self.pass_plan = None
        goal = self._goal_target(context)
        clear = lane_clearance(ball,goal,context.opponents.values())
        goal_distance = distance(ball,goal)
        pressure = min((distance(ball,(r.pose.x,r.pose.y)) for r in context.opponents.values()
                        if r.pose is not None),default=math.inf)
        shoot = (goal_distance < SHOT_CLOSE_DISTANCE_M or
                 (goal_distance < SHOT_MAX_DISTANCE_M and clear >= SHOT_LANE_CLEARANCE_M))
        option = self._pass_option(context,handler,rest,motion)
        if option is not None and not shoot:
            _,receiver,target,power = option
            if target[0] > ball[0]+PASS_PROGRESS_MIN_M or pressure < PASS_PRESSURE_M:
                self.pass_plan = dict(passer=handler.id,receiver=receiver,target=target,power=power,
                                      origin=ball,launched=False,until=context.now+PASS_PLAN_HOLD_S)
                _log.info('TACTIC pass_plan passer=%d receiver=%d distance=%.2f power=%.2f',
                          handler.id,receiver,distance(ball,target),power)
                return Intent('pass',intercept,target,power)
        if shoot:
            return Intent('shoot',intercept,goal,SHOT_POWER)
        # 防守先把球送离门前，进攻时可沿空侧推进；始终只由一个人处理球。
        side = 1.0 if ball[1] >= 0 else -1.0
        if mode == 'protect_lead':
            target = (context.field.length/2-FIELD_SAFE_MARGIN_M,side*TACTIC_CLEARANCE_Y_M)
            return Intent('press:clear',intercept,target,SHOT_POWER)
        return Intent('advance',intercept,goal,ADVANCE_POWER)

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
        x = -context.field.length/2+DEFEND_KEEPER_LINE_M
        projected = motion.crossing_y(x)
        if projected is None:
            fraction = DEFEND_KEEPER_LINE_M/max(ball[0]+context.field.length/2,DEFEND_KEEPER_LINE_M)
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

    def plan(self,context,players,motion):
        if not players or context.ball is None:
            return {}
        self._update_teammate_velocity(context,players)
        self._update_possession(context,players)
        self._update_pass(context,players,motion)
        margin = self.score_margin(context)
        mode = 'protect_lead' if self._protect_lead(context) else ('chase' if margin<0 else 'balanced')
        self.mode = mode
        if mode != self._last_mode_log:
            _log.info('TACTIC mode=%s possession_estimate=%s',mode,self.possession)
            self._last_mode_log = mode
        goal_point = (-context.field.length/2,0.)
        keeper = next((p for p in players if p.id == self.anchor_id),None)
        if keeper is None:
            keeper = min(players,key=lambda p:distance(point_of(p),goal_point))
        self.anchor_id = keeper.id
        field_players = [p for p in players if p is not keeper]
        ball_point = (context.ball.x,context.ball.y)
        crossing = motion.crossing_y(-context.field.length/2+DEFEND_KEEPER_LINE_M)
        incoming = (motion.speed > KEEPER_FAST_SHOT_MPS
                    and motion.velocity[0] < KEEPER_INCOMING_VX_MPS
                    and crossing is not None
                    and abs(crossing) <= context.field.goal_width/2+GOAL_DANGER_WIDTH_MARGIN_M)
        hold_keeper = (incoming and field_players
                       and ball_point[0] > keeper.pose.x+KEEPER_GOALSIDE_MARGIN_M
                       and distance(point_of(keeper),ball_point) > KEEPER_FOOT_REACH_M)
        can_sweep = (not hold_keeper and
                     distance(point_of(keeper),ball_point) <= KEEPER_SWEEP_REACH_M)
        candidates = players if can_sweep or not field_players else field_players
        handler,intercept = self._choose_attacker(context,candidates,motion)
        if goal_danger(context):
            # 球靠近或压在门线上都不是“已进球”；当帧选最近可用者解围。
            self.pass_plan = None
            handler = min(field_players if hold_keeper else players,
                          key=lambda p:distance(point_of(p),ball_point))
            if motion.speed<=RECEIVE_SPEED_MPS:
                intercept = (context.ball.x,context.ball.y)
            else:
                intercept,_ = motion.intercept(context,handler.pose)
        rest = [p for p in players if p.id != handler.id]
        ball = (context.ball.x,context.ball.y)
        opponents = [r for r in context.opponents.values() if r.pose is not None]
        holder = min(opponents,key=lambda r:distance(ball,(r.pose.x,r.pose.y)),default=None)
        defending = self.possession == 'theirs'
        marks = self._marking(context,players,handler,holder,mode=='protect_lead')
        if goal_danger(context):
            result = {handler.id:Intent('goal:clear',intercept,handler.clearance_target(),SHOT_POWER)}
        elif self.pass_plan is not None and self.pass_plan['launched'] and not defending:
            result = {handler.id:Intent('receive:intercept',intercept,receive=True)}
        else:
            result = {handler.id:self._ball_intent(context,handler,intercept,rest,motion,mode)}
        if defending and not goal_danger(context) and result[handler.id].role not in ('penalty:clear','boundary:save'):
            # 逼抢先到球门侧，不站持球人身后追。近身后由 SDK 执行争球。
            goal = (-context.field.length/2,0.)
            length = max(distance(ball,goal),PRESS_GOALSIDE_M)
            goalside = (ball[0]+(goal[0]-ball[0])*PRESS_GOALSIDE_M/length,
                        ball[1]+(goal[1]-ball[1])*PRESS_GOALSIDE_M/length)
            if distance(point_of(handler),ball)>PRESS_CONTEST_M:
                result[handler.id] = Intent('press:goalside',legal_target(context,goalside))
            else:
                result[handler.id] = Intent('press:contest',ball,self._goal_target(context),SHOT_POWER)
        if not rest:
            return result
        goal = (-context.field.length/2,0.0)
        anchor = min(rest,key=lambda p:distance(point_of(p),goal))
        # 人员被突破后按新的拦截时间选第一人，余下最近门者接替保护。
        previous = next((p for p in rest if p.id == self.anchor_id),None)
        if previous is not None and (not defending or
                distance(point_of(previous),goal) <= distance(point_of(anchor),goal)+ATTACKER_KEEP_DIST_MARGIN_M):
            anchor = previous
        self.anchor_id = anchor.id
        ball = (context.ball.x,context.ball.y)
        if mode == 'protect_lead':
            anchor_target = self._keeper_target(context,motion)
            wing_target = self._cover_target(context,anchor_target)
            anchor_role,wing_role = 'protect:goal','protect:lane'
        else:
            depth = CHASE_BASE_DEPTH_M if mode=='chase' else ATTACK_BASE_DEPTH_M
            anchor_target = (min(ATTACK_BASE_LIMIT_X_M,ball[0]-depth),ball[1]/3)
            side = -1.0 if ball[1] >= 0 else 1.0
            wing_target = (ball[0]+PASS_RUN_AHEAD_M,side*TRIANGLE_WIDTH_M)
            anchor_role,wing_role = 'triangle:base','triangle:wing'
        result[anchor.id] = Intent(anchor_role,legal_target(context,anchor_target))
        for p in rest:
            if p.id == anchor.id:
                continue
            if p.id == self.last_passer and context.now < self.run_until and not defending:
                wing_target = (ball[0]+PASS_RUN_AHEAD_M,-ball[1]/2)
                wing_role = 'pass:run_ahead'
            result[p.id] = Intent(wing_role,legal_target(context,wing_target))
        if defending:
            for p in rest:
                if p.id in marks:
                    result[p.id] = marks[p.id]
        if mode != 'protect_lead':
            # 常规攻防也要有人预先站在球门侧；其余两人继续前场争抢和接应。
            result[anchor.id] = Intent('mark:goal_cover' if defending else 'cover:goal',
                                       self._keeper_target(context,motion))
        plan = self.pass_plan
        if plan is not None and not goal_danger(context) and not defending:
            receiver = next((p for p in players if p.id==plan['receiver']),None)
            if receiver is not None:
                if plan['launched']:
                    target,_ = motion.intercept(context,receiver.pose)
                    target = target or ball
                    # 传球人在球离脚后前插，不能和接球人同时持续踢球。
                    passer = next((p for p in players if p.id==plan['passer']),None)
                    if passer is not None and passer.id == handler.id:
                        result[passer.id] = Intent('pass:run_ahead',legal_target(
                            context,(ball[0]+PASS_RUN_AHEAD_M,-ball[1]/2)))
                    result[receiver.id] = Intent('receive:intercept',target,receive=True)
                else:
                    result[receiver.id] = Intent('receive:ready',plan['target'],receive=True)
        return result
