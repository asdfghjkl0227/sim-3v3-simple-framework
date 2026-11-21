"""从公开 Context 位置差分估球速；碰撞/摆球不用于滚动阻力拟合。

预测是战术估计，不判断球权、触球身份、出界或进球；无文件/网络 IO。
"""
from collections import deque
import logging
import math

from ..framework.types import GameState
from ..param import *

_log = logging.getLogger(__name__)


class BallMotion:
    def __init__(self):
        self.samples = deque()
        self.position = None
        self.velocity = (0.0, 0.0)
        self.deceleration = BALL_DECEL_INITIAL_MPS2
        self.roll_samples = 0
        self.contact_changes = 0
        self.max_speed = 0.0
        self.last_log = None
        self.last_sample_at = None
        self._last_free = False
        self._kick_request = None
        self.kick_samples = []

    @property
    def speed(self):
        return math.hypot(*self.velocity)

    def reset(self):
        self.samples.clear()
        self.position = None
        self.velocity = (0.0, 0.0)
        self.last_sample_at = None
        self._last_free = False
        self._kick_request = None

    @staticmethod
    def _clear_of_robots(context, point, exclude=None):
        return all(math.hypot(r.pose.x-point[0], r.pose.y-point[1]) > BALL_FREE_ROLL_CLEARANCE_M
                   for r in (*context.teammates.values(), *context.opponents.values())
                   if r.pose is not None and r.player_id != exclude)

    def update(self, context):
        ball, game = context.ball, context.game
        if ball is None or game is None or game.state != GameState.PLAYING or game.stopped:
            self.reset()
            return
        now, point = context.now, (ball.x, ball.y)
        if self.last_sample_at is not None:
            dt = now-self.last_sample_at
            if dt < BALL_SAMPLE_MIN_DT_S:
                return
            displacement = math.dist(point, self.position)
            if (dt > BALL_SAMPLE_MAX_GAP_S or dt <= 0 or
                    displacement > BALL_MAX_OBSERVED_SPEED_MPS*dt+BALL_TELEPORT_MARGIN_M):
                self.reset()
        self.samples.append((now, *point))
        while len(self.samples) > 2 and now-self.samples[1][0] >= BALL_VELOCITY_WINDOW_S:
            self.samples.popleft()
        previous_velocity = self.velocity
        previous_speed = self.speed
        previous_time = self.last_sample_at
        self.position = point
        self.last_sample_at = now
        duration = now-self.samples[0][0]
        if duration < BALL_SAMPLE_MIN_DT_S:
            return
        raw = ((point[0]-self.samples[0][1])/duration,
               (point[1]-self.samples[0][2])/duration)
        if math.hypot(*raw) > BALL_MAX_OBSERVED_SPEED_MPS:
            self.reset()
            return
        alpha = BALL_VELOCITY_FILTER
        self.velocity = tuple(alpha*r+(1-alpha)*v for r,v in zip(raw,previous_velocity))
        if math.hypot(*raw) < BALL_STOP_SPEED_MPS:
            self.velocity = (0.0,0.0)
        speed = self.speed
        self.max_speed = max(self.max_speed, speed)
        free = self._clear_of_robots(context,point)
        if previous_time is not None and min(speed,previous_speed) > BALL_ROLL_MIN_SPEED_MPS:
            dt = now-previous_time
            cos_turn = sum(a*b for a,b in zip(previous_velocity,self.velocity))/(speed*previous_speed)
            turn = math.acos(max(-1.0,min(1.0,cos_turn)))
            decel = (previous_speed-speed)/dt
            if free and self._last_free and turn < BALL_ROLL_MAX_TURN_RAD:
                if BALL_DECEL_MIN_MPS2 <= decel <= BALL_DECEL_MAX_MPS2:
                    self.deceleration = ((1-BALL_DECEL_FILTER)*self.deceleration
                                         +BALL_DECEL_FILTER*decel)
                    self.roll_samples += 1
            elif turn > BALL_ROLL_MAX_TURN_RAD:
                self.contact_changes += 1
        self._last_free = free
        self._sample_kick(context)
        if self.last_log is None or now-self.last_log >= BALL_CALIBRATION_LOG_S:
            self.last_log = now
            _log.info("BALL_MODEL speed=%.2f peak=%.2f decel=%.3f roll_samples=%d contacts=%d kick_samples=%d",
                      speed,self.max_speed,self.deceleration,self.roll_samples,
                      self.contact_changes,len(self.kick_samples))

    def observe_kick_request(self, context, player_id, power, direction):
        """只标注命令与球运动的相关性，绝不当作官方触球事实。"""
        if context.ball is None:
            return
        if self._kick_request is None or self._kick_request['player'] != player_id:
            self._kick_request = dict(player=player_id, power=power, direction=direction,
                                      at=context.now, origin=(context.ball.x,context.ball.y))

    def _sample_kick(self, context):
        request = self._kick_request
        if request is None:
            return
        elapsed = context.now-request['at']
        if elapsed > BALL_KICK_SAMPLE_WINDOW_S:
            self._kick_request = None
            return
        if self.position is None or math.dist(self.position,request['origin']) < BALL_INTERCEPT_REACH_M:
            return
        direction = math.atan2(self.velocity[1],self.velocity[0])
        turn = abs(math.atan2(math.sin(direction-request['direction']),math.cos(direction-request['direction'])))
        # 对手或其他队友在球旁时无法可靠归因，不记作力度实验样本。
        if (self.speed >= BALL_KICK_SAMPLE_MIN_SPEED_MPS and turn < BALL_KICK_DIRECTION_TOL_RAD
                and self._clear_of_robots(context,self.position,exclude=request['player'])):
            self.kick_samples.append((request['power'],self.speed))
            _log.info("BALL_KICK_OBSERVATION power=%.2f speed=%.2f elapsed=%.2f (command correlation, not touch proof)",
                      request['power'],self.speed,elapsed)
            self._kick_request = None

    def predict(self, context, seconds):
        if self.position is None:
            return None
        speed = self.speed
        seconds = max(0.0,min(BALL_PREDICTION_MAX_S,seconds))
        t = min(seconds,speed/self.deceleration) if speed else 0.0
        travel = max(0.0,speed*t-self.deceleration*t*t/2)
        ux,uy = (v/speed for v in self.velocity) if speed else (0.0,0.0)
        x,y = self.position[0]+ux*travel,self.position[1]+uy*travel
        # 越过边线后由裁判决定重启，不编造“围墙反弹”。
        hx = context.field.length/2-BALL_PREDICTION_BOUNDARY_M
        hy = context.field.width/2-BALL_PREDICTION_BOUNDARY_M
        return max(-hx,min(hx,x)),max(-hy,min(hy,y))

    def intercept(self, context, pose):
        if self.position is None:
            return None, math.inf
        best = self.position
        best_cost = math.inf
        steps = int(BALL_PREDICTION_MAX_S/BALL_PREDICTION_STEP_S)
        for i in range(steps+1):
            t = i*BALL_PREDICTION_STEP_S
            point = self.predict(context,t)
            distance = math.dist((pose.x,pose.y),point)
            heading = math.atan2(point[1]-pose.y,point[0]-pose.x)
            turn = abs(math.atan2(math.sin(heading-pose.theta),math.cos(heading-pose.theta)))
            walk_time = (max(0.0,distance-BALL_INTERCEPT_REACH_M)/BALL_WALK_SPEED_MPS
                         + (turn*BALL_TURN_LATENCY_S_PER_RAD if distance>OMNI_DIST else 0.0))
            cost = max(walk_time,t)
            if cost < best_cost:
                best,best_cost = point,cost
            if walk_time <= t:
                return point,t
        return best,best_cost

    def crossing_y(self, x):
        if self.position is None or abs(self.velocity[0]) < BALL_STOP_SPEED_MPS:
            return None
        dx = x-self.position[0]
        if dx*self.velocity[0] <= 0:
            return None
        speed = self.speed
        travel = dx*speed/self.velocity[0]
        if travel > speed*speed/(2*self.deceleration):
            return None
        t = (speed-math.sqrt(max(0.0,speed*speed-2*self.deceleration*travel)))/self.deceleration
        if t > BALL_PREDICTION_MAX_S:
            return None
        return self.position[1]+dx*self.velocity[1]/self.velocity[0]

    def power_for_distance(self, distance, arrival_speed):
        required = math.sqrt(arrival_speed*arrival_speed+2*self.deceleration*distance)
        gain = BALL_KICK_SPEED_PER_POWER
        if self.kick_samples:
            ratios = sorted((speed-BALL_KICK_SPEED_INITIAL_MPS)/power
                            for power,speed in self.kick_samples if power > 0)
            gain = max(BALL_KICK_SPEED_PER_POWER/2,ratios[len(ratios)//2])
        return (required-BALL_KICK_SPEED_INITIAL_MPS)/gain
