"""纯几何合规工具；只使用官方 Context，不推断裁判放行或触球身份。"""
import math

from ..param import (
    FIELD_SAFE_MARGIN_M, HALF_SAFE_MARGIN_M, RULE_TARGET_MARGIN_M,
    RULE_LOOKAHEAD_S, RULE_ESCAPE_SPEED_MPS, RULE_ANGLE_SAMPLES,
    RESTART_GOAL_MARGIN_M, RESTART_PUSH_LENGTH_M, MAX_LINEAR,
    GOAL_DANGER_DEPTH_M, GOAL_DANGER_WIDTH_MARGIN_M,
    LIVE_BALL_FIELD_MARGIN_M, TOUCHLINE_ZONE_M, TOUCHLINE_ROBOT_EXCURSION_M,
    GOAL_DEPTH, GOAL_RESCUE_BACK_MARGIN_M, GOAL_RESCUE_POST_MARGIN_M,
)
from ..framework.types import GameState, SetPlay


def bounds(context, own_half=False, ball_access=False):
    x = context.field.length / 2 - FIELD_SAFE_MARGIN_M
    y = context.field.width / 2 - FIELD_SAFE_MARGIN_M
    if ball_access and live_ball_access(context) and not own_half:
        x = context.field.length/2-LIVE_BALL_FIELD_MARGIN_M
        y = context.field.width/2-LIVE_BALL_FIELD_MARGIN_M
        ball = context.ball
        # 底线靠近边角时，也需绕到球外侧再向场内出球；中间球门区域不开放。
        if (abs(ball.x) > context.field.length/2-TOUCHLINE_ZONE_M
                and abs(ball.y) > context.field.goal_width/2+GOAL_RESCUE_POST_MARGIN_M):
            lo = -context.field.length/2-TOUCHLINE_ROBOT_EXCURSION_M if ball.x<0 else -x
            hi = context.field.length/2+TOUCHLINE_ROBOT_EXCURSION_M if ball.x>0 else x
            bottom = -context.field.width/2-TOUCHLINE_ROBOT_EXCURSION_M if ball.y<0 else -y
            top = context.field.width/2+TOUCHLINE_ROBOT_EXCURSION_M if ball.y>0 else y
            return lo,hi,bottom,top
        if (ball.x < -context.field.length/2+FIELD_SAFE_MARGIN_M
                and abs(ball.y)<context.field.goal_width/2-GOAL_RESCUE_POST_MARGIN_M):
            return -context.field.length/2-GOAL_DEPTH+GOAL_RESCUE_BACK_MARGIN_M,x,-y,y
        # 仅在边线活球的外侧给救球者留短暂绕行空间，不把球门后当通路。
        if abs(ball.y) > context.field.width/2-TOUCHLINE_ZONE_M:
            if ball.y > 0:
                return -x,x,-y,context.field.width/2+TOUCHLINE_ROBOT_EXCURSION_M
            return -x,x,-context.field.width/2-TOUCHLINE_ROBOT_EXCURSION_M,y
    return -x, -HALF_SAFE_MARGIN_M if own_half else x, -y, y


def live_ball_access(context):
    g = context.game
    return (context.ball is not None and g is not None and g.state == GameState.PLAYING
            and not g.stopped and ((g.set_play == SetPlay.NONE and g.secondary_time <= 0)
                or (g.set_play != SetPlay.NONE and g.kicking_team == context.team_id)))


def goal_danger(context):
    """只判断救险优先级；进球/死球永远服从裁判状态。"""
    ball = context.ball
    return (ball is not None and ball.x <= -context.field.length/2+GOAL_DANGER_DEPTH_M
            and abs(ball.y) <= context.field.goal_width/2+GOAL_DANGER_WIDTH_MARGIN_M)


def segment_clear(start, end, discs, margin=0.0):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    for c in discs:
        t = max(0.0, min(1.0, ((c.x-start[0])*dx + (c.y-start[1])*dy) / length2)) if length2 else 0.0
        if math.hypot(start[0]+t*dx-c.x, start[1]+t*dy-c.y) < c.radius + margin:
            return False
    return True


def legal_target(context, target, discs=(), own_half=False, *, ball_access=False):
    """投影后重新验圈，避免边线钳制造成目标重新落入禁区。"""
    lo, hi, bottom, top = bounds(context, own_half,ball_access)
    def clip(p):
        return max(lo, min(hi, p[0])), max(bottom, min(top, p[1]))
    def legal(p):
        return all(math.hypot(p[0]-c.x, p[1]-c.y) >= c.radius + RULE_TARGET_MARGIN_M for c in discs)
    p = clip(target)
    if legal(p):
        return p
    candidates = [(lo,bottom), (lo,top), (hi,bottom), (hi,top)]
    for c in discs:
        for i in range(RULE_ANGLE_SAMPLES):
            a = math.tau * i / RULE_ANGLE_SAMPLES
            r = c.radius + RULE_TARGET_MARGIN_M
            candidates.append(clip((c.x+r*math.cos(a), c.y+r*math.sin(a))))
    candidates = [p for p in candidates if legal(p)]
    return min(candidates, key=lambda p: math.hypot(p[0]-target[0], p[1]-target[1])) if candidates else None


def safe_velocity(context, pose, vx, vy, discs=(), own_half=False, *, ball_access=False):
    """场地速度闸门；禁区内只允许真实退离，禁区外检查整段路径。"""
    lo, hi, bottom, top = bounds(context, own_half,ball_access)
    start = (pose.x, pose.y)
    speed = math.hypot(vx,vy)
    if speed > MAX_LINEAR:
        vx,vy = vx*MAX_LINEAR/speed, vy*MAX_LINEAR/speed
    inside = [c for c in discs if math.hypot(pose.x-c.x, pose.y-c.y) < c.radius]
    def valid(x, y):
        end = (pose.x+x*RULE_LOOKAHEAD_S, pose.y+y*RULE_LOOKAHEAD_S)
        # 已在界外时允许回场，不允许继续远离合法区域。
        if not min(lo,pose.x) <= end[0] <= max(hi,pose.x): return False
        if not min(bottom,pose.y) <= end[1] <= max(top,pose.y): return False
        for c in discs:
            d = math.hypot(pose.x-c.x, pose.y-c.y)
            if c in inside:
                if math.hypot(x,y) < RULE_ESCAPE_SPEED_MPS/2:
                    return False
                # 已在对方半场中圈内时，先沿切线离圈再回半场。
                # 强制较大径向速度会与不得继续深入对方半场的约束互相锁死。
                if math.hypot(end[0]-c.x,end[1]-c.y) <= d+1e-4:
                    return False
                if x*(pose.x-c.x)+y*(pose.y-c.y) < -1e-6:
                    return False
            elif not segment_clear(start, end, [c]):
                return False
        return True
    if valid(vx, vy): return vx, vy
    if not inside: return 0.0, 0.0
    # 重摆球时机器人可能已在禁区内，立即全向撤离，不能先原地转身。
    candidates = []
    for i in range(RULE_ANGLE_SAMPLES):
        a = math.tau*i/RULE_ANGLE_SAMPLES
        x,y = RULE_ESCAPE_SPEED_MPS*math.cos(a), RULE_ESCAPE_SPEED_MPS*math.sin(a)
        if valid(x,y): candidates.append((x,y))
    radial = [v for v in candidates if all(
        (v[0]*(pose.x-c.x)+v[1]*(pose.y-c.y))/max(1e-6,math.hypot(pose.x-c.x,pose.y-c.y))
        >= RULE_ESCAPE_SPEED_MPS/2 for c in inside)]
    return min(radial or candidates, key=lambda p: (p[0]-vx)**2+(p[1]-vy)**2) if candidates else (0.0,0.0)


def safe_push_direction(context):
    """轻推离开摆球点：排除两侧球门方向，并优先把球留在场内。"""
    ball = context.ball
    if ball is None: return None
    lo,hi,bottom,top = bounds(context)
    candidates = []
    for i in range(RULE_ANGLE_SAMPLES):
        a = math.tau*i/RULE_ANGLE_SAMPLES
        dx,dy = math.cos(a), math.sin(a)
        unsafe = False
        for goal_x in (-context.field.length/2, context.field.length/2):
            if abs(dx) < 1e-6: continue
            t = (goal_x-ball.x)/dx
            if t >= 0 and abs(ball.y+t*dy) <= context.field.goal_width/2+RESTART_GOAL_MARGIN_M:
                unsafe = True
        end = (ball.x+dx*RESTART_PUSH_LENGTH_M, ball.y+dy*RESTART_PUSH_LENGTH_M)
        if not unsafe and lo <= end[0] <= hi and bottom <= end[1] <= top:
            candidates.append((a, dx))
    # 前斜向出球，开球时从本方半场接近球。
    return max(candidates, key=lambda item: item[1])[0] if candidates else None
