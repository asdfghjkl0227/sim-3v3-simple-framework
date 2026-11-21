"""仅整形行走指令，不改变SDK步态、倒地判定或持续踢球任务。"""
import math

from ..param import (
    MAX_LINEAR, MAX_ANGULAR, WALK_SIDE_LIMIT_MPS, WALK_BACK_LIMIT_MPS,
    WALK_TURN_LIMIT_RADPS, WALK_TURN_SLOWDOWN, WALK_ACCEL_MPS2,
    WALK_YAW_ACCEL_RADPS2, WALK_COMMAND_MAX_DT_S, WALK_COMMAND_RESET_GAP_S,
)


class WalkCommand:
    def __init__(self):
        self.reset()

    def reset(self):
        self.at = None
        self.world = (0.0, 0.0)
        self.yaw = 0.0

    def shape(self, pose, now, dt, vx, vy, yaw, speed_limit=MAX_LINEAR):
        # 明确停止不受加速度限制，避免收到SET后残留位移。
        if not (vx or vy or yaw):
            self.reset()
            return 0.0, 0.0, 0.0
        elapsed = now-self.at if self.at is not None else dt
        if elapsed <= 0 or elapsed > WALK_COMMAND_RESET_GAP_S:
            self.reset()
            elapsed = dt
        elapsed = max(0.0, min(WALK_COMMAND_MAX_DT_S, elapsed))
        yaw = max(-MAX_ANGULAR, min(MAX_ANGULAR, yaw))
        if vx or vy:
            yaw = max(-WALK_TURN_LIMIT_RADPS, min(WALK_TURN_LIMIT_RADPS, yaw))
        envelope = math.hypot(vx/(MAX_LINEAR if vx >= 0 else WALK_BACK_LIMIT_MPS),
                              vy/WALK_SIDE_LIMIT_MPS)
        scale = max(1.0, envelope, math.hypot(vx, vy)/speed_limit)
        scale *= 1.0+WALK_TURN_SLOWDOWN*abs(yaw)
        vx, vy = vx/scale, vy/scale
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        world = (vx*c-vy*s, vx*s+vy*c)
        delta = (world[0]-self.world[0], world[1]-self.world[1])
        size = math.hypot(*delta)
        fraction = min(1.0, WALK_ACCEL_MPS2*elapsed/size) if size else 1.0
        wx, wy = (a+fraction*b for a, b in zip(self.world, delta))
        yaw = self.yaw+max(-WALK_YAW_ACCEL_RADPS2*elapsed,
                          min(WALK_YAW_ACCEL_RADPS2*elapsed, yaw-self.yaw))
        # 变向期间旧方向的速度也必须满足当前机体方向的速度包络。
        vx, vy = wx*c+wy*s, -wx*s+wy*c
        scale = max(1.0, math.hypot(vx/(MAX_LINEAR if vx >= 0 else WALK_BACK_LIMIT_MPS),
                                  vy/WALK_SIDE_LIMIT_MPS), math.hypot(vx, vy)/speed_limit)
        return vx/scale, vy/scale, yaw

    def record(self, pose, now, vx, vy, yaw):
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        self.world = (vx*c-vy*s, vx*s+vy*c)
        self.yaw, self.at = yaw, now
