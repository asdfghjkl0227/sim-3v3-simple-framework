"""对方重启时保持合法前场封堵，官方放行后立即争球。"""
import unittest
import math
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D, GameState, SetPlay
from src.player import Player
from src.utils.ball_motion import BallMotion
from src.param import *


class RestartScreen(unittest.TestCase):
    def setUp(self):
        self.store = SimpleNamespace()
        SoccerSimAgent.init_store(None,self.store)
        self.players = [Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]

    def tick(self,ctx):
        for p in self.players:p.context=ctx
        SoccerSimAgent.play(ctx,self.players,self.store)

    def test_opponent_kickoff_keeps_legal_central_screen(self):
        ctx = replace(context(kicking_team=2,secondary_time=10),teammates={
            1:RobotState(1,Pose2D(-6.5,0.)),2:RobotState(2,Pose2D(-.8,2.3)),
            3:RobotState(3,Pose2D(-.8,-2.3))})
        self.tick(ctx)
        front = self.players[1:]
        self.assertEqual({p._nav_goal for p in front},
                         {(OPP_KICKOFF_FRONT_X_M,0.),(OPP_KICKOFF_BACK_X_M,0.)})
        for p in front:
            x,y = p._nav_goal
            self.assertLess(x,0.)
            self.assertGreaterEqual(math.hypot(x,y),OPPONENT_RESTART_AVOID_M+RULE_TARGET_MARGIN_M)
            self.assertFalse(p.is_kicking)
        # 官方未放行不能凭本地时间开抢。
        self.tick(replace(ctx,now=100.))
        self.assertFalse(any(p.is_kicking for p in self.players))

    def test_opponent_goal_kick_does_not_force_everyone_to_own_penalty_area(self):
        ctx = replace(context(kicking_team=2,set_play=SetPlay.GOAL_KICK),
                      ball=BallState(6.,0.))
        self.tick(ctx)
        targets = [p._nav_goal for p in self.players if p._nav_goal is not None]
        self.assertGreaterEqual(sum(t[0]>0. for t in targets),2)
        self.assertFalse(any(p.is_kicking for p in self.players))

    def test_ready_already_has_two_central_layers_before_opponent_kicks(self):
        ctx = replace(context(GameState.READY,kicking_team=2),teammates={
            1:RobotState(1,Pose2D(-6.5,0.)),2:RobotState(2,Pose2D(-.8,2.3)),
            3:RobotState(3,Pose2D(-.8,-2.3))})
        self.tick(ctx)
        self.assertEqual(set(self.store.ready_targets.values()),{
            (READY_BASE_X_M,0.),(OPP_KICKOFF_FRONT_X_M,0.),(OPP_KICKOFF_BACK_X_M,0.)})
        self.assertFalse(any(p.is_kicking for p in self.players))

    def test_powerful_central_kick_bypasses_front_but_back_defender_takes_over(self):
        ctx = replace(context(),now=1.,ball=BallState(-2.8,0.),teammates={
            1:RobotState(1,Pose2D(-6.5,0.)),2:RobotState(2,Pose2D(-2.3,0.)),
            3:RobotState(3,Pose2D(-3.8,0.))},opponents={
            4:RobotState(4,Pose2D(0.,0.))})
        for p in self.players:p.context=ctx
        tactics = self.store.tactics
        tactics.anchor_id = 1
        tactics.start_opponent_restart_pressure(ctx,kickoff=True)
        tactics.attacker_id = 2  # 旧前层不能锁住已被越过的争球角色。
        motion = BallMotion();motion.position=(-2.8,0.);motion.velocity=(-4.,0.)
        intents = tactics.plan(ctx,self.players,motion)
        self.assertIsNotNone(intents[3].kick_target)
        self.assertEqual(intents[2].role,'press:outlet')
        self.assertLessEqual(abs(intents[2].target[1]),PRESS_SUPPORT_WIDTH_M)
        self.assertLess(intents[2].target[0],ctx.ball.x)
        self.assertIsNotNone(intents[3].kick_target)  # 抢球者直接截球，另一人转身回防。
        self.assertGreater(intents[2].target[0],-ctx.field.length/2)
        self.assertEqual(sum(i.role=='cover:goal' for i in intents.values()),1)

    def test_ready_returns_facing_route_and_preserves_final_positions(self):
        ctx = replace(context(GameState.READY),teammates={
            1:RobotState(1,Pose2D(-6.,0.)),2:RobotState(2,Pose2D(4.,2.3)),
            3:RobotState(3,Pose2D(4.,-2.3))})
        self.tick(ctx)
        for p in self.players[1:]:
            velocity = p._backend.calls[-1]
            self.assertLess(velocity[1],0.)
            self.assertGreater(abs(velocity[3]),0.)
            self.assertEqual(p._nav_goal,self.store.ready_targets[p.id])


if __name__ == '__main__':
    unittest.main()
