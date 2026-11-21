"""开球出球与接应必须共用球路，官方放行之前队友不碰球。"""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context,Backend,SoccerSimAgent
from src.framework.types import BallState,RobotState,Pose2D,GameState
from src.player import Player
from src.param import *


class KickoffLink(unittest.TestCase):
    def setUp(self):
        self.store=SimpleNamespace()
        SoccerSimAgent.init_store(None,self.store)
        self.players=[Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        self.ctx=replace(context(kicking_team=1,secondary_time=10),teammates={
            1:RobotState(1,Pose2D(-.4,-.3)),2:RobotState(2,Pose2D(-2.3,0.)),
            3:RobotState(3,Pose2D(-6.5,0.))})

    def tick(self,ctx):
        for p in self.players:p.context=ctx
        SoccerSimAgent.play(ctx,self.players,self.store)

    def test_taker_starts_low_power_control_push_without_waiting(self):
        self.tick(self.ctx)
        self.assertTrue(self.players[0].is_kicking)
        self.assertEqual(self.players[0].action,'restart:push')
        self.assertEqual(self.store.kickoff_receiver,2)
        self.assertFalse(self.players[1].rule_can_kick)

    def test_first_kick_matches_ready_receiver_side_and_uses_minimum_power(self):
        for side in (1.,-1.):
            self.setUp()
            ctx=replace(self.ctx,teammates={**self.ctx.teammates,
                2:RobotState(2,Pose2D(KICKOFF_RECEIVER_X_M,side*KICKOFF_RECEIVER_Y_M))})
            self.tick(ctx)
            command=next(c for c in self.players[0]._backend.calls if c[0]=='kick')
            direction=command[1]+self.players[0].pose.theta
            self.assertAlmostEqual(direction,side*KICKOFF_PASS_ANGLE_RAD)
            self.assertEqual(command[2],RESTART_PUSH_POWER)
            self.assertGreater(math.cos(direction),0.)
            self.assertFalse(self.players[0].kick_can_score(direction))

    def test_released_ball_in_own_half_keeps_nearer_taker_and_one_recoverer(self):
        ctx=replace(self.ctx,teammates={**self.ctx.teammates,
            2:RobotState(2,Pose2D(KICKOFF_RECEIVER_X_M,KICKOFF_RECEIVER_Y_M))})
        self.tick(ctx)
        # 主罚仍更近时连续追球，不能因为预定接应人就放弃最近触球机会。
        self.tick(replace(ctx,now=.2,ball=BallState(-.01,.6),game=context().game))
        self.assertEqual(self.players[0].action,'kickoff:contest')
        self.assertEqual(self.players[1].action,'triangle:wing')
        self.assertTrue(self.players[0].is_kicking)
        self.assertEqual(sum(p.is_kicking for p in self.players),1)

    def test_receiver_setup_has_bounded_wait_and_set_still_stops_everyone(self):
        self.tick(self.ctx)
        self.tick(replace(self.ctx,now=KICKOFF_RECEIVER_WAIT_S+.1))
        self.assertTrue(self.players[0].is_kicking)
        self.tick(replace(self.ctx,now=6.,game=replace(self.ctx.game,state=GameState.SET)))
        self.assertFalse(any(p.is_kicking for p in self.players))

    def test_timeout_without_ball_movement_does_not_lock_far_receiver(self):
        self.tick(self.ctx)
        self.tick(replace(self.ctx,now=10.,game=context().game))
        self.assertIsNone(self.store.tactics.kickoff_receiver_id)
        self.assertNotEqual(self.store.tactics.attacker_id,self.store.kickoff_receiver)

    def test_official_release_keeps_taker_sdk_task_and_push_heading(self):
        ctx=replace(self.ctx,teammates={**self.ctx.teammates,
            2:RobotState(2,Pose2D(KICKOFF_RECEIVER_X_M,KICKOFF_RECEIVER_Y_M))})
        self.tick(ctx)
        taker=self.players[0]
        self.assertTrue(taker.is_kicking)
        taker._backend.calls.clear()
        self.tick(replace(ctx,now=.2,ball=BallState(.2,.6),game=context().game))
        self.assertTrue(taker.is_kicking)
        self.assertFalse(any(c[0]=='release' for c in taker._backend.calls))
        command=next(c for c in taker._backend.calls if c[0]=='kick')
        self.assertAlmostEqual(command[1]+taker.pose.theta,KICKOFF_PASS_ANGLE_RAD)
        self.assertEqual(command[2],RESTART_PUSH_POWER)

    def test_nearer_receiver_can_take_over_without_two_kickers(self):
        ctx=replace(self.ctx,teammates={**self.ctx.teammates,
            2:RobotState(2,Pose2D(KICKOFF_RECEIVER_X_M,KICKOFF_RECEIVER_Y_M))})
        self.tick(ctx)
        released=replace(ctx,now=.2,ball=BallState(.4,1.2),game=context().game,
                         teammates={**ctx.teammates,2:RobotState(2,Pose2D(.45,1.2))})
        self.tick(released)
        self.assertEqual(self.store.tactics.attacker_id,2)
        self.assertFalse(self.players[0].is_kicking)
        self.assertEqual(sum(p.is_kicking for p in self.players),1)


if __name__=='__main__':unittest.main()
