"""近门主动解围、快射保位及对方重启后的压迫衔接。"""
import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D, GameState
from src.player import Player
from src.tactics import TeamTactics
from src.utils.ball_motion import BallMotion
from src.param import *


class ActiveGoalkeeping(unittest.TestCase):
    def setUp(self):
        self.players = [Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        self.store = SimpleNamespace()
        SoccerSimAgent.init_store(None,self.store)
        self.ctx = replace(context(),ball=BallState(-4.5,0.),teammates={
            1:RobotState(1,Pose2D(-6.5,0.)),2:RobotState(2,Pose2D(-1.,2.)),
            3:RobotState(3,Pose2D(0.,-2.))},opponents={
            4:RobotState(4,Pose2D(-4.4,.1))})

    def plan(self,ctx,velocity=(0.,0.)):
        for p in self.players:p.context=ctx
        motion = self.store.ball_motion
        motion.position = (ctx.ball.x,ctx.ball.y)
        motion.velocity = velocity
        return self.store.tactics.plan(ctx,self.players,motion)

    def tick(self,ctx):
        for p in self.players:p.context=ctx
        SoccerSimAgent.play(ctx,self.players,self.store)

    def test_goalkeeper_attacks_dribbler_even_during_expired_retreat(self):
        self.store.tactics.transition_active = True
        self.store.tactics.counterpress_until = -1.
        intents = self.plan(self.ctx)
        self.assertIsNotNone(intents[1].kick_target)
        self.assertGreater(intents[1].kick_target[0],self.ctx.ball.x)
        self.assertEqual(sum(a.kick_target is not None for a in intents.values()),1)
        self.assertEqual(sum(a.role=='cover:goal' for a in intents.values()),1)

    def test_nearer_field_defender_keeps_challenge_and_keeper_steps_out_to_narrow_angle(self):
        ctx = replace(self.ctx,teammates={**self.ctx.teammates,
            2:RobotState(2,Pose2D(-4.7,.2))})
        intents = self.plan(ctx)
        self.assertIsNotNone(intents[2].kick_target)
        self.assertIsNone(intents[1].kick_target)
        self.assertGreater(intents[1].target[0],-ctx.field.length/2+DEFEND_KEEPER_LINE_M)
        self.assertLess(intents[1].target[0],ctx.ball.x)

    def test_fast_shot_does_not_send_keeper_chasing_away_from_trajectory(self):
        intents = self.plan(self.ctx,(-3.,0.))
        self.assertIsNone(intents[1].kick_target)
        self.assertEqual(intents[1].target[0],-6.5)
        advanced = replace(self.ctx,teammates={**self.ctx.teammates,
            1:RobotState(1,Pose2D(-5.75,0.))})
        intents = self.plan(advanced,(-3.,0.))
        self.assertIsNotNone(intents[1].kick_target)

    def test_ball_at_feet_is_cleared_even_with_fast_incoming_estimate(self):
        ctx = replace(self.ctx,ball=BallState(-6.3,0.))
        intents = self.plan(ctx,(-3.,0.))
        self.assertEqual(intents[1].role,'goal:clear')
        self.assertIsNotNone(intents[1].kick_target)

    def test_keeper_cannot_start_second_kick_while_teammate_clears(self):
        ctx = replace(self.ctx,ball=BallState(-5.,0.),teammates={
            1:RobotState(1,Pose2D(-6.2,0.)),2:RobotState(2,Pose2D(-5.2,.1)),
            3:RobotState(3,Pose2D(-3.,2.))})
        self.tick(ctx)
        self.assertEqual(sum(p.is_kicking for p in self.players),1)
        self.assertTrue(self.players[1].is_kicking)
        self.assertFalse(self.players[0].is_kicking)

    def test_active_kick_gets_time_to_approach_before_inactivity_repositions(self):
        p = self.players[0]
        ctx = replace(self.ctx,ball=BallState(-5.,0.))
        p.context = ctx
        p.rule_can_kick = True
        p.kick(0.,SHOT_POWER)
        self.assertFalse(p.recover_inactivity(emergency=True))
        p.context = replace(ctx,now=ACTIVE_STALL_S+.1)
        self.assertFalse(p.recover_inactivity(emergency=True))
        self.assertTrue(p.is_kicking)
        p.context = replace(ctx,now=KICK_TASK_STALL_S+.1)
        self.assertTrue(p.recover_inactivity(emergency=True))
        self.assertFalse(p.is_kicking)

    def test_official_restart_release_immediately_assigns_press_and_second_cover(self):
        ctx = replace(self.ctx,ball=BallState(),game=context(kicking_team=2,secondary_time=10).game,
                      teammates={**self.ctx.teammates,
                          2:RobotState(2,Pose2D(-.8,2.05)),3:RobotState(3,Pose2D(-.8,-2.05))},
                      opponents={4:RobotState(4,Pose2D(.1,0.))})
        self.tick(ctx)
        self.assertFalse(any(p.is_kicking for p in self.players))
        released = replace(ctx,now=.1,ball=BallState(-.2,.3),game=context().game)
        self.tick(released)
        self.assertEqual(sum(p.action in ('press:tackle','press:approach')
                             for p in self.players),1)
        self.assertEqual(sum(p.action=='press:outlet' for p in self.players),1)
        self.assertGreater(self.store.tactics.opponent_restart_press_until,released.now)
        self.tick(replace(released,now=.2,game=context(GameState.SET).game))
        self.assertFalse(any(p.is_kicking for p in self.players))
        self.assertEqual(self.store.tactics.opponent_restart_press_until,0.)


if __name__ == '__main__':
    unittest.main()
