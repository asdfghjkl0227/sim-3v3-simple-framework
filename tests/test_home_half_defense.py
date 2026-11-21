"""球过中线后的立即分工，独立于持球估计、比分和旧进攻计划。"""
import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D, TeamState
from src.player import Player
from src.tactics import TeamTactics
from src.utils.ball_motion import BallMotion


class HomeHalfDefense(unittest.TestCase):
    def setUp(self):
        self.ctx = replace(context(),ball=BallState(-.01,.4),teammates={
            1:RobotState(1,Pose2D(-6.5,0.)),2:RobotState(2,Pose2D(.5,.4)),
            3:RobotState(3,Pose2D(2.,-2.))})
        self.players = [Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        self.tactics = TeamTactics()
        self.motion = BallMotion()

    def plan(self,ctx):
        for p in self.players:p.context=ctx
        self.motion.position=(ctx.ball.x,ctx.ball.y)
        return self.tactics.plan(ctx,self.players,self.motion)

    def test_own_half_possession_does_not_order_automatic_retreat(self):
        intents=self.plan(self.ctx)
        self.assertEqual(intents[3].role,'triangle:wing')
        self.assertGreater(intents[3].target[0],self.ctx.ball.x)
        self.assertFalse(any(i.role.startswith('transition:') or i.role=='defend:quarter' for i in intents.values()))

    def test_score_never_enables_parking(self):
        for ours,theirs in ((0,3),(1,1),(4,0)):
            ctx=replace(self.ctx,game=replace(self.ctx.game,teams=(
                TeamState(team_number=1,score=ours),TeamState(team_number=2,score=theirs))))
            intents=self.plan(ctx)
            self.assertEqual(self.tactics.mode,'possession')
            self.assertGreater(intents[3].target[0],ctx.ball.x)

    def test_losing_ball_keeps_one_challenger_and_one_outlet_pressure(self):
        ctx=replace(self.ctx,ball=BallState(-2.,0.),opponents={4:RobotState(4,Pose2D(-2.,.1))})
        intents=self.plan(ctx)
        self.assertTrue(any(i.role in ('press:tackle','press:approach') for i in intents.values()))
        self.assertEqual(sum(i.role=='press:outlet' for i in intents.values()),1)
        self.assertEqual(sum(i.role=='cover:goal' for i in intents.values()),1)

    def test_only_one_field_player_keeps_press_and_goal_cover(self):
        self.players=self.players[:2]
        intents=self.plan(self.ctx)
        self.assertEqual(len(intents),2)
        self.assertEqual(intents[1].role,'cover:goal')
        self.assertIsNotNone(intents[2].kick_target)

    def test_expired_counterpress_does_not_disable_challenge(self):
        self.tactics.transition_active=True
        self.tactics.counterpress_until=-1.
        intents=self.plan(self.ctx)
        self.assertIsNotNone(intents[2].kick_target)
        self.assertFalse(any(i.role.startswith('transition:') for i in intents.values()))


if __name__=='__main__':unittest.main()
