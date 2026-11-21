import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D
from src.player import Player
from src.tactics import TeamTactics
from src.utils.ball_motion import BallMotion


class DefensiveCover(unittest.TestCase):
    def test_far_mark_does_not_pull_second_defender_out_of_own_box(self):
        ctx = replace(context(),ball=BallState(-3.5,-.9),teammates={
            1:RobotState(1,Pose2D(-6.5,0.)),2:RobotState(2,Pose2D(-4.3,-.7)),
            3:RobotState(3,Pose2D(0.,3.))},opponents={
            4:RobotState(4,Pose2D(-3.5,-.8)),5:RobotState(5,Pose2D(0.,3.)),
            6:RobotState(6,Pose2D(4.,-2.))})
        players=[Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        for p in players:p.context=ctx
        tactics=TeamTactics()
        motion=BallMotion();motion.update(ctx)
        intents=tactics.plan(ctx,players,motion)
        self.assertEqual(intents[1].role,'cover:goal')
        self.assertEqual(intents[2].role,'press:tackle')
        self.assertEqual(intents[3].role,'press:outlet')
        self.assertLess(intents[3].target[0],ctx.ball.x)
        self.assertIsNotNone(intents[2].kick_target)

    def test_long_support_turns_to_travel_before_facing_ball(self):
        p=Player(1,SimpleNamespace(team_id=1),Backend())
        p.context=replace(context(),ball=BallState(-1.,0.),
                          teammates={1:RobotState(1,Pose2D(0.,0.,math.pi))})
        p.move_to_position((4.,0.))
        self.assertGreater(abs(p._backend.calls[-1][3]),0.)

    def test_live_goal_mouth_ball_brings_third_player_back_even_if_possession_seems_ours(self):
        ctx = replace(context(),ball=BallState(-6.6,0.),teammates={
            1:RobotState(1,Pose2D(-6.3,0.)),2:RobotState(2,Pose2D(-6.5,.7)),
            3:RobotState(3,Pose2D(-3.,2.))},opponents={
            4:RobotState(4,Pose2D(-4.,0.))})
        players=[Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        for p in players:p.context=ctx
        tactics=TeamTactics();motion=BallMotion();motion.update(ctx)
        intents=tactics.plan(ctx,players,motion)
        self.assertEqual(tactics.possession,'ours')
        self.assertEqual(intents[3].role,'press:outlet')
        self.assertLess(intents[3].target[0],players[2].pose.x)


if __name__ == '__main__':
    unittest.main()
