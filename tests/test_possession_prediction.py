"""主动控球版：比较到球时间、截球通道和出脚机会。"""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context,Backend
from src.framework.types import BallState,RobotState,Pose2D
from src.player import Player
from src.tactics import TeamTactics
from src.utils.ball_motion import BallMotion


class PossessionPrediction(unittest.TestCase):
    def test_turning_changes_arrival_time_for_same_distance(self):
        ctx=replace(context(),ball=BallState(2.,0.))
        motion=BallMotion();motion.position=(2.,0.)
        _,forward=motion.intercept(ctx,Pose2D(0.,0.,0.))
        _,reverse=motion.intercept(ctx,Pose2D(0.,0.,math.pi))
        self.assertGreater(reverse,forward)

    def test_crossing_ball_uses_future_point_instead_of_chasing_old_position(self):
        ctx=replace(context(),ball=BallState())
        motion=BallMotion();motion.position=(0.,0.);motion.velocity=(2.,0.)
        point,_=motion.intercept(ctx,Pose2D(2.,1.))
        self.assertGreater(point[0],.5)
        self.assertLess(point[0],2.5)

    def test_open_looking_pass_rejected_when_opponent_can_arrive_first(self):
        ctx=replace(context(),ball=BallState(),teammates={
            1:RobotState(1,Pose2D(-.2,0.)),2:RobotState(2,Pose2D(5.,0.))},
            opponents={4:RobotState(4,Pose2D(3.,1.))})
        players=[Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2)]
        for p in players:p.context=ctx
        motion=BallMotion();motion.update(ctx)
        self.assertIsNone(TeamTactics()._pass_option(ctx,players[0],[players[1]],motion))

    def test_near_ball_shoots_even_with_blocked_goal_and_long_distance(self):
        ctx=replace(context(),ball=BallState(-2.,0.),teammates={1:RobotState(1,Pose2D(-2.2,0.))},
                    opponents={4:RobotState(4,Pose2D(0.,0.))})
        p=Player(1,SimpleNamespace(team_id=1),Backend());p.context=ctx
        motion=BallMotion();motion.update(ctx)
        action=TeamTactics()._ball_intent(ctx,p,(-2.,0.),[],motion,'possession')
        self.assertEqual(action.role,'shoot')
        self.assertGreater(action.kick_target[0],ctx.ball.x)


if __name__=='__main__':unittest.main()
