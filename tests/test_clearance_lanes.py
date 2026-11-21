import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context, Backend
from src.framework.types import BallState, RobotState, Pose2D
from src.player import Player
from src.param import GOAL_CLEAR_SIDE_M, FIELD_SAFE_MARGIN_M


class ClearanceLanes(unittest.TestCase):
    def setUp(self):
        self.p = Player(1,SimpleNamespace(team_id=1),Backend())
        self.p.context = replace(context(),ball=BallState(-5.,0.),teammates={
            1:RobotState(1,Pose2D(-5.5,0.))},opponents={})

    def test_blocked_same_side_uses_other_forward_lane(self):
        middle = (self.p.context.field.length/2-FIELD_SAFE_MARGIN_M-5.)/2
        self.p.context = replace(self.p.context,opponents={
            4:RobotState(4,Pose2D(middle,0.)),
            5:RobotState(5,Pose2D(middle,GOAL_CLEAR_SIDE_M/2))})
        target = self.p.clearance_target()
        self.assertLess(target[1],0.)
        self.assertGreater(target[0],self.p.context.ball.x)

    def test_clearance_does_not_aim_through_teammate(self):
        middle = (self.p.context.field.length/2-FIELD_SAFE_MARGIN_M-5.)/2
        self.p.context = replace(self.p.context,teammates={**self.p.context.teammates,
            2:RobotState(2,Pose2D(middle,0.))})
        self.assertGreater(abs(self.p.clearance_target()[1]),0.)

    def test_empty_field_clearance_heads_upfield_without_short_diagonal_exit(self):
        target = self.p.clearance_target()
        self.assertEqual(target,(self.p.context.field.length/2-FIELD_SAFE_MARGIN_M,0.))

    def test_equally_clear_lanes_keep_direction_but_new_block_can_change_it(self):
        target = self.p.clearance_target()
        self.p.context = replace(self.p.context,now=.1,ball=BallState(-4.95,.02))
        self.assertEqual(self.p.clearance_target(),target)
        self.p.context = replace(self.p.context,now=.2,opponents={
            4:RobotState(4,Pose2D((target[0]-4.95)/2,(target[1]+.02)/2))})
        self.assertNotEqual(self.p.clearance_target(),target)
        self.p.reset_navigation()
        self.assertIsNone(self.p._clear_aim)


if __name__ == '__main__':
    unittest.main()
