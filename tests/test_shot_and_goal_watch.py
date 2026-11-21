"""门将不被通用脱困带离门线；中远射选点保持连续。"""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D
from src.player import Player
from src.tactics import TeamTactics
from src.param import *


class ShotAndGoalWatch(unittest.TestCase):
    def test_close_shot_uses_open_side_of_central_keeper(self):
        ctx = replace(context(), ball=BallState(4.,0.), opponents={
            4:RobotState(4,Pose2D(context().field.length/2-.4,0.))})
        tactics = TeamTactics()
        target = tactics._goal_target(ctx)
        self.assertGreater(abs(target[1]),.5)
        self.assertLess(abs(target[1]),ctx.field.goal_width/2)
        # 小范围门将观测抖动，不应每帧换侧，扰乱SDK调整出脚朝向。
        for i in range(1,12):
            moved = replace(ctx, now=i/30, opponents={
                4:RobotState(4,Pose2D(ctx.field.length/2-.4,.02 if i%2 else -.02))})
            self.assertEqual(tactics._goal_target(moved),target)

    def test_keeper_stays_available_after_short_idle_and_shuffles_on_line(self):
        store = SimpleNamespace()
        SoccerSimAgent.init_store(None,store)
        players = [Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        ctx = replace(context(), ball=BallState(-4.83,-.86),teammates={
            1:RobotState(1,Pose2D(-6.5,-.2)),2:RobotState(2,Pose2D(-3.,-.9)),
            3:RobotState(3,Pose2D(-2.,2.))},opponents={
            4:RobotState(4,Pose2D(-4.7,-.86))})
        for t in (0.,.1,1.,1.7,2.):
            current = replace(ctx,now=t,teammates={**ctx.teammates,
                2:RobotState(2,Pose2D(-3.-.35*t,-.9))})
            for p in players:p.context=current
            SoccerSimAgent.play(current,players,store)
            self.assertIn(1,store.active_ids)
            self.assertIn(players[0].action,('cover:goal','mark:goal_cover'))
            self.assertEqual(store.tactics.anchor_id,1)
        keeper = players[0]
        keeper.context = replace(ctx,now=KEEPER_IDLE_SHUFFLE_S+.1)
        with patch.object(keeper,'walk_to') as walk:
            keeper.defend_goal((-6.5,-.2))
            target = walk.call_args.args[0]
            self.assertEqual(target[0],-6.5)
            self.assertAlmostEqual(abs(target[1]+.2),KEEPER_IDLE_SHUFFLE_M)

    def test_unfaced_pursuit_turns_towards_travel(self):
        player = Player(1,SimpleNamespace(team_id=1),Backend())
        player.context = replace(context(), teammates={1:RobotState(1,Pose2D(0.,0.,0.))})
        player.walk_to((0.,3.))
        velocity = player._backend.calls[-1]
        self.assertEqual(velocity[0],'velocity')
        self.assertGreater(velocity[3],0.)


if __name__ == '__main__':
    unittest.main()
