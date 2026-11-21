"""直接入场、角球通道、主动抢断及卡住恢复；仅使用假后端。"""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D, GameState
from src.framework.types import SetPlay, TeamState, PlayerState, Penalty
from src.player import Player
from src.param import *


def segment_distance(point, start, end):
    dx, dy = end[0]-start[0], end[1]-start[1]
    length2 = dx*dx+dy*dy
    t = max(0., min(1., ((point[0]-start[0])*dx+(point[1]-start[1])*dy)/length2)) if length2 else 0.
    return math.hypot(point[0]-start[0]-t*dx, point[1]-start[1]-t*dy)


class EntryCornerPressure(unittest.TestCase):
    def setUp(self):
        self.store = SimpleNamespace()
        SoccerSimAgent.init_store(None, self.store)
        self.players = [Player(i, SimpleNamespace(team_id=1), Backend()) for i in (1, 2, 3)]

    def tick(self, ctx):
        for p in self.players:
            p.context = ctx
        SoccerSimAgent.play(ctx, self.players, self.store)

    def corner(self, positions, now=0.):
        return replace(context(set_play=SetPlay.CORNER_KICK, kicking_team=1),
                       now=now, ball=BallState(6.95, 4.45), teammates={
                           i: RobotState(i, Pose2D(*p)) for i, p in enumerate(positions, 1)})

    def test_ready_direct_diagonal_entry_reaches_central_layers(self):
        ctx = replace(context(GameState.READY, kicking_team=1), teammates={
            1: RobotState(1, Pose2D(-5., 5.)),
            2: RobotState(2, Pose2D(-6., 0.)),
            3: RobotState(3, Pose2D(-5., -5.))})
        self.tick(ctx)
        self.assertEqual(set(self.store.ready_targets.values()), {
            (READY_BASE_X_M, 0.), (OPP_KICKOFF_FRONT_X_M,0.),
            (OPP_KICKOFF_BACK_X_M,0.)})
        for p in self.players:
            self.assertEqual(p._nav_goal, self.store.ready_targets[p.id])
        self.assertGreater(self.players[0]._backend.calls[-1][1], 0.)
        self.assertLess(self.players[0]._backend.calls[-1][2], 0.)
        self.assertGreater(self.players[2]._backend.calls[-1][1], 0.)
        self.assertGreater(self.players[2]._backend.calls[-1][2], 0.)
        for p in self.players:
            self.assertEqual(p._nav_goal, self.store.ready_targets[p.id])
        vx, vy = self.players[0]._backend.calls[-1][1:3]
        self.assertGreater(vx, 0.)
        self.assertLess(vy, 0.)

    def test_unblocked_corner_keeps_one_stable_nearest_taker(self):
        ctx = self.corner([(-4., 0.), (6., 3.8), (3., -2.)])
        self.tick(ctx)
        self.assertEqual(self.store.kickoff_taker, 2)
        self.assertEqual([p.id for p in self.players if p.rule_can_kick], [2])
        self.assertEqual([p.id for p in self.players if p.is_kicking], [2])
        # 略近的新候选不打断持续主罚任务。
        ctx = replace(ctx, now=.1, teammates={**ctx.teammates, 3: RobotState(3, Pose2D(6.8, 3.1))})
        self.tick(ctx)
        self.assertEqual(self.store.kickoff_taker, 2)

    def test_blocked_corner_switches_to_robot1_and_clears_actual_lane(self):
        ctx = self.corner([(3., 2.), (6., 3.8), (2., -2.)])
        self.tick(ctx)
        self.assertEqual(self.store.kickoff_taker, 2)
        ctx = replace(ctx, now=.1, teammates={**ctx.teammates,
            3: RobotState(3, Pose2D(6.5, 4.1))})
        self.tick(ctx)
        self.assertTrue(self.store.corner_robot1_locked)
        self.assertEqual(self.store.kickoff_taker, 1)
        self.assertEqual([p.id for p in self.players if p.is_kicking], [1])
        self.assertIn(('release',), self.players[1]._backend.calls)
        start, end = (3., 2.), (ctx.ball.x, ctx.ball.y)
        targets = []
        for p in self.players[1:]:
            self.assertEqual(p.action, 'corner:yield')
            self.assertIsNotNone(p._nav_goal)
            targets.append(p._nav_goal)
            self.assertGreaterEqual(segment_distance(p._nav_goal, start, end),
                                    CORNER_LANE_RADIUS_M+RULE_TARGET_MARGIN_M-1e-9)
            vx, vy = p._backend.calls[-1][1:3]
            self.assertGreater(math.hypot(vx, vy), 0.)
            # 目标与初始速度都必须增加离通道的距离。
            before = segment_distance((p.pose.x, p.pose.y), start, end)
            after = segment_distance((p.pose.x+vx*.1, p.pose.y+vy*.1), start, end)
            self.assertGreater(after, before)
        self.assertGreaterEqual(math.dist(*targets), READY_TEAM_SEPARATION_M)
        clear = replace(ctx, now=.2, teammates={**ctx.teammates,
            2: RobotState(2, Pose2D(3., -2.)), 3: RobotState(3, Pose2D(5., 1.))})
        self.tick(clear)
        self.assertEqual(self.store.kickoff_taker, 1)

    def test_unavailable_robot1_cannot_be_selected_for_corner(self):
        ctx = self.corner([(3., 2.), (6., 3.8), (2., -2.)])
        self.tick(ctx)
        penalties = (PlayerState(penalty=Penalty.SENT_OFF), PlayerState(), PlayerState())
        ctx = replace(ctx, now=.1, game=replace(ctx.game, teams=(
            TeamState(team_number=1, players=penalties),)), teammates={**ctx.teammates,
                3: RobotState(3, Pose2D(6.5, 4.1))})
        self.tick(ctx)
        self.assertNotEqual(self.store.kickoff_taker, 1)
        self.assertEqual(sum(p.rule_can_kick for p in self.players), 1)
        self.assertFalse(self.players[0].is_kicking)

    def test_touchline_corner_queue_yields_infield_on_both_sides(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                self.setUp()
                ctx = self.corner([(4., side*4.4), (6., side*4.4), (2., -side*2.)])
                ctx = replace(ctx, ball=BallState(6.95, side*4.45))
                self.tick(ctx)
                ctx = replace(ctx, now=.1, teammates={**ctx.teammates,
                    3: RobotState(3, Pose2D(6.5, side*4.43))})
                self.tick(ctx)
                self.assertEqual(self.store.kickoff_taker, 1)
                for p in self.players[1:]:
                    self.assertIsNotNone(p._nav_goal)
                    self.assertLessEqual(abs(p._nav_goal[1]), ctx.field.width/2-FIELD_SAFE_MARGIN_M)
                    self.assertGreaterEqual(segment_distance(p._nav_goal, (4., side*4.4),
                        (ctx.ball.x, ctx.ball.y)), CORNER_LANE_RADIUS_M+RULE_TARGET_MARGIN_M-1e-9)
                    vx, vy = p._backend.calls[-1][1:3]
                    self.assertLess(side*vy, 0.)

    def test_robot1_reposition_does_not_reselect_inactive_taker(self):
        ctx = self.corner([(3., 2.), (6., 3.8), (2., -2.)])
        self.tick(ctx)
        ctx = replace(ctx, now=.1, teammates={**ctx.teammates,
            3: RobotState(3, Pose2D(6.5, 4.1))})
        with patch.object(self.players[0], 'recover_inactivity', return_value=True):
            self.tick(ctx)
        self.assertTrue(self.store.corner_robot1_locked)
        self.assertNotEqual(self.store.kickoff_taker, 1)
        self.assertEqual([p.id for p in self.players if p.is_kicking], [self.store.kickoff_taker])

    def defense(self, player_x):
        return replace(context(), ball=BallState(-1., 0.), teammates={
            1: RobotState(1, Pose2D(player_x, 0.)),
            2: RobotState(2, Pose2D(0., 3.)),
            3: RobotState(3, Pose2D(-6.5, 0.))}, opponents={
            4: RobotState(4, Pose2D(-.8, .1)),
            5: RobotState(5, Pose2D(-1., 3.))})

    def test_close_holder_is_tackled_while_another_player_covers_goal(self):
        self.tick(self.defense(-2.))
        self.assertEqual(self.players[0].action, 'press:tackle')
        self.assertTrue(self.players[0].is_kicking)
        self.assertEqual(self.players[2].action, 'cover:goal')
        self.assertEqual(sum(p.action == 'press:tackle' for p in self.players), 1)

    def test_distant_defender_targets_ball_instead_of_goalside_waiting_point(self):
        self.tick(self.defense(-3.5))
        defender = self.players[0]
        self.assertEqual(defender.action, 'press:approach')
        self.assertEqual(defender._nav_goal, (-1., 0.))
        self.assertTrue(defender.ball_access)
        self.assertGreater(defender._backend.calls[-1][1], 0.)

    def test_stationary_live_ball_repositions_quickly_towards_play(self):
        p = self.players[0]
        p.context = self.defense(-2.)
        self.assertFalse(p.recover_inactivity(emergency=True))
        p.context = replace(p.context, now=ACTIVE_STALL_S+.01)
        self.assertTrue(p.recover_inactivity(emergency=True))
        self.assertEqual(p.action, 'unstick:reposition')
        self.assertGreater(p._rescue_target[0], p.pose.x)
        self.assertGreater(math.hypot(*p._backend.calls[-1][1:3]), 0.)

    def test_ball_progress_resets_stall_without_robot_movement(self):
        p = self.players[0]
        p.context = self.defense(-2.)
        self.assertFalse(p.recover_inactivity(emergency=True))
        for step in range(1, 5):
            p.context = replace(p.context, now=step*(ACTIVE_STALL_S-.1),
                                ball=BallState(-1.+step*(INACTIVITY_PROGRESS_M+.01), 0.))
            self.assertFalse(p.recover_inactivity(emergency=True))
        self.assertFalse(p._backend.calls)


if __name__ == '__main__':
    unittest.main()
