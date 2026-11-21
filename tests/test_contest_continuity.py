"""实赛中近身处理者反复换人、取消 SDK 踢球任务的回归。"""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D
from src.player import Player


class ContestContinuity(unittest.TestCase):
    def setUp(self):
        self.store = SimpleNamespace()
        SoccerSimAgent.init_store(None, self.store)
        self.players = [Player(i, SimpleNamespace(team_id=1), Backend()) for i in (1, 2, 3)]
        self.ctx = replace(context(), now=1., ball=BallState(6.19, 2.31), teammates={
            1: RobotState(1, Pose2D(-6.47, 0.)),
            2: RobotState(2, Pose2D(5.5, 2.1)),
            3: RobotState(3, Pose2D(5.5, 2.7))}, opponents={
                4: RobotState(4, Pose2D(6.19, 2.4))})
        self.store.tactics.possession = 'theirs'
        self.store.tactics.transition_active = True
        self.store.tactics.counterpress_until = .8

    def tick(self, ctx):
        for p in self.players:
            p.context = ctx
        SoccerSimAgent.play(ctx, self.players, self.store)

    def test_near_equal_distances_do_not_interrupt_tackler_each_frame(self):
        self.tick(self.ctx)
        self.assertEqual(self.store.tactics.attacker_id, 2)
        releases = sum(c[0] == 'release' for c in self.players[1]._backend.calls)
        for i in range(1, 13):
            ctx = replace(self.ctx, now=1.+i/30, teammates={**self.ctx.teammates,
                3: RobotState(3, Pose2D(5.7 if i % 2 else 5.4, 2.7))})
            self.tick(ctx)
            self.assertEqual(self.store.tactics.attacker_id, 2)
            self.assertEqual([p.id for p in self.players if p.is_kicking], [2])
        self.assertEqual(sum(c[0] == 'release' for c in self.players[1]._backend.calls), releases)

    def test_bypassed_tackler_yields_immediately_to_goalside_teammate(self):
        self.tick(self.ctx)
        ctx = replace(self.ctx, now=1.1, ball=BallState(4., 2.31), teammates={**self.ctx.teammates,
            3: RobotState(3, Pose2D(3.5, 2.3))}, opponents={
                4: RobotState(4, Pose2D(4., 2.4))})
        self.tick(ctx)
        self.assertFalse(self.players[1].is_kicking)
        self.assertEqual(self.players[1].action, 'press:outlet')
        self.assertTrue(self.players[2].is_kicking)

    def test_possession_estimate_loss_keeps_available_handler_candidate(self):
        tactics = self.store.tactics
        for p in self.players:p.context = self.ctx
        tactics.possession = 'ours'
        tactics.attacker_id = 2
        tactics._update_possession(self.ctx, self.players)
        self.assertEqual(tactics.possession, 'theirs')
        self.assertEqual(tactics.attacker_id, 2)
        self.assertTrue(tactics.transition_active)

    def test_loose_ball_prediction_jitter_preserves_winning_handler(self):
        for p in self.players:p.context = self.ctx
        tactics = self.store.tactics
        tactics.attacker_id = 2
        ctx = replace(self.ctx, opponents={4: RobotState(4, Pose2D(9., 0.))})
        # 对手仍晚到，两个我方预测的细小变化不应每帧换人。
        with patch.object(self.store.ball_motion, 'intercept', side_effect=[
                ((6.19, 2.31), 2.), ((6.19, 2.31), .5), ((6.19, 2.31), .2)]):
            race = tactics._loose_ball_race(ctx, self.players, self.store.ball_motion, 1)
        self.assertIsNotNone(race)
        self.assertEqual(race[0].id, 2)


if __name__ == '__main__':
    unittest.main()
