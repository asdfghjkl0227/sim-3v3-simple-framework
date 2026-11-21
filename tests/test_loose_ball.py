"""对方出球后优先赢得自由球，不能被回撤状态或旧球权估计锁住。"""
import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D, TeamState
from src.player import Player
from src.param import *


class LooseBallScenarios(unittest.TestCase):
    def setUp(self):
        self.store=SimpleNamespace()
        SoccerSimAgent.init_store(None,self.store)
        self.players=[Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        self.ctx=replace(context(),now=1.,ball=BallState(-2.,0.),teammates={
            1:RobotState(1,Pose2D(-3.6,.4)),2:RobotState(2,Pose2D(1.,2.5)),
            3:RobotState(3,Pose2D(-6.5,0.))},opponents={4:RobotState(4,Pose2D(2.,0.))})
        # 反抢已过期，旧持球估计尚未消退；球距我方最近者仍大于1.4米。
        tactics=self.store.tactics
        tactics.possession='theirs'
        tactics.possession_at=.9
        tactics.transition_active=True
        tactics.counterpress_until=.8

    def plan(self,ctx,velocity=(0.,0.)):
        for p in self.players:p.context=ctx
        motion=self.store.ball_motion
        motion.position=(ctx.ball.x,ctx.ball.y)
        motion.velocity=velocity
        return self.store.tactics.plan(ctx,self.players,motion)

    def test_winning_race_overrides_expired_retreat_and_stale_opponent_possession(self):
        intents=self.plan(self.ctx,(-1.2,0.))
        self.assertEqual(self.store.tactics.possession,'theirs')
        self.assertEqual(intents[1].role,'press:tackle')
        self.assertIsNotNone(intents[1].kick_target)
        self.assertGreater(intents[1].kick_target[0],self.ctx.ball.x)
        self.assertEqual(sum(a.role=='press:tackle' for a in intents.values()),1)
        self.assertTrue(any(a.role=='cover:goal' for a in intents.values()))
        self.assertLessEqual(self.store.tactics.counterpress_until,.8)

    def test_slow_free_ball_with_head_start_immediately_starts_clearance(self):
        for p in self.players:p.context=self.ctx
        SoccerSimAgent.play(self.ctx,self.players,self.store)
        winner=self.players[0]
        self.assertEqual(winner.action,'press:tackle')
        kick=next(c for c in winner._backend.calls if c[0]=='kick')
        self.assertGreater(math.cos(kick[1]+winner.pose.theta),0.)
        self.assertEqual(kick[2],SHOT_POWER)
        self.assertTrue(winner.is_kicking)

    def test_opponent_wins_race_keeps_recovery(self):
        ctx=replace(self.ctx,opponents={4:RobotState(4,Pose2D(-.8,0.))})
        intents=self.plan(ctx)
        self.assertFalse(any(a.role.startswith('loose:') for a in intents.values()))
        self.assertTrue(any(a.role=='press:outlet' for a in intents.values()))

    def test_keeper_can_sweep_with_replacement_cover(self):
        ctx=replace(self.ctx,ball=BallState(-4.,0.),teammates={
            1:RobotState(1,Pose2D(0.,1.)),2:RobotState(2,Pose2D(3.,-1.)),
            3:RobotState(3,Pose2D(-6.,0.))})
        intents=self.plan(ctx)
        self.assertEqual(intents[3].role,'penalty:clear')
        self.assertEqual(intents[1].role,'cover:goal')
        self.assertEqual(self.store.tactics.anchor_id,1)
        self.assertGreater(intents[3].kick_target[0],0.)

    def test_leading_two_goals_still_clears_free_ball(self):
        ctx=replace(self.ctx,game=replace(self.ctx.game,teams=(
            TeamState(team_number=1,score=2),TeamState(team_number=2,score=0))))
        intents=self.plan(ctx)
        self.assertEqual(self.store.tactics.mode,'possession')
        self.assertEqual(intents[1].role,'press:tackle')

    def test_blocked_center_chooses_forward_or_wide_clearance(self):
        ctx=replace(self.ctx,opponents={
            4:RobotState(4,Pose2D(0.,0.)),5:RobotState(5,Pose2D(1.,-1.))})
        intents=self.plan(ctx)
        self.assertEqual(intents[1].role,'press:tackle')
        target=intents[1].kick_target
        self.assertEqual(target[1],0.)  # 最新要求拿到出脚机会直接射门。
        self.assertGreater(target[0],ctx.ball.x)

    def test_opponent_reaching_ball_cancels_race_without_reopening_counterpress(self):
        self.plan(self.ctx,(-1.2,0.))
        caught=replace(self.ctx,now=1.1,opponents={4:RobotState(4,Pose2D(-1.8,0.))})
        intents=self.plan(caught)
        self.assertFalse(any(a.role.startswith('loose:') for a in intents.values()))
        self.assertTrue(any(a.role=='press:outlet' for a in intents.values()))
        self.assertLessEqual(self.store.tactics.counterpress_until,.8)


if __name__=='__main__':
    unittest.main()
