"""开球追接和失球转换的跨帧回归，使用假后端，不启动SDK。"""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D
from src.player import Player
from src.tactics import TeamTactics
from src.utils.ball_motion import BallMotion
from src.param import *


class TransitionScenarios(unittest.TestCase):
    def setUp(self):
        self.store=SimpleNamespace()
        SoccerSimAgent.init_store(None,self.store)
        self.players=[Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        self.ctx=replace(context(),ball=BallState(0.,0.),teammates={
            1:RobotState(1,Pose2D(0.,0.)),2:RobotState(2,Pose2D(3.,1.5)),
            3:RobotState(3,Pose2D(-6.5,0.))},opponents={4:RobotState(4,Pose2D(4.,0.))})
        self.motion=BallMotion()

    def tick(self,ctx):
        for p in self.players:p.context=ctx
        SoccerSimAgent.play(ctx,self.players,self.store)

    def plan(self,ctx):
        for p in self.players:p.context=ctx
        self.motion.position=(ctx.ball.x,ctx.ball.y)
        return self.store.tactics.plan(ctx,self.players,self.motion)

    def lose_ball(self):
        self.plan(self.ctx)
        self.assertEqual(self.store.tactics.possession,'ours')
        lost=replace(self.ctx,now=.1,ball=BallState(1.1,0.),
                     opponents={4:RobotState(4,Pose2D(1.2,0.))})
        return lost,self.plan(lost)

    def test_kickoff_release_contests_close_ball_and_preserves_rule_origin(self):
        ctx=replace(self.ctx,game=replace(self.ctx.game,kicking_team=1,secondary_time=10),teammates={
            1:RobotState(1,Pose2D(-.4,0.)),2:RobotState(2,Pose2D(-.8,2.3)),
            3:RobotState(3,Pose2D(-6.,0.))})
        self.tick(ctx)
        ready=next(p for p in self.players if p.action=='restart:receive_ready')
        self.assertLess(math.dist((ready.pose.x,ready.pose.y),ready._nav_goal),1.)
        self.assertLess(ready._nav_goal[0],0.)
        self.assertGreater(math.hypot(*ready._nav_goal),ctx.field.circle_radius)
        self.assertFalse(ready.rule_can_kick)
        # 球已移动但官方仍保留开球限制，不能靠位移自行放行。
        held=replace(ctx,now=.1,ball=BallState(.2,.2))
        self.tick(held)
        self.assertFalse(any(p.action=='kickoff:follow' for p in self.players))
        self.assertEqual(sum(p.rule_can_kick for p in self.players),1)
        for p in self.players:p._backend.calls.clear()
        released=replace(ctx,now=.3,ball=BallState(.8,1.5),
                         game=replace(ctx.game,kicking_team=255,secondary_time=0))
        self.tick(released)
        followers=[p for p in self.players if p.action=='kickoff:contest']
        self.assertEqual(len(followers),1)
        follower=followers[0]
        self.assertEqual(follower.id,self.store.kickoff_taker)
        self.assertTrue(follower.is_kicking)
        self.assertTrue(any(c[0]=='kick' for c in follower._backend.calls))
        self.assertEqual(follower.rule_shot_origin,(0.,0.))

    def test_loss_keeps_pressing_before_and_after_old_counterpress_deadline(self):
        lost,intents=self.lose_ball()
        for now in (lost.now,lost.now+1.):
            intents=self.plan(replace(lost,now=now))
            self.assertEqual(sum(a.role in ('press:tackle','press:approach') for a in intents.values()),1)
            self.assertEqual(sum(a.role=='press:outlet' for a in intents.values()),1)
            self.assertFalse(any(a.role.startswith('transition:') for a in intents.values()))

    def test_lost_then_loose_then_theirs_does_not_restart_counterpress_window(self):
        lost,_=self.lose_ball()
        deadline=self.store.tactics.counterpress_until
        loose=replace(lost,now=deadline+.05,ball=BallState(1.8,0.),
                      opponents={4:RobotState(4,Pose2D(5.,0.))})
        self.plan(loose)
        self.assertEqual(self.store.tactics.possession,'loose')
        intents=self.plan(replace(lost,now=deadline+.1))
        self.assertTrue(self.store.tactics.transition_active)
        self.assertLessEqual(self.store.tactics.counterpress_until,deadline)
        self.assertFalse(any(a.role=='transition:counterpress' for a in intents.values()))

    def test_bypassed_player_immediately_falls_back(self):
        own=replace(self.ctx,ball=BallState(1.,0.),teammates={
            **self.ctx.teammates,1:RobotState(1,Pose2D(1.,0.))})
        self.plan(own)
        lost=replace(own,now=.1,ball=BallState(-.4,0.),
                     opponents={4:RobotState(4,Pose2D(-.3,0.))})
        intents=self.plan(lost)
        self.assertTrue(self.store.tactics.transition_active)
        self.assertFalse(any(a.role=='transition:counterpress' for a in intents.values()))

    def test_regain_and_phase_reset_clear_transition(self):
        lost,_=self.lose_ball()
        regained=replace(self.ctx,now=lost.now+.2)
        intents=self.plan(regained)
        self.assertFalse(self.store.tactics.transition_active)
        self.assertFalse(any(a.role.startswith('transition:') for a in intents.values()))
        self.plan(replace(lost,now=1.))
        self.store.tactics.start_kickoff_followup(lost)
        self.store.tactics.reset()
        self.assertFalse(self.store.tactics.transition_active)
        self.assertEqual(self.store.tactics.kickoff_follow_until,0.)

    def test_goal_line_clearance_overrides_expired_counterpress(self):
        lost,_=self.lose_ball()
        deadline=self.store.tactics.counterpress_until
        emergency=replace(lost,now=deadline+.1,ball=BallState(-6.8,0.),
                          opponents={4:RobotState(4,Pose2D(-6.8,.2))})
        intents=self.plan(emergency)
        self.assertTrue(any(a.role=='goal:clear' for a in intents.values()))

    def test_advanced_keeper_also_turns_and_recovers(self):
        lost,_=self.lose_ball()
        advanced=replace(lost,now=self.store.tactics.counterpress_until,
                         teammates={**lost.teammates,3:RobotState(3,Pose2D(-2.,0.))})
        intents=self.plan(advanced)
        self.assertEqual(intents[3].role,'cover:goal')
        self.assertLess(intents[3].target[0],-5.)

    def test_goalside_defender_approaches_after_counterpress_without_distance_gate(self):
        ctx = replace(self.ctx,now=2.,ball=BallState(0.,0.),teammates={
            1:RobotState(1,Pose2D(-4.,0.)),2:RobotState(2,Pose2D(2.,2.)),
            3:RobotState(3,Pose2D(-6.5,0.))},opponents={
            4:RobotState(4,Pose2D(.1,0.))})
        self.store.tactics.transition_active=True
        self.store.tactics.counterpress_until=0.
        intents=self.plan(ctx)
        self.assertTrue(any(a.role=='press:approach' for a in intents.values()))
        self.assertTrue(next(a for a in intents.values() if a.role=='press:approach').receive)
        self.assertGreater(intents[1].target[0],ctx.teammates[1].pose.x)
        self.assertEqual(sum(a.role=='press:outlet' for a in intents.values()),1)

    def test_recovery_turns_towards_route_then_uses_forward_speed(self):
        p=self.players[0]
        p.context=replace(self.ctx,ball=BallState(2.,0.),teammates={1:RobotState(1,Pose2D())},opponents={})
        p._kicking=True
        p.recover_position((-4.,0.))
        self.assertFalse(p.is_kicking)
        self.assertEqual(p._backend.calls[-1][1:3],(0.,0.))
        self.assertNotEqual(p._backend.calls[-1][3],0.)
        p.reset_navigation()
        for i in range(25):
            p.context=replace(p.context,now=i/30,teammates={1:RobotState(1,Pose2D(0.,0.,math.pi))})
            p.recover_position((-4.,0.))
        self.assertGreater(p._backend.calls[-1][1],WALK_BACK_LIMIT_MPS)
        self.assertAlmostEqual(p._backend.calls[-1][3],0.)


if __name__=='__main__':
    unittest.main()
