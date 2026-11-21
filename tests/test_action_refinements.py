"""倒地恢复、动作冲突、边线与传接球的跨帧回归；不连接SDK。"""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D, GameState, SetPlay
from src.player import Player
from src.tactics import TeamTactics
from src.evaluation import MatchEvaluation
from src.utils.ball_motion import BallMotion
from src.utils.motion_control import WalkCommand
from src.utils.rules import safe_push_direction
from src.param import *


class ActionRefinements(unittest.TestCase):
    def make_player(self, pose=Pose2D(-1.,0.)):
        p=Player(1,SimpleNamespace(team_id=1),Backend())
        p.context=replace(context(),teammates={1:RobotState(1,pose)})
        p.rule_can_kick=True
        return p

    def test_world_acceleration_reverse_and_immediate_official_stop(self):
        p=self.make_player()
        p.set_velocity(2.,0.,0.)
        first=p._backend.calls[-1][1]
        self.assertLessEqual(first,WALK_ACCEL_MPS2*p.context.dt)
        for i in range(1,21):
            p.context=replace(p.context,now=i/30)
            p.set_velocity(2.,0.,0.)
        before=p._backend.calls[-1][1]
        p.context=replace(p.context,now=21/30)
        p.set_velocity(-2.,0.,0.)
        after=p._backend.calls[-1][1]
        self.assertLessEqual(abs(after-before),WALK_ACCEL_MPS2/30+1e-9)
        p.context=replace(p.context,game=replace(p.context.game,state=GameState.SET))
        p.set_velocity(2.,1.,1.)
        self.assertEqual(p._backend.calls[-1],('velocity',0.,0.,0.))

    def test_sideways_turn_and_post_recovery_speed_envelopes(self):
        command=WalkCommand()
        pose=Pose2D()
        for i in range(100):
            vx,vy,yaw=command.shape(pose,i/30,1/30,2.,2.,2.)
            command.record(pose,i/30,vx,vy,yaw)
            self.assertLessEqual(math.hypot(vx/MAX_LINEAR,vy/WALK_SIDE_LIMIT_MPS),1.+1e-9)
            self.assertLessEqual(abs(yaw),WALK_TURN_LIMIT_RADPS+1e-9)
        p=self.make_player()
        p._backend.fall_down_state='fallen'
        p.ensure_ready()
        p._backend.fall_down_state='normal'
        self.assertTrue(p.ensure_ready())
        p.kick(0.,SHOT_POWER)
        self.assertFalse(p.is_kicking)
        p.set_velocity(2.,0.,0.)
        self.assertLessEqual(math.hypot(*p._backend.calls[-1][1:3]),RECOVERY_WALK_LIMIT_MPS)
        p.context=replace(p.context,now=RECOVERY_KICK_SETTLE_S+.01)
        p.kick(0.,SHOT_POWER)
        self.assertTrue(p.is_kicking)

    def test_observed_getting_up_never_repeats_or_requests_walk(self):
        p=self.make_player()
        p._backend.fall_down_state='getting_up'
        for t in (0.,1.,5.,12.):
            p.context=replace(p.context,now=t)
            self.assertFalse(p.ensure_ready())
        self.assertTrue(p._recovering)
        self.assertFalse(any(c[0] in ('get_up','mode') for c in p._backend.calls))
        p._backend.fall_down_state=None
        p._backend.mode=None
        p.context=replace(p.context,now=12.5)
        self.assertFalse(p.ensure_ready())
        p._backend.fall_down_state='fallen'
        p.context=replace(p.context,now=13.)
        p.ensure_ready()
        self.assertEqual(sum(c[0]=='get_up' for c in p._backend.calls),1)

    def test_unknown_state_does_not_start_walk_and_requests_are_throttled(self):
        p=self.make_player()
        p._backend.mode=None
        p._backend.fall_down_state=None
        self.assertFalse(p.ensure_ready())
        self.assertFalse(any(c[0]=='mode' for c in p._backend.calls))
        p._backend.fall_down_state='normal'
        for t in (0.,.01,.1,.2):
            p.context=replace(p.context,now=t)
            p.ensure_ready()
        self.assertEqual(sum(c[0]=='mode' for c in p._backend.calls),1)

    def test_boundary_pass_has_infield_receiver_and_locked_passer(self):
        ctx=replace(context(),ball=BallState(0.,4.45),teammates={
            1:RobotState(1,Pose2D(-.3,4.5)),2:RobotState(2,Pose2D(2.5,2.0)),
            3:RobotState(3,Pose2D(-6.5,0.))})
        players=[Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        for p in players:p.context=ctx
        motion=BallMotion();motion.position=(0.,4.45)
        tactics=TeamTactics()
        intents=tactics.plan(ctx,players,motion)
        self.assertEqual(intents[1].role,'boundary:pass')
        self.assertEqual(intents[2].role,'receive:ready')
        self.assertEqual(tactics.pass_plan['receiver'],2)
        self.assertLess(intents[1].kick_target[1],ctx.ball.y-BOUNDARY_RECEIVE_INSET_M)
        # 即使另一个人当前拦截成本更低，未离脚前不能改变主罚者。
        with unittest.mock.patch.object(motion,'intercept',side_effect=[((0.,4.45),10.),((0.,4.45),0.)]):
            selected,_=tactics._choose_attacker(ctx,players[:2],motion)
        self.assertEqual(selected.id,1)

    def test_boundary_fallback_is_short_inward_and_restart_angle_has_margin(self):
        p=self.make_player(Pose2D(0.,4.7))
        p.context=replace(p.context,ball=BallState(0.,4.45))
        motion=BallMotion();motion.position=(0.,4.45)
        intent=TeamTactics().plan(p.context,[p],motion)[1]
        self.assertEqual(intent.role,'boundary:save')
        self.assertGreater(intent.kick_target[1],0.)
        self.assertLess(intent.kick_target[1],p.context.ball.y-1.)
        for sign in (-1,1):
            ctx=replace(p.context,ball=BallState(0.,sign*4.45),
                        game=replace(p.context.game,set_play=SetPlay.THROW_IN,kicking_team=1))
            a=safe_push_direction(ctx)
            self.assertIsNotNone(a)
            self.assertLessEqual(sign*math.sin(a),-RESTART_INWARD_MIN_SIN+1e-9)

    def test_pending_pass_aborts_blocked_lane_and_launched_pass_aborts_overshoot(self):
        p=self.make_player()
        tactics=TeamTactics()
        plan=dict(passer=1,receiver=2,target=(2.,0.),power=1.5,
                  origin=(0.,0.),launched=False,until=10.)
        receiver=Player(2,SimpleNamespace(team_id=1),Backend())
        ctx=replace(p.context,ball=BallState(0.,0.),opponents={4:RobotState(4,Pose2D(1.,0.))})
        tactics.pass_plan=dict(plan)
        tactics._update_pass(ctx,[p,receiver],SimpleNamespace(speed=0.))
        self.assertIsNone(tactics.pass_plan)
        tactics.pass_plan={**plan,'launched':True}
        ctx=replace(ctx,ball=BallState(3.1,0.),opponents={})
        tactics._update_pass(ctx,[p,receiver],SimpleNamespace(speed=1.))
        self.assertIsNone(tactics.pass_plan)

    def test_receive_does_not_kick_ball_away_before_control_is_confirmed(self):
        p=self.make_player(Pose2D(0.,0.))
        p.context=replace(p.context,ball=BallState(.3,0.))
        p.receive_ball((.55,0.))
        self.assertFalse(any(c[0]=='kick' for c in p._backend.calls))
        p._backend.calls.clear()
        p.receive_ball((1.,1.),incoming=False)
        self.assertFalse(p.ball_access)
        self.assertFalse(any(c[0]=='kick' for c in p._backend.calls))

    def test_pass_receiver_runs_downstream_of_fast_ball(self):
        p=self.make_player(Pose2D(2.,0.))
        motion=BallMotion();motion.position=(0.,0.);motion.velocity=(2.,0.)
        raw,_=motion.intercept(p.context,p.pose)
        target=TeamTactics._receive_target(p.context,p,motion)
        self.assertGreater(target[0],raw[0])

    def test_receive_confirmation_releases_plan_and_resumes_ball_action(self):
        ctx=replace(context(),ball=BallState(0.,0.),teammates={
            1:RobotState(1,Pose2D(-.3,0.)),2:RobotState(2,Pose2D(3.,0.)),
            3:RobotState(3,Pose2D(-6.5,0.))})
        players=[Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        tactics=TeamTactics();tactics.evaluation=MatchEvaluation()
        motion=BallMotion();motion.position=(0.,0.)
        for p in players:p.context=ctx
        tactics._start_pass(ctx,players[0],(1.,2,(3.,0.),1.5))
        ctx=replace(ctx,now=.5,ball=BallState(1.,0.))
        motion.position=(1.,0.);motion.velocity=(1.5,0.)
        for p in players:p.context=ctx
        intents=tactics.plan(ctx,players,motion)
        self.assertTrue(tactics.pass_plan['launched'])
        self.assertEqual([i for i,a in intents.items() if a.receive],[2])
        for t in (1.,1.+CONTROL_HOLD_S+.01):
            ctx=replace(ctx,now=t,ball=BallState(2.8,0.))
            motion.position=(2.8,0.);motion.velocity=(0.,0.)
            for p in players:p.context=ctx
            tactics.evaluation.update(ctx)
            intents=tactics.plan(ctx,players,motion)
        self.assertIsNone(tactics.pass_plan)
        self.assertEqual(tactics.evaluation.owner,('ours',2))
        self.assertEqual(sum(e['event']=='receive' for e in tactics.evaluation.events),1)
        self.assertFalse(any(a.receive for a in intents.values()))
        self.assertIsNotNone(intents[2].kick_target)

    def test_intercepted_pass_releases_receiver_to_contest(self):
        ctx=replace(context(),now=1.,ball=BallState(1.,0.),teammates={
            1:RobotState(1,Pose2D(-2.,0.)),2:RobotState(2,Pose2D(3.,0.)),
            3:RobotState(3,Pose2D(-6.5,0.))},opponents={1:RobotState(1,Pose2D(1.,.1))})
        players=[Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
        for p in players:p.context=ctx
        motion=BallMotion();motion.position=(1.,0.)
        tactics=TeamTactics()
        tactics.pass_plan=dict(passer=1,receiver=2,origin=(0.,0.),target=(3.,0.),
                               power=1.5,launched=True,until=10.)
        intents=tactics.plan(ctx,players,motion)
        self.assertIsNone(tactics.pass_plan)
        self.assertFalse(any(a.receive for a in intents.values()))
        self.assertTrue(any(a.role.startswith('press:') for a in intents.values()))

    def test_same_number_opponent_still_limits_close_contact_speed(self):
        p=self.make_player()
        for i in range(30):
            p.context=replace(p.context,now=i/30)
            p.set_velocity(2.,0.,0.)
        p.context=replace(p.context,now=1.,opponents={1:RobotState(1,Pose2D(-.5,0.))})
        p.set_velocity(2.,0.,0.)
        self.assertLessEqual(math.hypot(*p._backend.calls[-1][1:3]),WALK_NEAR_ROBOT_LIMIT_MPS)


if __name__=='__main__':
    unittest.main()
