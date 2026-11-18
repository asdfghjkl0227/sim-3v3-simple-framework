"""离线策略测试：假控制后端，不启动 SDK、ROS 或仿真服务。"""
import math
import sys
import types
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from src.framework.types import (
    Context, GameControlState, GameState, SetPlay, BallState, RobotState,
    Pose2D, Penalty, PlayerState, TeamState, ADULT_FIELD_DIMENSIONS,
)
from src.player import Player
from src.param import *
from src.utils.obstacles import Obstacle
from src.utils.rules import legal_target, safe_velocity, segment_clear, safe_push_direction
from src.utils.path_planner import plan_global_path
from src.utils.ball_motion import BallMotion

# 只替换入口基类依赖；测试真实 play/Player，绝不实例化平台 Agent。
platform = types.ModuleType('booster_agent_framework')
platform.AgentBase = type('AgentBase', (), {})
mixin = types.ModuleType('src.framework.agent')
mixin.SoccerAgentMixin = type('SoccerAgentMixin', (), {})
with patch.dict(sys.modules, {'booster_agent_framework': platform, 'src.framework.agent': mixin}):
    from src.main import SoccerSimAgent, get_phase, Phase


class Backend:
    mode = 'walk'
    fall_down_state = 'normal'
    fall_down_recoverable = True
    def __init__(self): self.calls = []
    def set_velocity(self,*args): self.calls.append(('velocity',*args))
    def kick(self,*args): self.calls.append(('kick',*args))
    def release_kick(self): self.calls.append(('release',))
    def request_mode(self,*args): self.calls.append(('mode',*args))
    def get_up(self): self.calls.append(('get_up',))


def context(state=GameState.PLAYING, **game):
    return Context(0.,1/30,1,ADULT_FIELD_DIMENSIONS,
                   GameControlState(state=state,**game),BallState(),
                   {i:RobotState(i,Pose2D(-i*2.,i-2.,0.)) for i in (1,2,3)})


class ComplianceTests(unittest.TestCase):
    def setUp(self):
        self.store = SimpleNamespace()
        SoccerSimAgent.init_store(None,self.store)
        self.players = [Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]
    def tick(self,ctx):
        for p in self.players: p.context=ctx
        SoccerSimAgent.play(ctx,self.players,self.store)
    def kicks(self):
        return [c for p in self.players for c in p._backend.calls if c[0]=='kick']
    def assertStopped(self):
        for p in self.players:
            self.assertFalse(p.is_kicking)
            self.assertEqual(p._backend.calls[-1],('velocity',0.,0.,0.))
            self.assertFalse(any(c[0] in ('kick','get_up','mode') for c in p._backend.calls))

    def test_set_stops_fallen_and_persistent_kicks(self):
        for p in self.players: p._kicking=True; p._backend.fall_down_state='fallen'
        self.tick(context(GameState.SET)); self.assertStopped()
    def test_stopped_overrides_ready(self):
        ctx=context(GameState.READY,stopped=True)
        self.assertEqual(get_phase(ctx),Phase.STOPPED)
        self.tick(ctx); self.assertStopped()
    def test_initial_finished_and_missing_game(self):
        for ctx in (context(GameState.INITIAL),context(GameState.FINISHED),replace(context(),game=None)):
            self.setUp(); self.tick(ctx); self.assertStopped()
    def test_penalized_no_recovery_or_local_return(self):
        players=tuple(PlayerState(penalty=Penalty.SENT_OFF) for _ in range(20))
        ctx=context(teams=(TeamState(team_number=1,players=players),))
        self.tick(ctx); self.assertStopped()
        self.tick(replace(ctx,now=100.)); self.assertStopped()
    def test_fallen_cancels_kick_and_recovers_when_allowed(self):
        p=self.players[0];p._backend.fall_down_state='fallen';p._kicking=True
        self.tick(context())
        self.assertFalse(p.is_kicking)
        self.assertIn(('get_up',),p._backend.calls)
    def test_missing_pose_stops(self):
        self.tick(replace(context(),teammates={}))
        self.assertStopped()
    def test_missing_ball_stops_persistent_kicks(self):
        for p in self.players:p._kicking=True
        self.tick(replace(context(),ball=None));self.assertStopped()
    def test_unknown_restart_owner_stops(self):
        self.tick(context(set_play=SetPlay.THROW_IN));self.assertStopped()
    def test_opponent_restarts_never_kick(self):
        for kind in SetPlay:
            self.setUp()
            ctx=context(set_play=kind,kicking_team=2,secondary_time=45)
            self.tick(ctx)
            self.assertFalse(self.kicks())
            self.assertTrue(all(p.rule_discs for p in self.players))
            self.tick(replace(ctx,now=100.))  # 本地超时不放行
            self.assertFalse(self.kicks())
    def test_ready_own_half_and_outside_circle(self):
        for owner in (1,2,255):
            self.setUp();self.tick(context(GameState.READY,kicking_team=owner))
            for p in self.players:
                self.assertTrue(p.rule_own_half)
                target=legal_target(p.context,(9.,0.),p.rule_discs,True)
                self.assertLessEqual(target[0],-HALF_SAFE_MARGIN_M)
                self.assertGreater(math.hypot(*target),1.5)
    def test_only_one_own_restart_taker(self):
        self.tick(context(set_play=SetPlay.GOAL_KICK,kicking_team=1))
        self.assertEqual(sum(p.rule_can_kick for p in self.players),1)
    def test_kickoff_taker_can_finish_approach_inside_midline_margin(self):
        p=self.players[0]
        ctx=context(kicking_team=1,secondary_time=10)
        ctx.teammates[1]=RobotState(1,Pose2D(-0.2,-0.2,0.))
        self.tick(ctx)
        self.assertEqual(self.store.kickoff_taker,1)
        self.assertTrue(any(c[0]=='kick' for c in p._backend.calls))
        # 仍在本方半场，但已超过主罚余量时先退回，不能继续SDK踢球。
        p._backend.calls=[]
        p.context=replace(ctx,teammates={1:RobotState(1,Pose2D(-0.05,-0.2,0.))})
        p.kick(0.,5.)
        self.assertFalse(any(c[0]=='kick' for c in p._backend.calls))
        self.assertTrue(any(c[0]=='velocity' and c[1]<0 for c in p._backend.calls))
    def test_sdk_outside_touchline_margin_returns_to_field(self):
        p=self.players[0]
        p.context=replace(context(set_play=SetPlay.THROW_IN,kicking_team=1),
            ball=BallState(-1.,4.5),teammates={1:RobotState(1,Pose2D(-1.,5.5,0.))})
        p.rule_can_kick=True
        p.kick(0.,5.)
        self.assertFalse(any(c[0]=='kick' for c in p._backend.calls))
        self.assertTrue(any(c[0]=='velocity' and c[2]<0 for c in p._backend.calls))
    def test_restart_taker_replaced_while_repositioning(self):
        ctx=context(set_play=SetPlay.THROW_IN,kicking_team=1)
        self.tick(ctx)
        original=self.store.kickoff_taker
        with patch.object(self.players[original-1],'recover_inactivity',return_value=True):
            self.tick(replace(ctx,now=1.))
        self.assertNotEqual(self.store.kickoff_taker,original)
        self.assertEqual(sum(p.rule_can_kick for p in self.players),1)
        replacement=self.players[self.store.kickoff_taker-1]
        self.assertTrue(replacement.is_kicking)
    def test_all_restarts_push_and_restriction_persists_after_release(self):
        for kind in SetPlay:
            self.setUp();ctx=context(set_play=kind,kicking_team=1,secondary_time=10)
            self.tick(ctx)
            self.tick(context())
            self.assertEqual(self.store.shot_origin,(0.,0.))
            p=self.players[0];p.context=context();p.rule_can_kick=True;p.rule_discs=[]
            p.rule_own_half=False;p._backend.calls=[];p.kick(0.,9.)
            kick=[c for c in p._backend.calls if c[0]=='kick'][-1]
            self.assertEqual(kick[2],RESTART_PUSH_POWER)
            self.assertFalse(p.kick_can_score(kick[1]+p.pose.theta))
    def test_shot_allowed_only_beyond_margin(self):
        p=self.players[0];p.context=replace(context(),ball=BallState(1.,0.))
        p.rule_can_kick=True;p.rule_shot_origin=(0.,0.)
        p.kick(0.,5.)
        self.assertEqual([c for c in p._backend.calls if c[0]=='kick'][-1][2],5.)
    def test_new_restart_replaces_origin(self):
        self.tick(context(kicking_team=1,secondary_time=10))
        self.tick(context(GameState.SET))
        self.tick(replace(context(kicking_team=1,secondary_time=10),ball=BallState(-1.,2.)))
        self.assertEqual(self.store.shot_origin,(-1.,2.))
    def test_direct_motion_exports_fail_closed(self):
        p=self.players[0];p.context=context(GameState.SET);p.rule_can_kick=True
        p.set_velocity(2.,1.,1.);p.kick();p.get_up();p.request_mode('walk')
        self.assertFalse(any(c[0] in ('kick','get_up','mode') for c in p._backend.calls))
        self.assertEqual(p._backend.calls[-1],('velocity',0.,0.,0.))
    def test_wait_target_and_path_outside_three_metres(self):
        self.tick(context(set_play=SetPlay.THROW_IN,kicking_team=2))
        for p in self.players:
            self.assertTrue(any(c.radius==WAIT_BALL_DISTANCE_M for c in p.rule_discs))
    def test_inactivity_triggers_real_retreat(self):
        p=self.players[0];p.context=context()
        self.assertFalse(p.recover_inactivity())
        p.context=replace(p.context,now=INACTIVITY_RECOVERY_S+0.1)
        self.assertTrue(p.recover_inactivity())
        self.assertFalse(p.is_kicking)
    def test_missing_ball_cancels_direct_kick(self):
        p=self.players[0];p.context=replace(context(),ball=None);p.rule_can_kick=True;p._kicking=True
        p.kick();self.assertFalse(p.is_kicking)
    def test_running_get_up_is_not_resubmitted_or_overwritten(self):
        p=self.players[0];p.context=context();p._backend.fall_down_state='fallen'
        self.assertFalse(p.ensure_ready())
        p._backend.fall_down_recoverable=False
        p._backend.fall_down_state=None;p._backend.mode=None  # 异步派发后缓存被清空
        p.context=replace(p.context,now=5.)
        self.assertFalse(p.ensure_ready())
        self.assertEqual(sum(c[0]=='get_up' for c in p._backend.calls),1)
        self.assertFalse(any(c[0]=='mode' for c in p._backend.calls))
        p._backend.mode='walk';p._backend.fall_down_state='normal'
        self.assertTrue(p.ensure_ready())
    def test_unrecoverable_fall_waits_for_official_recoverable(self):
        p=self.players[0];p.context=context();p._backend.fall_down_state='falling'
        p._backend.fall_down_recoverable=False;p.ensure_ready()
        self.assertFalse(any(c[0]=='get_up' for c in p._backend.calls))
        p._backend.fall_down_recoverable=True;p.ensure_ready()
        self.assertIn(('get_up',),p._backend.calls)
    def test_recovered_player_requests_walk_without_circular_wait(self):
        p=self.players[0];p.context=context()
        p._backend.fall_down_state='fallen';p.ensure_ready()
        p._backend.fall_down_state='normal';p._backend.mode='prepare'
        self.assertFalse(p.ensure_ready())
        self.assertFalse(p._recovering)
        self.assertIn(('mode','walk'),p._backend.calls)
        p._backend.mode='walk'
        self.assertTrue(p.ensure_ready())
    def test_getting_up_state_does_not_repeat_recovery_even_if_recoverable(self):
        p=self.players[0];p.context=context()
        p._backend.fall_down_state='getting_up';p._recovering=True
        self.assertFalse(p.ensure_ready())
        self.assertFalse(any(c[0] in ('get_up','mode') for c in p._backend.calls))
    def test_guard_escapes_near_goal_without_heading_into_net(self):
        p=self.players[0];pose=Pose2D(-6.5,0.)
        ctx=replace(context(),ball=BallState(-6.14,1.88),teammates={})
        for i in range(240):
            p.context=replace(ctx,now=i/30,teammates={1:RobotState(1,pose)})
            p.rule_discs=[]
            p.guard()
            v=[c for c in p._backend.calls if c[0]=='velocity'][-1]
            c,s=math.cos(pose.theta),math.sin(pose.theta)
            pose=Pose2D(pose.x+(v[1]*c-v[2]*s)/30,
                        pose.y+(v[1]*s+v[2]*c)/30,pose.theta+v[3]/30)
            self.assertGreaterEqual(pose.x,-6.6)
            self.assertLessEqual(abs(pose.y),4.1)
        self.assertGreater(math.hypot(pose.x-ctx.ball.x,pose.y-ctx.ball.y),3.0)

    def test_ready_returns_from_opponent_half_without_stopping_to_turn(self):
        poses={i:Pose2D(6.,y,0.) for i,y in ((1,-3.),(2,0.),(3,3.))}
        # 简化运动学检验回位路径，不替代 SDK 物理测试。
        for frame in range(450):
            ctx=replace(context(GameState.READY,kicking_team=2),now=frame/30,
                        teammates={i:RobotState(i,pose) for i,pose in poses.items()})
            self.tick(ctx)
            for p in self.players:
                pose=poses[p.id]
                v=[c for c in p._backend.calls if c[0]=='velocity'][-1]
                if frame==0:
                    self.assertLess(v[1],0.)  # 面朝前也立刻后退回半场
                c,s=math.cos(pose.theta),math.sin(pose.theta)
                poses[p.id]=Pose2D(pose.x+(v[1]*c-v[2]*s)/30,
                                  pose.y+(v[1]*s+v[2]*c)/30,pose.theta+v[3]/30)
        for pose in poses.values():
            self.assertLess(pose.x,-HALF_SAFE_MARGIN_M)
            self.assertGreater(math.hypot(pose.x,pose.y),1.5)

    def test_ready_transition_discards_attack_detour(self):
        self.tick(context())
        for p in self.players:
            p._nav_detour=(5.,0.);p._nav_detour_until=100.
        self.tick(replace(context(GameState.READY),now=1.))
        self.assertTrue(all(p._nav_detour is None for p in self.players))

    def test_returned_player_immediately_joins_attack_wing(self):
        penalties=list(PlayerState() for _ in range(20))
        penalties[2]=PlayerState(penalty=Penalty.ILLEGAL_POSITIONING)
        team=TeamState(team_number=1,players=tuple(penalties))
        positions={1:RobotState(1,Pose2D(3.8,0.)),2:RobotState(2,Pose2D(-2.,0.)),
                   3:RobotState(3,Pose2D(-1.,-3.))}
        ctx=replace(context(teams=(team,TeamState(team_number=2))),
                    ball=BallState(4.,0.),teammates=positions)
        self.tick(ctx)
        p=self.players[2]
        p._nav_detour=(-4.,0.);p._nav_detour_until=99.
        p._retreating=True
        released=replace(ctx,now=.1,game=GameControlState(state=GameState.PLAYING))
        self.tick(released)
        self.assertEqual(p.action,'triangle:wing')
        self.assertFalse(p._retreating)
        self.assertIsNone(p._nav_detour)
        self.assertGreater(p._nav_goal[0],0.)

    def test_goal_line_live_ball_triggers_immediate_clearance(self):
        positions={1:RobotState(1,Pose2D(-6.75,0.)),2:RobotState(2,Pose2D(-4.,1.)),
                   3:RobotState(3,Pose2D(-3.,-2.))}
        ctx=replace(context(),ball=BallState(-7.,0.),teammates=positions)
        p=self.players[0]
        p._retreating=True;p._activity_since=-10.;p._activity_anchor=(-6.75,0.)
        self.tick(ctx)
        self.assertEqual(self.store.cur_phase,Phase.NORMAL)
        self.assertEqual(p.action,'goal:clear')
        self.assertTrue(p.is_kicking)
        kick=[c for c in p._backend.calls if c[0]=='kick'][-1]
        self.assertGreater(math.cos(kick[1]+p.pose.theta),0.)

    def test_goal_line_rescue_never_overrides_official_stop_or_penalty(self):
        for ctx in (context(GameState.SET),context(stopped=True),
                    context(set_play=SetPlay.CORNER_KICK,kicking_team=2)):
            self.setUp()
            ctx=replace(ctx,ball=BallState(-7.,0.),
                        teammates={1:RobotState(1,Pose2D(-6.75,0.))})
            self.tick(ctx)
            self.assertFalse(self.kicks())

    def test_goal_line_failed_kick_repositions_and_teammate_takes_over(self):
        positions={1:RobotState(1,Pose2D(-6.75,0.)),2:RobotState(2,Pose2D(-5.8,1.)),
                   3:RobotState(3,Pose2D(-4.,-2.))}
        ctx=replace(context(),ball=BallState(-7.,0.),teammates=positions)
        self.tick(ctx)
        self.tick(replace(ctx,now=.1))
        positions={**positions,2:RobotState(2,Pose2D(-5.8,.6))}
        self.tick(replace(ctx,now=INACTIVITY_RECOVERY_S+.2,teammates=positions))
        p=self.players[0]
        self.assertEqual(p.action,'goal:reposition')
        self.assertFalse(p.is_kicking)
        self.assertIsNotNone(p._rescue_target)
        self.assertLess(math.dist((p.pose.x,p.pose.y),p._rescue_target),1.)
        self.assertTrue(any(other.action=='goal:clear' for other in self.players[1:]))
        target=p._rescue_target
        positions={**positions,1:RobotState(1,Pose2D(*target))}
        self.tick(replace(ctx,now=INACTIVITY_RECOVERY_S+.8,teammates=positions))
        self.assertFalse(p._retreating)
        self.assertIsNone(p._rescue_target)

    def test_goal_line_reposition_is_cancelled_by_set(self):
        ctx=replace(context(),ball=BallState(-7.,0.))
        self.tick(ctx)
        self.tick(replace(ctx,now=INACTIVITY_RECOVERY_S+.1))
        for p in self.players:p._backend.calls.clear()
        self.tick(replace(ctx,game=GameControlState(state=GameState.SET)))
        self.assertStopped()

    def test_ready_does_not_treat_ball_near_goal_as_active_play(self):
        self.tick(replace(context(GameState.READY),ball=BallState(-7.,0.)))
        self.assertEqual(self.store.cur_phase,Phase.READY)
        self.assertFalse(self.kicks())

    def test_replanning_detour_preserves_restart_exclusion(self):
        p=self.players[0];p.context=context();p.rule_discs=[Obstacle(0.,0.,1.9)]
        p._navigation_target((-4.,0.),p.rule_discs)
        p.context=replace(p.context,now=NAV_STALL_S+.1)
        target=p._navigation_target((-4.,0.),p.rule_discs)
        self.assertGreaterEqual(math.hypot(*target),1.9)
        self.assertTrue(segment_clear((p.pose.x,p.pose.y),target,p.rule_discs))

    def test_goal_line_emergency_still_obeys_stop_penalty_and_opponent_restart(self):
        for ctx in (context(GameState.SET),context(stopped=True),
                    context(kicking_team=2,set_play=SetPlay.CORNER_KICK)):
            self.setUp()
            self.tick(replace(ctx,ball=BallState(-7.,0.)))
            self.assertFalse(self.kicks())

    def test_stalled_navigation_changes_route_and_expires(self):
        p=self.players[0];p.context=context()
        p.walk_to((3.,2.),avoid_robots=True)
        p.context=replace(p.context,now=NAV_STALL_S+.1)
        p.walk_to((3.,2.),avoid_robots=True)
        self.assertIsNotNone(p._nav_detour)
        self.assertGreater(p.routing_blocked_until,p.context.now)
        self.assertNotEqual(p._nav_detour,(3.,2.))

    def test_ball_on_goal_line_never_changes_to_ready_without_official_state(self):
        for x in (-6.95,-7.,-7.02):
            ctx=replace(context(),ball=BallState(x,0.))
            self.assertEqual(get_phase(ctx),Phase.NORMAL)

    def test_tactical_plan_retains_restart_shot_gate(self):
        self.tick(context(kicking_team=1,set_play=SetPlay.THROW_IN))
        self.tick(replace(context(),now=.1))
        self.assertEqual(self.store.shot_origin,(0.,0.))
        for c in self.kicks():
            self.assertEqual(c[2],RESTART_PUSH_POWER)


class GeometryTests(unittest.TestCase):
    def test_opponent_half_circle_escape_has_tangent_route(self):
        ctx=context(GameState.READY)
        pose=Pose2D(1.48,.15)
        v=safe_velocity(ctx,pose,-1.,1.,[Obstacle(0.,0.,1.8)],True)
        self.assertLessEqual(v[0],0.)
        self.assertGreater(math.hypot(*v),.3)
        self.assertGreater(math.hypot(pose.x+v[0]*RULE_LOOKAHEAD_S,
                                     pose.y+v[1]*RULE_LOOKAHEAD_S),math.hypot(pose.x,pose.y))
    def test_boundary_projection_does_not_reenter_restart_circle(self):
        ctx=context();discs=[Obstacle(6.8,4.4,1.9)]
        t=legal_target(ctx,(8.,5.),discs)
        self.assertIsNotNone(t)
        self.assertLessEqual(t[0],6.6);self.assertLessEqual(t[1],4.1)
        self.assertGreaterEqual(math.hypot(t[0]-6.8,t[1]-4.4),1.9)
    def test_velocity_cannot_cross_circle_or_boundary_or_half(self):
        ctx=context()
        self.assertEqual(safe_velocity(ctx,Pose2D(-2.,0.),1.2,0.,[Obstacle(0.,0.,1.9)]),(0.,0.))
        self.assertEqual(safe_velocity(ctx,Pose2D(6.5,0.),1.,0.),(0.,0.))
        self.assertEqual(safe_velocity(ctx,Pose2D(-.4,0.),1.,0.,own_half=True),(0.,0.))
    def test_inside_circle_retreat_has_positive_radial_speed(self):
        ctx=context();v=safe_velocity(ctx,Pose2D(-1.,0.),1.,0.,[Obstacle(0.,0.,1.9)])
        self.assertLess(v[0],-.29)
    def test_safe_push_avoids_both_goals_including_corners(self):
        for x,y in ((0.,0.),(-6.5,0.),(6.8,4.5),(-6.8,-4.5)):
            ctx=replace(context(),ball=BallState(x,y));a=safe_push_direction(ctx)
            self.assertIsNotNone(a)
            dx,dy=math.cos(a),math.sin(a)
            for gx in (-7.,7.):
                if abs(dx)>1e-6 and (gx-x)/dx>=0:
                    self.assertGreater(abs(y+(gx-x)/dx*dy),1.3+RESTART_GOAL_MARGIN_M)
    def test_astar_no_diagonal_cut_and_no_blocked_goal(self):
        ctx=context();discs=[Obstacle(0.,0.,1.9)]
        self.assertIsNone(plan_global_path(ctx,(-3.,0.),(0.,0.),discs))
        path=plan_global_path(ctx,(-3.,0.),(3.,0.),discs)
        self.assertIsNotNone(path)
        self.assertTrue(all(segment_clear(a,b,discs,GLOBAL_OBSTACLE_MARGIN_M) for a,b in zip(path,path[1:])))

    def test_half_field_path_never_returns_to_opponent_side(self):
        ctx=context();discs=[Obstacle(-2.,0.,.8)]
        path=plan_global_path(ctx,(-.7,-2.),(-3.,2.),discs,own_half=True)
        self.assertIsNotNone(path)
        self.assertTrue(all(x <= -HALF_SAFE_MARGIN_M for x,y in path))


class BallMotionTests(unittest.TestCase):
    def test_motion_predicts_ahead_and_resets_on_official_ready(self):
        model=BallMotion()
        for i in range(15):
            ctx=replace(context(),now=i/30,ball=BallState(2*i/30,0.),teammates={})
            model.update(ctx)
        self.assertGreater(model.velocity[0],1.5)
        self.assertGreater(model.predict(ctx,.5)[0],ctx.ball.x)
        model.update(replace(ctx,game=GameControlState(state=GameState.READY)))
        self.assertIsNone(model.position)

    def test_teleport_not_fitted_as_ball_roll(self):
        model=BallMotion();model.update(context())
        model.update(replace(context(),now=.1,ball=BallState(6.,0.)))
        self.assertEqual(model.roll_samples,0)
        self.assertEqual(model.velocity,(0.,0.))


if __name__=='__main__': unittest.main()
