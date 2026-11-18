"""针对比赛故障的场景测试；只用假后端，不运行 SDK Agent。"""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_compliance import context, Backend, SoccerSimAgent
from src.framework.types import BallState, RobotState, Pose2D, TeamState, PlayerState, Penalty, GameState
from src.player import Player
from src.tactics import TeamTactics
from src.evaluation import MatchEvaluation
from src.utils.ball_motion import BallMotion
from src.utils.rules import bounds, legal_target
from src.param import *
from unittest.mock import patch


class TacticalScenarios(unittest.TestCase):
    def setUp(self):
        self.store=SimpleNamespace()
        SoccerSimAgent.init_store(None,self.store)
        self.players=[Player(i,SimpleNamespace(team_id=1),Backend()) for i in (1,2,3)]

    def tick(self,ctx):
        for p in self.players:p.context=ctx
        SoccerSimAgent.play(ctx,self.players,self.store)

    def test_score_modes_never_park_for_one_goal_lead_or_late_clock(self):
        for margin,mode in ((-2,'chase'),(0,'balanced'),(1,'balanced'),(2,'protect_lead'),(3,'protect_lead')):
            with self.subTest(margin=margin):
                self.setUp()
                ctx=replace(context(teams=(TeamState(team_number=1,score=3+margin),
                                          TeamState(team_number=2,score=3)),secs_remaining=1),ball=BallState(3.,0.))
                self.tick(ctx)
                self.assertEqual(self.store.tactics.mode,mode)

    def test_attack_support_advances_while_keeper_covers_when_losing(self):
        ctx=replace(context(teams=(TeamState(team_number=1),TeamState(team_number=2,score=1))),
                    ball=BallState(4.,0.))
        self.tick(ctx)
        wing=next(p for p in self.players if p.action=='triangle:wing')
        self.assertGreater(wing._nav_goal[0],0.)
        keeper=next(p for p in self.players if p.action=='cover:goal')
        self.assertLess(keeper._nav_goal[0],-5.)

    def test_ready_all_players_move_towards_midfield_from_deep_positions(self):
        self.tick(context(GameState.READY))
        self.assertTrue(all(t[0]>=READY_BASE_X_M for t in self.store.ready_targets.values()))
        self.assertEqual(sum(t[0]==READY_PRESS_X_M for t in self.store.ready_targets.values()),2)

    def test_goal_cover_intercepts_without_avoiding_incoming_ball(self):
        p=self.players[0]
        p.context=replace(context(),ball=BallState(-3.,0.),
                          teammates={1:RobotState(1,Pose2D(-6.,0.))})
        with patch.object(p,'walk_to') as walk:
            p.defend_goal((-6.15,0.))
            self.assertFalse(walk.call_args.kwargs['avoid_ball'])
        p.context=replace(p.context,ball=BallState(-5.,0.))
        p.rule_can_kick=True
        p.defend_goal((-6.15,0.))
        self.assertTrue(p.is_kicking)

    def test_corner_taker_can_finish_sdk_step_beyond_end_line(self):
        from src.framework.types import SetPlay
        ctx=replace(context(set_play=SetPlay.CORNER_KICK,kicking_team=1),
            ball=BallState(7.,4.45),teammates={1:RobotState(1,Pose2D(7.2,4.65))})
        self.tick(ctx)
        self.assertTrue(self.players[0].is_kicking)
        self.tick(replace(ctx,game=replace(ctx.game,stopped=True)))
        self.assertFalse(self.players[0].is_kicking)

    def test_fast_goal_danger_preserves_keeper_block_and_slow_ball_clears(self):
        ctx=replace(context(),ball=BallState(-5.8,0.),teammates={
            1:RobotState(1,Pose2D(-6.5,0.)),2:RobotState(2,Pose2D(-4.,1.)),
            3:RobotState(3,Pose2D(0.,-1.))})
        for p in self.players:
            p.context=ctx
            p.rule_can_kick=True
        motion=BallMotion()
        motion.position=(-5.8,0.)
        motion.velocity=(-3.,0.)
        intents=self.store.tactics.plan(ctx,self.players,motion)
        self.assertEqual(intents[1].role,'cover:goal')
        keeper=self.players[0]
        keeper.ball_motion=motion
        keeper.defend_goal(intents[1].target)
        self.assertFalse(keeper.is_kicking)
        motion.velocity=(0.,0.)
        intents=self.store.tactics.plan(ctx,self.players,motion)
        self.assertEqual(intents[1].role,'goal:clear')
        keeper.defend_goal((-6.5,0.))
        self.assertTrue(keeper.is_kicking)

    def test_touchline_restart_sdk_approach_is_not_cancelled_at_observed_pose(self):
        from src.framework.types import SetPlay
        ctx=replace(context(set_play=SetPlay.THROW_IN,kicking_team=1),
            ball=BallState(-1.12,4.45),teammates={1:RobotState(1,Pose2D(-1.,5.03))})
        self.tick(ctx)
        self.assertTrue(self.players[0].is_kicking)
        self.tick(replace(ctx,game=replace(ctx.game,state=GameState.READY)))
        self.assertFalse(self.players[0].is_kicking)

    def test_kickoff_support_does_not_retreat_to_own_goal(self):
        ctx=replace(context(kicking_team=1,secondary_time=10),teammates={
            1:RobotState(1,Pose2D(-.4,0.)),2:RobotState(2,Pose2D(-.8,2.3)),
            3:RobotState(3,Pose2D(-.8,-2.3))})
        self.tick(ctx)
        support=[p for p in self.players if p.id!=self.store.kickoff_taker]
        self.assertEqual(sum(p._nav_goal[0]>-2. for p in support),1)
        self.assertEqual(sum(p._nav_goal[0]<-5. for p in support),1)

    def test_keeper_does_not_abandon_goal_for_ball_two_metres_away(self):
        ctx=replace(context(),ball=BallState(-4.5,0.),teammates={
            1:RobotState(1,Pose2D(-6.5,0.)),2:RobotState(2,Pose2D(0.,1.)),
            3:RobotState(3,Pose2D(1.,-1.))})
        self.tick(ctx)
        self.assertEqual(self.players[0].action,'cover:goal')
        self.assertFalse(self.players[0].is_kicking)
        self.assertEqual(self.store.tactics.anchor_id,1)

    def test_three_unique_man_marks_with_ball_goal_blocking(self):
        ctx=replace(context(),ball=BallState(1.,0.),opponents={
            4:RobotState(4,Pose2D(1.,.2)),5:RobotState(5,Pose2D(2.,3.)),6:RobotState(6,Pose2D(-1.,-3.))})
        self.tick(ctx)
        mapping=self.store.tactics.mark_assignments
        self.assertEqual(len(set(mapping.values())),3)
        press=next(p for p in self.players if p.action=='press:goalside')
        self.assertEqual(mapping[press.id],4)
        self.assertLess(press._nav_goal[0],ctx.ball.x)
        self.assertAlmostEqual(press._nav_goal[1],0.)
        self.assertEqual(sum(p.action.startswith('mark:') for p in self.players),2)

    def test_sideline_ball_can_be_kicked_inward_from_outside(self):
        ctx=replace(context(),ball=BallState(0.,4.45),teammates={1:RobotState(1,Pose2D(0.,4.7))})
        self.tick(ctx)
        p=self.players[0]
        self.assertEqual(p.action,'boundary:save')
        kick=next(c for c in p._backend.calls if c[0]=='kick')
        self.assertLess(math.sin(kick[1]),0.)
        self.assertTrue(p.is_kicking)

    def test_excursion_only_for_active_ball_access_and_clears_on_ready(self):
        ctx=replace(context(),ball=BallState(0.,4.45))
        p=self.players[0];p.context=ctx
        p.walk_to((0.,4.7),ball_access=True)
        self.assertGreater(p._nav_goal[1],ctx.field.width/2)
        self.assertLessEqual(legal_target(ctx,(0.,4.7))[1],ctx.field.width/2-FIELD_SAFE_MARGIN_M)
        self.tick(replace(ctx,game=replace(ctx.game,state=GameState.READY)))
        self.assertFalse(p.ball_access)
        self.assertLessEqual(p._nav_goal[1],ctx.field.width/2-FIELD_SAFE_MARGIN_M)

    def test_launched_pass_has_exclusive_receiver_and_passer_releases_ball(self):
        ctx=replace(context(),ball=BallState(0.,0.),teammates={
            1:RobotState(1,Pose2D(-.2,0.)),2:RobotState(2,Pose2D(2.,1.)),3:RobotState(3,Pose2D(-2.,2.))})
        self.tick(ctx)
        self.store.tactics.pass_plan=dict(passer=1,receiver=2,target=(2.,1.),power=1.,
            origin=(0.,0.),launched=True,until=10.)
        self.tick(replace(ctx,now=.1,ball=BallState(.6,.3)))
        self.assertEqual(self.players[1].action,'receive:intercept')
        self.assertFalse(self.players[0].is_kicking)
        self.assertIsNotNone(self.store.tactics.pass_plan)

    def test_distant_pass_receiver_does_not_create_second_receiver(self):
        ctx=replace(context(),ball=BallState(1.,0.),teammates={
            1:RobotState(1,Pose2D(-1.,0.)),2:RobotState(2,Pose2D(6.,3.)),
            3:RobotState(3,Pose2D(.9,0.))})
        self.tick(ctx)
        self.store.tactics.pass_plan=dict(passer=1,receiver=2,target=(6.,3.),power=1.,
            origin=(0.,0.),launched=True,until=10.)
        self.tick(replace(ctx,now=.1))
        self.assertEqual([p.id for p in self.players if p.action=='receive:intercept'],[2])

    def test_loose_ball_in_own_half_keeps_goal_cover(self):
        self.tick(replace(context(),ball=BallState(-2.,3.)))
        cover=next(p for p in self.players if p.action=='cover:goal')
        self.assertLess(cover._nav_goal[0],-5.)

    def test_historical_receiver_owner_does_not_end_moving_pass(self):
        ctx=replace(context(),ball=BallState(1.,0.))
        for p in self.players:p.context=ctx
        tactics=self.store.tactics
        tactics.pass_plan=dict(passer=1,receiver=2,target=(3.,0.),power=1.,
            origin=(0.,0.),launched=True,until=10.)
        self.store.evaluation.owner=('ours',2)
        self.store.evaluation.candidate=None
        tactics._update_pass(ctx,self.players,SimpleNamespace(speed=2.))
        self.assertIsNotNone(tactics.pass_plan)

    def test_receive_movement_does_not_avoid_incoming_ball(self):
        p=self.players[0]
        p.context=replace(context(),ball=BallState(-1.,0.),teammates={1:RobotState(1,Pose2D(-2.,0.))})
        p.receive_ball((-1.,0.))
        velocity=[c for c in p._backend.calls if c[0]=='velocity'][-1]
        self.assertGreater(velocity[1],0.)

    def test_keeper_is_allowed_to_be_first_defender(self):
        ctx=replace(context(teams=(TeamState(team_number=1),TeamState(team_number=2,score=1))),
            ball=BallState(-4.,0.),teammates={1:RobotState(1,Pose2D(-4.5,0.)),
                2:RobotState(2,Pose2D(1.,2.)),3:RobotState(3,Pose2D(2.,-2.))})
        self.tick(ctx)
        self.assertEqual(self.players[0].action,'penalty:clear')
        self.assertTrue(self.players[0].is_kicking)

    def test_goal_line_kick_survives_short_step_behind_line(self):
        ctx=replace(context(),ball=BallState(-6.95,0.),
                    teammates={1:RobotState(1,Pose2D(-7.2,0.))})
        self.tick(ctx)
        self.assertEqual(self.players[0].action,'goal:clear')
        self.assertTrue(self.players[0].is_kicking)
        self.tick(replace(ctx,game=replace(ctx.game,stopped=True)))
        self.assertFalse(self.players[0].is_kicking)

    def test_goal_pocket_does_not_override_opponent_restart(self):
        from src.framework.types import SetPlay
        ctx=replace(context(set_play=SetPlay.CORNER_KICK,kicking_team=2),
                    ball=BallState(-6.95,0.),teammates={1:RobotState(1,Pose2D(-7.2,0.))})
        self.tick(ctx)
        self.assertFalse(self.players[0].is_kicking)

    def test_fast_ball_is_intercepted_before_sdk_kick(self):
        p=self.players[0]
        p.context=replace(context(),ball=BallState(-1.,0.),
                          teammates={1:RobotState(1,Pose2D(-4.,0.))})
        p.ball_motion=SimpleNamespace(speed=2.)
        p.play_ball((-1.5,.5),(7.,0.),1.)
        self.assertFalse(p.is_kicking)
        self.assertFalse(any(c[0]=='kick' for c in p._backend.calls))
        self.assertTrue(any(c[0]=='velocity' and any(c[1:3]) for c in p._backend.calls))

    def test_near_fast_ball_keeps_continuous_sdk_control(self):
        p=self.players[0]
        p.context=replace(context(),ball=BallState(-1.,0.),
                          teammates={1:RobotState(1,Pose2D(-2.5,0.))})
        p.ball_motion=BallMotion()
        p.ball_motion.velocity=(2.,0.)
        p.rule_can_kick=True
        p.play_ball((-1.,0.),(7.,0.),SHOT_POWER)
        self.assertTrue(p.is_kicking)
        p._backend.calls.clear()
        p.play_ball((-1.,0.),(7.,0.),SHOT_POWER)
        self.assertFalse(any(c[0]=='release' for c in p._backend.calls))

    def test_defense_keeps_goal_cover_against_decoy_mark(self):
        ctx=replace(context(),ball=BallState(-2.,0.),opponents={
            4:RobotState(4,Pose2D(-2.,.1)),5:RobotState(5,Pose2D(4.,3.)),6:RobotState(6,Pose2D(5.,-3.))})
        self.tick(ctx)
        self.assertEqual(len(set(self.store.tactics.mark_assignments.values())),3)
        cover=next(p for p in self.players if p.action=='mark:goal_cover')
        self.assertLess(cover._nav_goal[0],-5.)

    def test_ready_outside_player_first_returns_to_nearest_legal_point(self):
        ctx=replace(context(GameState.READY),teammates={1:RobotState(1,Pose2D(3.,4.7))})
        self.tick(ctx)
        p=self.players[0]
        self.assertEqual(p.action,'ready:legal_first')
        self.assertLess(p._nav_goal[0],0.)
        self.assertLessEqual(abs(p._nav_goal[1]),ctx.field.width/2-FIELD_SAFE_MARGIN_M)


class EvaluationScenarios(unittest.TestCase):
    def setUp(self):self.e=MatchEvaluation()
    def controlled(self,side,now,x=0.,y=0.):
        return replace(context(),now=now,ball=BallState(x,y),
            teammates={1:RobotState(1,Pose2D(x if side=='ours' else x-3,y))},
            opponents={4:RobotState(4,Pose2D(x if side=='theirs' else x+3,y))})

    def test_official_goals_and_penalties_count_once(self):
        self.e.update(context())
        ps=(PlayerState(penalty=Penalty.PUSHING),*tuple(PlayerState() for _ in range(19)))
        ctx=context(teams=(TeamState(team_number=1,score=1,players=ps),TeamState(team_number=2)))
        self.e.update(ctx);self.e.update(ctx)
        self.assertAlmostEqual(self.e.total,REWARD_GOAL+REWARD_PENALTY)
        self.assertEqual(len(self.e.events),2)

    def test_intercept_and_dispossessed_require_confirmed_owner_change(self):
        for side,t in (('theirs',0.),('theirs',.3),('ours',.5),('ours',.8),('theirs',1.),('theirs',1.3)):
            self.e.update(self.controlled(side,t))
        self.assertEqual([e['event'] for e in self.e.events],['intercept','loss','dispossessed'])
        self.assertAlmostEqual(self.e.total,-.2)

    def test_completed_pass_and_missed_pass_not_per_frame_rewards(self):
        self.e.start_pass(context(),dict(receiver=1,passer=2,until=3.,origin=(-2.,0.)))
        for t in (0.,.3,.6):self.e.update(self.controlled('ours',t))
        self.assertEqual([e['event'] for e in self.e.events],['receive'])
        self.e.start_pass(context(),dict(receiver=2,passer=1,until=1.,origin=(0.,0.)))
        self.e.update(self.controlled('ours',2.))
        self.assertEqual([e['event'] for e in self.e.events],['receive','missed_pass'])
        self.assertAlmostEqual(self.e.total,0.)

    def test_clearance_credited_only_after_ball_exits_penalty_area(self):
        ctx=self.controlled('ours',0.,-6.,0.)
        self.e.record_kick(ctx,1,(-1.,0.));self.e.update(ctx)
        self.assertEqual(self.e.total,0.)
        self.e.update(self.controlled('ours',1.,-3.,0.))
        self.e.update(self.controlled('ours',1.5,-2.5,0.))
        self.assertEqual([e['event'] for e in self.e.events],['penalty_area_clearance'])

    def test_boundary_rescue_credited_after_return_to_infield(self):
        ctx=self.controlled('ours',0.,0.,4.4)
        self.e.record_kick(ctx,1,(2.,0.));self.e.update(ctx)
        self.e.update(self.controlled('ours',1.,1.,2.))
        self.e.update(self.controlled('ours',1.5,1.,1.8))
        self.assertEqual([e['event'] for e in self.e.events],['boundary_save'])

    def test_goal_line_position_never_awards_goal(self):
        self.e.update(replace(context(),ball=BallState(7.1,0.)))
        self.assertEqual(self.e.total,0.)


if __name__=='__main__':unittest.main()
