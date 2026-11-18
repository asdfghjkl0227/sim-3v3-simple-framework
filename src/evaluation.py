"""用户奖励账本。比分/罚下为官方事实，其余为公开位置的事件估计。

只用于战术评估和日志，不向裁判写状态，不当作官方触球证据。
"""
from collections import deque
import logging
import math

from .framework.types import GameState, Penalty
from .param import *

_log = logging.getLogger(__name__)


class MatchEvaluation:
    def __init__(self):
        self.total = 0.0
        self.events = deque(maxlen=EVENT_HISTORY_LIMIT)
        self.owner = None
        self.candidate = None
        self.candidate_since = None
        self.loose_since = None
        self.score = None
        self.penalties = set()
        self.pending_pass = None
        self.pending_kick = None
        self.last_phase_live = False

    def award(self,context,event,value,player=None,official=False):
        self.total += value
        self.events.append(dict(at=context.now,event=event,value=value,player=player,
                                official=official,total=self.total))
        _log.info('EVAL event=%s value=%+.2f total=%.2f player=%s evidence=%s',
                  event,value,self.total,player,'official' if official else 'estimated')

    def start_pass(self,context,plan):
        self.pending_pass = dict(receiver=plan['receiver'],passer=plan['passer'],
                                 until=plan['until'],origin=plan['origin'])

    def record_kick(self,context,player,target):
        if self.pending_kick is not None or context.ball is None:
            return
        b,f = context.ball,context.field
        self.pending_kick = dict(player=player,origin=(b.x,b.y),at=context.now,
            clearance=b.x < -f.length/2+f.penalty_area_length and abs(b.y)<f.penalty_area_width/2,
            boundary=abs(b.y)>f.width/2-TOUCHLINE_ZONE_M,
            saved=False,cleared=False)

    def finish_pass(self,context,success):
        plan = self.pending_pass
        if plan is None:
            return
        self.award(context,'receive' if success else 'missed_pass',
                   REWARD_RECEIVE if success else REWARD_LOSS,plan['receiver'])
        self.pending_pass = None

    def update(self,context):
        g = context.game
        if g is None:
            return
        team = g.get_team_state(context.team_id)
        if team is not None:
            if self.score is not None and team.score > self.score:
                self.award(context,'goal',REWARD_GOAL*(team.score-self.score),official=True)
            self.score = team.score
            penalties = {i+1 for i,p in enumerate(team.players) if p.penalty not in
                         (Penalty.NONE,Penalty.SUBSTITUTE)}
            for player in penalties-self.penalties:
                self.award(context,'penalty',REWARD_PENALTY,player,official=True)
            self.penalties = penalties
        live = g.state == GameState.PLAYING and not g.stopped
        if not live or context.ball is None:
            if self.pending_pass is not None:
                self.finish_pass(context,False)
            self.owner = self.candidate = self.candidate_since = self.loose_since = None
            self.pending_kick = None
            return
        ball = (context.ball.x,context.ball.y)
        ours = [(math.dist(ball,(r.pose.x,r.pose.y)),i) for i,r in context.teammates.items()
                if r.pose is not None and (team is None or i>len(team.players)
                                          or team.players[i-1].penalty == Penalty.NONE)]
        theirs = [(math.dist(ball,(r.pose.x,r.pose.y)),i) for i,r in context.opponents.items()
                  if r.pose is not None]
        od,oi = min(ours,default=(math.inf,None))
        td,ti = min(theirs,default=(math.inf,None))
        claim = ('ours',oi) if od<CONTROL_REACH_M and od+CONTROL_ADVANTAGE_M<td else (
            ('theirs',ti) if td<CONTROL_REACH_M and td+CONTROL_ADVANTAGE_M<od else None)
        if claim != self.candidate:
            self.candidate,self.candidate_since = claim,context.now
        if claim is not None and context.now-self.candidate_since >= CONTROL_HOLD_S:
            old = self.owner
            if old != claim:
                if old is not None and old[0]=='theirs' and claim[0]=='ours':
                    self.award(context,'intercept',REWARD_INTERCEPT,claim[1])
                elif old is not None and old[0]=='ours' and claim[0]=='theirs':
                    self.award(context,'loss',REWARD_LOSS,old[1])
                    self.award(context,'dispossessed',REWARD_DISPOSSESSED,old[1])
                    self.finish_pass(context,False)
                self.owner = claim
            self.loose_since = None
        elif claim is None:
            if self.loose_since is None:
                self.loose_since = context.now
            if (self.owner is not None and self.owner[0]=='ours' and self.pending_pass is None
                    and context.now-self.loose_since > CONTROL_LOSS_S):
                self.award(context,'loss',REWARD_LOSS,self.owner[1])
                self.owner = None
        plan = self.pending_pass
        if plan is not None:
            if self.owner == ('ours',plan['receiver']) and self.candidate == self.owner:
                self.finish_pass(context,True)
            elif context.now > plan['until']:
                self.finish_pass(context,False)
        kick = self.pending_kick
        if kick is not None:
            if context.now-kick['at'] > EVENT_WINDOW_S or (self.owner and self.owner[0]=='theirs'):
                self.pending_kick = None
                return
            f = context.field
            travel = math.dist(ball,kick['origin'])
            if (kick['clearance'] and not kick['cleared'] and travel>EVENT_CLEARANCE_TRAVEL_M
                    and ball[0]>-f.length/2+f.penalty_area_length+EVENT_CLEARANCE_MARGIN_M
                    and abs(ball[1])<f.width/2):
                self.award(context,'penalty_area_clearance',REWARD_CLEARANCE,kick['player'])
                kick['cleared'] = True
            if (kick['boundary'] and not kick['saved'] and travel>EVENT_CLEARANCE_TRAVEL_M
                    and abs(ball[1])<f.width/2-EVENT_BOUNDARY_RECOVER_M and abs(ball[0])<f.length/2):
                self.award(context,'boundary_save',REWARD_BOUNDARY_SAVE,kick['player'])
                kick['saved'] = True
