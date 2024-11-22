import numpy as np
import pandas as pd
from typing import List
from abc import ABC, abstractmethod

ME_INDEX = 0


class GDPlayer:
    def __init__(self):
        self.dual_vars = [[1,1]]
        self.iteration_count = 0
        
    def update_grad(self, grad_budg: float, grad_roi: float):

        stepsize = 1/(1+self.iteration_count)
        dual_budg, dual_roi = self.dual_vars[-1]
        
        dual_budg_new = np.maximum(0, dual_budg - stepsize * grad_budg)
        dual_roi_new = np.maximum(0, dual_roi - stepsize * grad_roi)
        self.dual_vars.append([dual_budg_new,dual_roi_new])
        self.iteration_count +=1


class Autobidder(GDPlayer):
    def __init__(self, id: int, rho: float, gamma:float, total_budg: float=None, stop_when_no_budg:bool=False):
        super().__init__()
        self.id = id
        self.rho = rho
        self.gamma = gamma
        self.payments = []
        self.observed_values = []
        self.obtained_values = []
        self.bids = []
        self.costs = []
        self.total_budg = total_budg
        self.stop_when_no_budg = stop_when_no_budg
        if stop_when_no_budg:
            assert self.total_budg is not None, "Must speficy budget when player stops bidding when no budget remaining"
    def submit_bid(self, val: float):
        self.observed_values.append(val)
        dual_budg, dual_roi = self.dual_vars[-1]
        multiplier = (1+dual_roi)/(dual_budg + dual_roi * self.gamma)
        return multiplier * val
    
    def update(self, val: float, cost: float):
        self.obtained_values.append(val)
        self.payments.append(cost)
        is_winner = cost > 0
        g1 = (val - self.gamma * cost) * is_winner
        g2 = self.rho - cost 
        self.update_grad(grad_budg=g2,grad_roi=g1)
        
    def get_total_val(self):
        return np.sum(self.obtained_values)
    def get_total_payment(self):
        return np.sum(self.payments)
    
    

class Auction(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def allocate(self, bids: List[float]):
        pass
    
class SPA(Auction):
    def __init__(self):
        pass
    def allocate(self, bids: List[float]):
        ranking = np.argsort(bids)
        allocation = [0] * len(bids)
        payment = [0] * len(bids)
        allocation[ranking[0]]=1
        payment[ranking[0]]= bids[ranking[1]]
        return allocation, payment
    
class Channel:
    def __init__(self):
        self.players_over_phases = []
        self.phase_lens = []
        self.historical_payments = []
        self.historical_cost = []
     
    def init_new_phase(self,list_players: List[GDPlayer]):
        self.players_over_phases.append(list_players)
        
    def run_phase(self, values:np.ndarray, auction: Auction, phase_n_players: int =-1):
        num_rounds, num_players = values.shape
        self.phase_lens.append(num_rounds)
        playerlist = self.players_over_phases[phase_n_players]
        assert num_players == len(playerlist), "input values per round must match with number of players"
        phase_payment_hist = []
        phase_cost_hist = []
        for _round in range(num_rounds):
            player_values = values[_round]
            bids = []
            cost = 0
            for ind,player in enumerate(playerlist):
                bids.append(player.submit_bid(player_values[ind]))
                if player.id != ME_INDEX and player.submit_bid(player_values[ind]) > cost:
                    cost = player.submit_bid(player_values[ind])
            allocation, payment = auction.allocate(bids)
            phase_payment_hist.append(np.sum(payment))
            phase_cost_hist.append(cost)
            for ind,player in enumerate(playerlist):
                realized_val = player_values[ind] * allocation[ind]
                realized_cost = payment[ind]
                player.update(realized_val,realized_cost)
        self.historical_payments.append(phase_payment_hist)
        self.historical_cost.append(phase_cost_hist)
        
    def get_phase_total_vals(self, phase_n_players: int =-1):
        playerlist = self.players_over_phases[phase_n_players]
        return [player.get_total_val() for player in playerlist]

    def get_phase_total_pyament(self,phase_n_players: int =-1):
        playerlist = self.players_over_phases[phase_n_players]
        return [player.get_total_payment() for player in playerlist]
    
    
class AlgoRunner(GDPlayer):
    def __init__(
        self, 
        gamma: float,
        budg_per_phase: float, 
        arm_set: np.array, 
        init_budg_alloc: List[float],
        num_channels: int
    ):
        super().__init__()
        self.gamma = gamma
        self.budg_per_phase = budg_per_phase
        self.arm_set = arm_set
        self.budg_alloc = [init_budg_alloc]
        self.num_channels = num_channels
        self.pull_nums = [np.zeros(len(arm_set)) for ind in range(self.num_channels)] 
        self.V_bar = [np.zeros(len(arm_set)) for ind in range(self.num_channels)] 
        self.per_channel_budgets = []
        self.chosen_arms=[]
        self.obtained_values = []
    def update(self, val:List[float], cost: List[float]):
        self.obtained_values.append(val)
        total_spend = 0
        _per_channel_budg = self.per_channel_budgets[-1]
        _chosen_arms = self.chosen_arms[-1]
        for ind in range(self.num_channels):
            chosen_arm_id =  int(_chosen_arms[ind])
            total_spend += cost[ind]
        
            self.V_bar[ind][chosen_arm_id] = self.V_bar[ind][chosen_arm_id] * self.pull_nums[ind][chosen_arm_id] + val[ind]
            self.pull_nums[ind][chosen_arm_id] += 1
            self.V_bar[ind][chosen_arm_id] /= self.pull_nums[ind][chosen_arm_id]
            
        g1 = np.sum(val) - self.gamma * total_spend
        g2 = self.budg_per_phase - total_spend
        self.update_grad(grad_budg=g2, grad_roi=g1)
        self.remaining_budg = g2
            
    def get_phase_budg_alloc(self, scaler_const: float):
        dual_budg, dual_roi = self.dual_vars[-1]
        stepsize = 1/(1+self.iteration_count)
        _per_channel_budg = np.zeros(self.num_channels)
        _chosen_arms = []
        for ind in range(self.num_channels):
            t = self.iteration_count
            if t < len(self.arm_set):
                chosen_arm_ind = t
                _per_channel_budg[ind] = self.arm_set[t]
            else:
                UCB_jt = np.sqrt(scaler_const/self.pull_nums[ind])
                lagr_jt = (self.V_bar[ind] + UCB_jt) * (1+dual_roi) - (dual_roi * self.gamma + dual_budg) * self.arm_set
                chosen_arm_ind = np.argmax(lagr_jt)
                _per_channel_budg[ind] = chosen_arm_ind
            _chosen_arms.append(chosen_arm_ind)
        self.per_channel_budgets.append(_per_channel_budg)
        self.chosen_arms.append(_chosen_arms)
        return _per_channel_budg
        
    
            
    @staticmethod
    def deterministic_best_alloc(
         vals:List[float],
         costs:List[float],
         budget:float, 
         roi:float
    ) -> List[float]:
        assert(len(vals) == len(costs))
        alloc = [0] * len(vals)
        val_cost_ratio = [vals[ind]/costs[ind] for ind in range(len(vals))]

        sorted_index = np.argsort(val_cost_ratio)[::-1]
        budget_balance = np.cumsum([costs[ind] for ind in sorted_index])
        roi_balance = np.cumsum([vals[ind] - roi * costs[ind] for ind in sorted_index])
        simple_budget_alloc = budget_balance <= budget
        simple_roi_alloc = roi_balance >= 0

        budget_alloc = simple_budget_alloc * 1.0
        roi_alloc = simple_roi_alloc * 1.0
        budget_cutoff_ind = sum(simple_budget_alloc)
        roi_cutoff_ind = sum(simple_roi_alloc)

        sorted_costs = [costs[ind] for ind in sorted_index]
        sorted_vals = [vals[ind] for ind in sorted_index]
        if (budget_cutoff_ind <= len(vals) - 1) and (budget_cutoff_ind > 0):
            budget_alloc[budget_cutoff_ind] = (budget - budget_balance[budget_cutoff_ind-1])/sorted_costs[budget_cutoff_ind]
        elif budget_cutoff_ind == 0:
            budget_alloc[0] = budget/sorted_costs[budget_cutoff_ind]
        if (roi_cutoff_ind <= len(vals) - 1) and (roi_cutoff_ind > 0):
            roi_alloc[roi_cutoff_ind] = roi_balance[roi_cutoff_ind-1]/(roi * sorted_costs[roi_cutoff_ind]- sorted_vals[roi_cutoff_ind])
        alloc = np.minimum(budget_alloc,roi_alloc)
        res = np.zeros(len(vals))
        for ind in range(len(vals)):
            res[sorted_index[ind]] = alloc[ind]
        return res
    

if __name__ == "__main__":
    RECORD_RESULTS = dict()
    for trial_ind, COMPETITOR_NUM in enumerate(range(2,11)):

        """Hyper Params"""
        CHANNELS = [Channel() for ind in range(3)]
        NUM_PHASE = 30
        ROUNDS_PER_PHASE = 200
        TOTAL_ROUNDS = NUM_PHASE * ROUNDS_PER_PHASE
        COMPETITOR_RHO = 0.1
        COMPETITOR_GAMMA = 0.1
        STOP_NO_BUDG = False

        ME_ROI = 0.1
        ME_ARMSET = np.linspace(0,1,11)
        ME_RHO = 0.3
        ME_TOTAL_BUDG = ME_RHO * TOTAL_ROUNDS

        ME = AlgoRunner(
            gamma = ME_ROI, 
            budg_per_phase = ME_RHO,
            arm_set=ME_ARMSET, 
            init_budg_alloc=[min(ME_ARMSET) for channel in CHANNELS],
            num_channels= len(CHANNELS),
        )


        """Main Loop"""
        ME_historical_values = [[] for channel in CHANNELS]
        for phase in range(NUM_PHASE):

            ME_budg_allocation = ME.get_phase_budg_alloc(scaler_const=2*np.log(ROUNDS_PER_PHASE))


            for ind,channel in enumerate(CHANNELS):
                num_competitors = (ind + 1) * COMPETITOR_NUM
                playerlist = [Autobidder(id=ME_INDEX, rho=ME_budg_allocation[ind], gamma= 0, stop_when_no_budg=STOP_NO_BUDG)]\
                + [
                    Autobidder(id=ME_INDEX + i, rho=COMPETITOR_RHO, gamma= COMPETITOR_GAMMA,stop_when_no_budg=STOP_NO_BUDG)
                    for i in range(num_competitors)
                ]
                channel.init_new_phase(playerlist)
                values = np.random.uniform(low=0, high=1,size = (ROUNDS_PER_PHASE,len(playerlist)))
                ME_historical_values[ind].append(values[:,ME_INDEX])
                channel.run_phase(values=values, auction=SPA(), phase_n_players=-1)

            ME_phase_val = [channel.get_phase_total_vals()[ME_INDEX] for channel in CHANNELS]
            ME_phase_cost = [channel.get_phase_total_pyament()[ME_INDEX] for channel in CHANNELS]
            ME.update(val=ME_phase_val, cost=ME_phase_cost)

        """Organize Results"""
        flattened_historical_costs = []
        flattened_ME_historical_values = []
        avg_channel_cost = []
        avg_ME_channel_values = []
        for ind,channel in enumerate(CHANNELS):
            flattened_historical_costs += list(np.hstack(channel.historical_cost))
            avg_channel_cost.append(np.mean(np.hstack(channel.historical_cost)))
            flattened_ME_historical_values += list(np.hstack(ME_historical_values[ind]))
            avg_ME_channel_values.append(np.mean(np.hstack(ME_historical_values[ind])))
        optimal_hindsight_allocation = AlgoRunner.deterministic_best_alloc(
            vals=flattened_ME_historical_values,
            costs=flattened_historical_costs,
            budget=ME_TOTAL_BUDG, 
            roi=ME_ROI
        )

        globopt_wrt_mean_allocation = AlgoRunner.deterministic_best_alloc(
            vals=avg_ME_channel_values,
            costs=avg_channel_cost,
            budget=ME_TOTAL_BUDG/TOTAL_ROUNDS, 
            roi=ME_ROI
        )

        optimal_hindsight_obtained_value = np.array(flattened_ME_historical_values) @ np.array(optimal_hindsight_allocation)
        globopt_wrt_mean = np.array(avg_ME_channel_values) @ np.array(globopt_wrt_mean_allocation) * TOTAL_ROUNDS

        RESULTS = {
            "realized_value": np.sum(ME.obtained_values),
            "optimal_hindsight_obtained_value": optimal_hindsight_obtained_value,
            "globopt_wrt_mean": globopt_wrt_mean
        }
        RECORD_RESULTS[COMPETITOR_NUM] = RESULTS
    RECORD_RESULTS = pd.DataFrame(RECORD_RESULTS).T
