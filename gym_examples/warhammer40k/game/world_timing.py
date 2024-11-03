from gym_examples.warhammer40k.game.phase import Phase

def get_next_world_time_state(current_phase, current_turn, current_player_round, players, agents_by_player):
    # return Turn, Player index, phase
    
    if(current_phase == Phase.Init):
        return first_turn()
    
    if(current_phase == Phase.Movement):
        if all_agents_by_player_moved(agents_by_player, current_player_round):            
            return shooting_phase(current_turn, current_player_round)
        
    if(current_phase == Phase.Shooting):            
        if all_agents_by_player_shot(agents_by_player, current_player_round):
            if last_player_round(current_player_round, players):
                return next_turn_movement_phase(current_turn)
            
            return next_round_movement_phase(current_turn)
            
    return (current_turn, current_player_round, current_phase)

def first_turn():
    # Turn, Player index, phase
    return (1, 0, Phase.Movement)

def shooting_phase(current_turn, current_player_round):
    # Turn, Player index, phase
    return (current_turn, current_player_round, Phase.Shooting)

def next_turn_movement_phase(current_turn):
    # Turn, Player index, phase
    return (current_turn + 1, 0, Phase.Movement)

def next_round_movement_phase(current_turn):
    # Turn, Player index, phase
    return (current_turn, 1, Phase.Movement)

def all_agents_by_player_moved(agents_by_player, current_player_round):
    return all([agent.has_moved_this_phase for agent in agents_by_player[current_player_round]])

def all_agents_by_player_shot(agents_by_player, current_player_round):
    return all([agent.has_shot_this_phase for agent in agents_by_player[current_player_round]])

def last_player_round(current_player_round, players):
    return current_player_round == len(players) - 1