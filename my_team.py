# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from contest.distance_calculator import Distancer


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAStarAgent', second='DefensiveAStarAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


#############
# My Agents #
#############

class OffensiveAStarAgent(CaptureAgent):
    """
    Offensive A* Agent:
    This agent uses the A* search algorithm to navigate the maze in an offensive manner.
    It seeks enemy-side food, tries to avoid or exploit scared ghosts, and reacts to enemy proximity.
    """

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.start_position = None

    def register_initial_state(self, game_state):
        """
        Called before the game starts, sets the agent's initial position.
        """
        self.start_position = game_state.get_agent_position(self.index)
        super().register_initial_state(game_state)

    def get_closest_scared_ghost(self, game_state):
        """
        Find the closest scared ghost visible to this agent.

        Returns:
            (opponent_index, position, scared_timer) of the closest scared ghost,
            or None if no visible scared ghosts.
        """
        opponents = self.get_opponents(game_state)
        my_pos = game_state.get_agent_state(self.index).get_position()
        scared_ghosts = []

        for opp_index in opponents:
            enemy_state = game_state.get_agent_state(opp_index)
            # Only consider enemies that are ghosts and currently scared
            if not enemy_state.is_pacman and enemy_state.scared_timer > 0:
                enemy_pos = enemy_state.get_position()
                if enemy_pos is not None:  # Only consider visible ghosts
                    scared_ghosts.append(
                        (opp_index, enemy_pos, enemy_state.scared_timer))

        if not scared_ghosts:
            return None

        # Return the scared ghost closest to the agent
        return min(scared_ghosts, key=lambda g: self.get_maze_distance(my_pos, g[1]))

    def get_closest_enemy_distance(self, game_state):
        """
        Compute the shortest known distance from this agent to any enemy.

        Returns:
            int: Minimum distance to a known enemy. If an enemy's position is unknown, 
                 a default large distance (100) is used.
        """
        distancer = Distancer(game_state.data.layout)
        my_pos = game_state.get_agent_state(self.index).get_position()
        opponents = self.get_opponents(game_state)

        distances = []
        for opp_index in opponents:
            enemy_pos = game_state.get_agent_position(opp_index)
            if enemy_pos is None:
                distances.append(100)
            else:
                distances.append(distancer.get_distance(enemy_pos, my_pos))

        return min(distances) if distances else 100

    def select_goal(self, game_state):
        """
        Determine the most appropriate goal for this agent given the current game state.

        Priority considerations:
        1. Return home if no food is available.
        2. If a scared ghost is present, target it.
        3. If too close to enemies, seek a capsule if available or return home if not.
        4. Otherwise, target the nearest food.

        Returns:
            tuple (x, y): coordinates of the chosen goal.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        is_pacman_mode = game_state.get_agent_state(self.index).is_pacman
        available_food = self.get_food(game_state).as_list()
        closest_scared_ghost = self.get_closest_scared_ghost(game_state)
        closest_enemy_distance = self.get_closest_enemy_distance(game_state)

        # If no food is available, go home
        if not available_food:
            return game_state.get_initial_agent_position(self.index)

        # If we are Pacman:
        # Check for special conditions
        if is_pacman_mode:
            # If there's a scared ghost, target it
            if closest_scared_ghost:
                return closest_scared_ghost[1]

            # If enemies are dangerously close
            if closest_enemy_distance < 4:
                capsules = self.get_capsules(game_state)
                if capsules:
                    # Choose the closest capsule
                    return min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
                else:
                    # No capsules, retreat home
                    return game_state.get_initial_agent_position(self.index)

            # Default: choose the nearest food if no other conditions triggered
            return min(available_food, key=lambda food: self.get_maze_distance(my_pos, food))
        else:
            # If we are a ghost (not Pacman), simply go for the nearest food
            return min(available_food, key=lambda food: self.get_maze_distance(my_pos, food))

    def calculate_heuristic(self, current_pos, goal_pos, closest_enemy_distance):
        """
        Calculate a heuristic value to guide A* search.

        This heuristic considers:
        - The maze distance to the goal.
        - A small penalty for being too close to enemies. Instead of 
          dividing by distance, we can add a penalty that inversely scales with it.

        Args:
            current_pos (tuple): Current (x, y) position of the agent.
            goal_pos (tuple): Target (x, y) position.
            closest_enemy_distance (int): The shortest known distance to any enemy.

        Returns:
            float: A heuristic cost. Lower values encourage shorter paths and safer routes.
        """
        # Base heuristic: just the maze distance to the goal.
        base_distance = self.get_maze_distance(current_pos, goal_pos)

        # Add a slight penalty if enemies are close. The closer the enemy, the higher the penalty.
        # For example, if enemy_distance = 2, penalty might be higher than if enemy_distance = 10.
        # We'll use a small fraction like (10.0 / max(closest_enemy_distance, 1)) to avoid division by zero.
        enemy_penalty = 10.0 / max(closest_enemy_distance, 1)

        return base_distance + enemy_penalty

    def choose_action(self, game_state):
        """
        Choose an action by:
        1. Selecting a goal.
        2. Running A* to find a path to that goal.
        3. Returning the first step of the best found path.

        If no valid path is found to the chosen goal, fallback to going home.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        closest_enemy_distance = self.get_closest_enemy_distance(game_state)
        chosen_goal = self.select_goal(game_state)

        path = self.a_star_search(
            game_state, my_pos, chosen_goal, closest_enemy_distance)

        if path and len(path) > 0:
            return path[0]
        else:
            # Fallback: go home if no path to the chosen goal is found
            home_pos = game_state.get_initial_agent_position(self.index)
            fallback_path = self.a_star_search(
                game_state, my_pos, home_pos, closest_enemy_distance)
            if fallback_path and len(fallback_path) > 0:
                return fallback_path[0]
            # If even fallback fails, stop to avoid errors.
            return Directions.STOP

    def a_star_search(self, initial_game_state, start_pos, goal_pos, closest_enemy_distance):
        """
        Perform A* search to find a path from start_pos to goal_pos.

        Args:
            initial_game_state: The game state from which to start the search.
            start_pos (tuple): The starting (x, y) position.
            goal_pos (tuple): The goal (x, y) position.
            closest_enemy_distance (int): The shortest distance to an enemy, used in heuristic.

        Returns:
            list: A list of actions leading from start to goal. Empty list if no path is found.
        """
        expanded_nodes = set()
        frontier = util.PriorityQueue()
        frontier.push((initial_game_state, start_pos, []), 0)
        cost_so_far = {start_pos: 0}

        while not frontier.is_empty():
            current_state, current_pos, path = frontier.pop()

            if current_pos == goal_pos:
                return path  # Goal reached

            if current_pos in expanded_nodes:
                continue
            expanded_nodes.add(current_pos)

            for action in current_state.get_legal_actions(self.index):
                if action == Directions.STOP:
                    continue

                successor_state = current_state.generate_successor(
                    self.index, action)
                next_pos = successor_state.get_agent_state(
                    self.index).get_position()
                next_pos = nearest_point(next_pos)

                new_cost = cost_so_far[current_pos] + 1  # Uniform step cost
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    new_path = path + [action]
                    heuristic_value = self.calculate_heuristic(
                        next_pos, goal_pos, closest_enemy_distance)
                    priority = new_cost + heuristic_value
                    frontier.push(
                        (successor_state, next_pos, new_path), priority)

        # No path found
        return []


class DefensiveAStarAgent(CaptureAgent):
    """
    Defensive A* Agent:
    This agent uses A* to move defensively. Its behavior:
    - If an enemy Pacman (invader) is visible, it tries to chase them down.
    - If the only visible enemy is a scared ghost, it runs away from it.
    - If no enemies are visible, it positions itself strategically (e.g., near its start or a patrol point).
    """

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.start_position = None

    def register_initial_state(self, game_state):
        """
        Called before the game starts, sets the agent's initial position.
        """
        self.start_position = game_state.get_agent_position(self.index)
        super().register_initial_state(game_state)

    def get_visible_invaders(self, game_state):
        """
        Returns a list of (opponent_index, position) for visible enemies that are Pacman (invaders).
        """
        invaders = []
        for opp_index in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp_index)
            if opp_state.is_pacman and opp_state.get_position() is not None:
                invaders.append((opp_index, opp_state.get_position()))
        return invaders

    def get_visible_scared_ghosts(self, game_state):
        """
        Returns a list of (opponent_index, position, scared_timer) for visible scared enemy ghosts.
        """
        scared_ghosts = []
        for opp_index in self.get_opponents(game_state):
            opp_state = game_state.get_agent_state(opp_index)
            if not opp_state.is_pacman and opp_state.scared_timer > 0:
                pos = opp_state.get_position()
                if pos is not None:
                    scared_ghosts.append(
                        (opp_index, pos, opp_state.scared_timer))
        return scared_ghosts

    def get_closest_enemy_distance(self, game_state):
        """
        Compute the shortest known distance from this agent to any enemy.
        If unknown, return a large number.
        """
        distancer = Distancer(game_state.data.layout)
        my_pos = game_state.get_agent_state(self.index).get_position()
        distances = []
        for opp_index in self.get_opponents(game_state):
            enemy_pos = game_state.get_agent_position(opp_index)
            if enemy_pos is None:
                distances.append(100)
            else:
                distances.append(distancer.get_distance(enemy_pos, my_pos))
        return min(distances) if distances else 100

    def select_goal(self, game_state):
        """
        Determine the appropriate defensive goal:
        - If there's at least one visible invader, chase the closest one.
        - Otherwise, if there's a visible scared ghost, run away from it by choosing a safe spot.
        - If no threats are visible, stay near start or pick a patrol location.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        invaders = self.get_visible_invaders(game_state)
        scared_ghosts = self.get_visible_scared_ghosts(game_state)

        # 1. If we see any invaders, chase the closest one
        if invaders:
            return min([pos for _, pos in invaders], key=lambda p: self.get_maze_distance(my_pos, p))

        # 2. If we see a scared ghost, run away.
        #    We'll pick our start position as a safe spot, or any point far from the ghost.
        if scared_ghosts:
            # Running away: choose a position far from the closest scared ghost.
            # Here, we simply pick our start as a safe spot. Another approach could be
            # to find a spot maximizing distance from the ghost.
            return self.start_position

        # 3. If no enemies are visible, just return to start position or any defensive location.
        return self.start_position

    def calculate_heuristic(self, current_pos, goal_pos, closest_enemy_distance):
        """
        A heuristic for guiding A* search.
        We'll primarily use the maze distance to the goal. We could incorporate
        enemy proximity if desired, but as a defensive agent focusing on reaching
        a goal (like chasing an invader), a straightforward distance-based heuristic
        might suffice.
        """
        return self.get_maze_distance(current_pos, goal_pos)

    def choose_action(self, game_state):
        """
        Choose an action by:
        1. Selecting a defensive goal based on current conditions.
        2. Running A* to find a path to that goal.
        3. Returning the first step of the best found path.

        If no path is found, do a fallback (e.g., stay put or return home).
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        closest_enemy_distance = self.get_closest_enemy_distance(game_state)
        chosen_goal = self.select_goal(game_state)

        path = self.a_star_search(
            game_state, my_pos, chosen_goal, closest_enemy_distance)

        if path and len(path) > 0:
            return path[0]
        else:
            # Fallback: If no path, just stop to avoid random movements
            return Directions.STOP

    def a_star_search(self, initial_game_state, start_pos, goal_pos, closest_enemy_distance):
        """
        Perform A* search to find a path from start_pos to goal_pos.
        Similar to the OffensiveAStarAgent implementation.
        """
        expanded_nodes = set()
        frontier = util.PriorityQueue()
        frontier.push((initial_game_state, start_pos, []), 0)
        cost_so_far = {start_pos: 0}

        while not frontier.is_empty():
            current_state, current_pos, path = frontier.pop()

            if current_pos == goal_pos:
                return path  # Goal reached

            if current_pos in expanded_nodes:
                continue
            expanded_nodes.add(current_pos)

            for action in current_state.get_legal_actions(self.index):
                if action == Directions.STOP:
                    continue

                successor_state = current_state.generate_successor(
                    self.index, action)
                next_pos = successor_state.get_agent_state(
                    self.index).get_position()
                next_pos = nearest_point(next_pos)

                new_cost = cost_so_far[current_pos] + 1  # uniform step cost
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    new_path = path + [action]
                    heuristic_value = self.calculate_heuristic(
                        next_pos, goal_pos, closest_enemy_distance)
                    priority = new_cost + heuristic_value
                    frontier.push(
                        (successor_state, next_pos, new_path), priority)

        # No path found
        return []

##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = - \
            len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food)
                               for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i)
                   for i in self.get_opponents(successor)]
        invaders = [
            a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(
                my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
