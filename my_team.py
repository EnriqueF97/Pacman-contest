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
        5. If pacman is carrying 5 foods, return home.

        Returns:
            tuple (x, y): coordinates of the chosen goal.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        is_pacman_mode = game_state.get_agent_state(self.index).is_pacman
        available_food = self.get_food(game_state).as_list()
        closest_scared_ghost = self.get_closest_scared_ghost(game_state)
        closest_enemy_distance = self.get_closest_enemy_distance(game_state)
        numFoodCarrying = game_state.get_agent_state(self.index).num_carrying

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

            # If agent is carrying 5 foods
            if numFoodCarrying >= 5:
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
    - If scared: run back to start.
    - If invaders appear: chase closest invader.
    - Else: patrol a chosen midpoint location.
    Uses A* to reach these goals efficiently.
    """

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.start_position = None
        self.mid_x = None
        self.is_red = None
        self.patrol_point = None

    def register_initial_state(self, game_state):
        # Initialize positions and determine patrol point near the map's center.
        self.start_position = game_state.get_agent_position(self.index)
        super().register_initial_state(game_state)
        layout = game_state.data.layout
        w, h = layout.width, layout.height
        self.is_red = game_state.is_on_red_team(self.index)
        self.mid_x = w // 2
        points = []
        # Attempt to find a valid patrol spot along the middle boundary.
        for y in range(h):
            if not game_state.has_wall(self.mid_x - (0 if self.is_red else 1), y):
                px = self.mid_x - 1 if self.is_red else self.mid_x
                points.append((px, y))
        self.patrol_point = points[len(
            points)//2] if points else self.start_position

    def get_current_position(self, game_state):
        return game_state.get_agent_state(self.index).get_position()

    def is_on_our_side(self, pos):
        # Determine if a position is on our team's side based on the mid line.
        return pos[0] < self.mid_x if self.is_red else pos[0] >= self.mid_x

    def get_visible_invaders(self, game_state):
        # Visible enemy Pacmen on our side.
        invaders = []
        for o in self.get_opponents(game_state):
            st = game_state.get_agent_state(o)
            if st.is_pacman and st.get_position() and self.is_on_our_side(st.get_position()):
                invaders.append((o, st.get_position()))
        return invaders

    def get_closest_enemy_distance(self, game_state):
        # Compute shortest visible enemy distance or fallback to 100.
        distancer = Distancer(game_state.data.layout)
        my_pos = self.get_current_position(game_state)
        dists = []
        for o in self.get_opponents(game_state):
            epos = game_state.get_agent_position(o)
            dists.append(
                100 if epos is None else distancer.get_distance(epos, my_pos))
        return min(dists) if dists else 100

    def get_closest_enemy_position(self, game_state):
        # Return position of the closest visible enemy, else None.
        distancer = Distancer(game_state.data.layout)
        my_pos = self.get_current_position(game_state)
        visible = []
        for o in self.get_opponents(game_state):
            epos = game_state.get_agent_position(o)
            if epos:
                visible.append(epos)
        return min(visible, key=lambda p: distancer.get_distance(p, my_pos)) if visible else None

    def is_scared(self, game_state):
        # Check if this agent is currently a scared ghost.
        st = game_state.get_agent_state(self.index)
        return st.scared_timer > 0 and not st.is_pacman

    def find_valid_patrol_point(self, game_state):
        """
        Dynamically find a valid patrol point near the map's middle boundary.
        Ensures the patrol point isn't blocked by walls.
        """
        height = game_state.data.layout.height
        points = []

        # Search along the middle column (or columns, based on the team's side)
        for y in range(height):
            patrol_x = self.mid_x - (1 if self.is_red else 0)
            if not game_state.has_wall(patrol_x, y):
                points.append((patrol_x, y))

        # Return the middle patrol point or fallback to the start position
        return points[len(points) // 2] if points else self.start_position

    def select_goal(self, game_state):
        # Decide the goal based on current conditions:
        # Scared -> start position.
        # Visible invader -> chase it.
        # Else -> patrol point.
        inv = self.get_visible_invaders(game_state)
        scared = self.is_scared(game_state)
        if scared:
            enemy = self.get_closest_enemy_position(game_state)
            return self.start_position if enemy else self.start_position
        if inv:
            my_pos = self.get_current_position(game_state)
            return min((p for _, p in inv), key=lambda p: self.get_maze_distance(my_pos, p))
        if not self.patrol_point or game_state.has_wall(*self.patrol_point):
            self.patrol_point = self.find_valid_patrol_point(game_state)

        return self.patrol_point

    def calculate_heuristic(self, game_state, cpos, gpos):
        # Simple heuristic: just the maze distance.
        return self.get_maze_distance(cpos, gpos)

    def choose_action(self, game_state):
        # Compute path via A* and take the first action.
        edist = self.get_closest_enemy_distance(game_state)
        goal = self.select_goal(game_state)
        path = self.a_star_search(game_state, goal, edist)
        return path[0] if path else Directions.STOP

    def a_star_search(self, init_state, goal, edist):
        # A* search from current position to goal.
        start = self.get_current_position(init_state)
        expanded = set()
        frontier = util.PriorityQueue()
        frontier.push((init_state, start, []), 0)
        cost = {start: 0}
        while not frontier.is_empty():
            st, pos, p = frontier.pop()
            if pos == goal:
                return p
            if pos in expanded:
                continue
            expanded.add(pos)
            for a in st.get_legal_actions(self.index):
                if a == Directions.STOP:
                    continue
                succ = st.generate_successor(self.index, a)
                npos = succ.get_agent_state(self.index).get_position()
                npos = nearest_point(npos)
                new_cost = cost[pos] + 1
                if npos not in cost or new_cost < cost[npos]:
                    cost[npos] = new_cost
                    h = self.calculate_heuristic(succ, npos, goal)
                    frontier.push((succ, npos, p + [a]), new_cost + h)
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
