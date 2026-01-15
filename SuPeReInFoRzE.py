import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
import random
import pandas as pd
import json
import zipfile
import io
from copy import deepcopy
import time

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="Super Tic-Tac-Toe RL Arena",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎯"
)

st.title("🎯 Super Tic-Tac-Toe RL Arena")
st.markdown("""
**Ultimate Tic-Tac-Toe**: A game of games! Win 3 small boards in a row to claim victory.

**Game Rules:**
- 🎲 9 small tic-tac-toe boards arranged in a 3×3 meta-board
- 🎯 Your opponent's move determines which board you play next
- 🏆 Win 3 small boards in a row to win the game
- 🧠 Master both local tactics and global strategy

**AI Components:**
- 🧮 Multi-level Minimax with Alpha-Beta Pruning
- 📊 Hierarchical position evaluation (local + meta-board)
- 🎓 Q-Learning with experience replay
- 🔮 Strategic board prioritization
""")

# ============================================================================
# Super Tic-Tac-Toe Environment
# ============================================================================

class SuperTicTacToe:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # 9 small boards, each 3x3
        self.small_boards = [np.zeros((3, 3), dtype=int) for _ in range(9)]
        # Meta-board showing who won each small board (0=ongoing, 1=X, 2=O, -1=draw)
        self.meta_board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.active_board = None  # None means any board is playable
        self.game_over = False
        self.winner = None
        self.move_history = []
        return self.get_state()
    
    def get_state(self):
        """Returns a hashable state representation"""
        small_boards_flat = tuple(tuple(board.flatten()) for board in self.small_boards)
        return (small_boards_flat, tuple(self.meta_board), self.active_board)
    
    def get_available_actions(self):
        """Returns list of (board_idx, row, col) tuples"""
        actions = []
        
        if self.game_over:
            return actions
        
        # Determine which boards we can play in
        if self.active_board is not None and self.meta_board[self.active_board] == 0:
            # Must play in the active board
            boards_to_check = [self.active_board]
        else:
            # Can play in any non-won board
            boards_to_check = [i for i in range(9) if self.meta_board[i] == 0]
        
        # Find empty cells in available boards
        for board_idx in boards_to_check:
            for r in range(3):
                for c in range(3):
                    if self.small_boards[board_idx][r, c] == 0:
                        actions.append((board_idx, r, c))
        
        return actions
    
    def make_move(self, action):
        """Make a move: action = (board_idx, row, col)"""
        if self.game_over:
            return self.get_state(), 0, True
        
        board_idx, row, col = action
        
        # Validate move
        available = self.get_available_actions()
        if action not in available:
            return self.get_state(), -100, True  # Invalid move penalty
        
        # Make the move
        self.small_boards[board_idx][row, col] = self.current_player
        self.move_history.append((action, self.current_player))
        
        # Check if this move won the small board
        if self._check_small_board_win(board_idx, self.current_player):
            self.meta_board[board_idx] = self.current_player
            reward = 10  # Won a small board
        elif self._check_small_board_full(board_idx):
            self.meta_board[board_idx] = -1  # Draw
            reward = 0
        else:
            reward = 0
        
        # Check if won the meta-board (game)
        if self._check_meta_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
            return self.get_state(), 1000, True  # Game won!
        
        # Check if meta-board is full (draw)
        if np.all(self.meta_board != 0):
            self.game_over = True
            self.winner = 0
            return self.get_state(), 0, True
        
        # Set next active board based on where the move was made
        next_board = row * 3 + col
        if self.meta_board[next_board] == 0:
            self.active_board = next_board
        else:
            self.active_board = None  # Next player can play anywhere
        
        # Switch player
        self.current_player = 3 - self.current_player
        
        return self.get_state(), reward, False
    
    def _check_small_board_win(self, board_idx, player):
        """Check if player won the small board"""
        board = self.small_boards[board_idx]
        
        # Check rows and columns
        for i in range(3):
            if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                return True
        
        # Check diagonals
        if board[0, 0] == player and board[1, 1] == player and board[2, 2] == player:
            return True
        if board[0, 2] == player and board[1, 1] == player and board[2, 0] == player:
            return True
        
        return False
    
    def _check_small_board_full(self, board_idx):
        """Check if small board is full"""
        return np.all(self.small_boards[board_idx] != 0)
    
    def _check_meta_win(self, player):
        """Check if player won the meta-board"""
        meta = self.meta_board.reshape(3, 3)
        
        # Check rows and columns
        for i in range(3):
            if np.all(meta[i, :] == player) or np.all(meta[:, i] == player):
                return True
        
        # Check diagonals
        if meta[0, 0] == player and meta[1, 1] == player and meta[2, 2] == player:
            return True
        if meta[0, 2] == player and meta[1, 1] == player and meta[2, 0] == player:
            return True
        
        return False
    
    def evaluate_position(self, player):
        """Advanced heuristic evaluation"""
        if self.winner == player:
            return 100000
        if self.winner == (3 - player):
            return -100000
        if self.game_over:
            return 0
        
        opponent = 3 - player
        score = 0
        
        # Meta-board evaluation (most important)
        meta = self.meta_board.reshape(3, 3)
        
        # Count potential meta-board lines
        score += self._count_meta_lines(player, 2) * 500  # 2-in-a-row on meta
        score += self._count_meta_lines(player, 1) * 100  # 1 on meta
        score -= self._count_meta_lines(opponent, 2) * 600  # Block opponent
        score -= self._count_meta_lines(opponent, 1) * 100
        
        # Strategic board positions (center and corners are valuable)
        strategic_boards = [4]  # Center
        corner_boards = [0, 2, 6, 8]
        
        for b in strategic_boards:
            if self.meta_board[b] == player:
                score += 200
            elif self.meta_board[b] == opponent:
                score -= 200
        
        for b in corner_boards:
            if self.meta_board[b] == player:
                score += 100
            elif self.meta_board[b] == opponent:
                score -= 100
        
        # Local board evaluation (less important than meta)
        for board_idx in range(9):
            if self.meta_board[board_idx] == 0:  # Only evaluate ongoing boards
                board_score = self._evaluate_small_board(board_idx, player)
                score += board_score * 0.5  # Weighted less than meta-board
        
        # Active board bonus (controlling where opponent plays next is valuable)
        if self.active_board is not None:
            if self.meta_board[self.active_board] == 0:
                # If we're forcing opponent into a specific board, that's strategic
                score += 50
        
        return score
    
    def _count_meta_lines(self, player, count):
        """Count potential lines on meta-board with 'count' player pieces"""
        meta = self.meta_board.reshape(3, 3)
        lines = 0
        
        # Rows and columns
        for i in range(3):
            row = meta[i, :]
            col = meta[:, i]
            if np.sum(row == player) == count and np.sum(row == (3-player)) == 0:
                lines += 1
            if np.sum(col == player) == count and np.sum(col == (3-player)) == 0:
                lines += 1
        
        # Diagonals
        diag1 = [meta[0, 0], meta[1, 1], meta[2, 2]]
        diag2 = [meta[0, 2], meta[1, 1], meta[2, 0]]
        
        if diag1.count(player) == count and (3-player) not in diag1:
            lines += 1
        if diag2.count(player) == count and (3-player) not in diag2:
            lines += 1
        
        return lines
    
    def _evaluate_small_board(self, board_idx, player):
        """Evaluate a single small board"""
        board = self.small_boards[board_idx]
        opponent = 3 - player
        score = 0
        
        # Count potential lines
        lines_2 = 0  # 2-in-a-row
        lines_1 = 0  # 1 piece
        opp_lines_2 = 0
        
        # Check all lines (rows, cols, diagonals)
        for i in range(3):
            row = board[i, :]
            col = board[:, i]
            
            if np.sum(row == player) == 2 and np.sum(row == opponent) == 0:
                lines_2 += 1
            elif np.sum(row == player) == 1 and np.sum(row == opponent) == 0:
                lines_1 += 1
            
            if np.sum(row == opponent) == 2 and np.sum(row == player) == 0:
                opp_lines_2 += 1
            
            if np.sum(col == player) == 2 and np.sum(col == opponent) == 0:
                lines_2 += 1
            elif np.sum(col == player) == 1 and np.sum(col == opponent) == 0:
                lines_1 += 1
            
            if np.sum(col == opponent) == 2 and np.sum(col == player) == 0:
                opp_lines_2 += 1
        
        # Diagonals
        diag1 = [board[0, 0], board[1, 1], board[2, 2]]
        diag2 = [board[0, 2], board[1, 1], board[2, 0]]
        
        if diag1.count(player) == 2 and opponent not in diag1:
            lines_2 += 1
        if diag2.count(player) == 2 and opponent not in diag2:
            lines_2 += 1
        if diag1.count(opponent) == 2 and player not in diag1:
            opp_lines_2 += 1
        if diag2.count(opponent) == 2 and player not in diag2:
            opp_lines_2 += 1
        
        score = lines_2 * 10 + lines_1 * 2 - opp_lines_2 * 12
        
        return score

# ============================================================================
# Strategic RL Agent
# ============================================================================

class SuperTTTAgent:
    def __init__(self, player_id, lr=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.9995, epsilon_min=0.05):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = {}
        self.experience_replay = deque(maxlen=50000)
        self.minimax_depth = 2  # Start shallow due to complexity
        
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, env, training=True):
        available = env.get_available_actions()
        if not available:
            return None
        
        # LEVEL 1: Immediate Tactics
        # Check for instant small board win
        for action in available:
            sim = self._simulate_move(env, action, self.player_id)
            board_idx = action[0]
            if sim.meta_board[board_idx] == self.player_id and env.meta_board[board_idx] == 0:
                # Check if this wins the game
                if sim._check_meta_win(self.player_id):
                    return action
        
        # Check for instant meta-board win (must win a specific small board)
        for action in available:
            sim = self._simulate_move(env, action, self.player_id)
            if sim.winner == self.player_id:
                return action
        
        # Block opponent from winning
        opponent = 3 - self.player_id
        for action in available:
            sim = self._simulate_move(env, action, opponent)
            if sim.winner == opponent:
                return action
        
        # LEVEL 2: Strategic Planning
        if training and random.random() < self.epsilon:
            # Prioritize strategic boards when exploring
            strategic_actions = [a for a in available if a[0] in [4, 0, 2, 6, 8]]
            if strategic_actions:
                return random.choice(strategic_actions)
            return random.choice(available)
        
        # Minimax with limited depth
        best_score = -float('inf')
        best_actions = []
        
        alpha = -float('inf')
        beta = float('inf')
        
        for action in available:
            sim = self._simulate_move(env, action, self.player_id)
            score = self._minimax(sim, self.minimax_depth - 1, alpha, beta, False)
            
            # Q-learning boost
            q_boost = self.get_q_value(env.get_state(), action) * 0.05
            total_score = score + q_boost
            
            if total_score > best_score:
                best_score = total_score
                best_actions = [action]
            elif abs(total_score - best_score) < 0.01:
                best_actions.append(action)
            
            alpha = max(alpha, best_score)
        
        return random.choice(best_actions) if best_actions else random.choice(available)
    
    def _minimax(self, env, depth, alpha, beta, is_maximizing):
        if env.winner == self.player_id:
            return 10000 + depth
        if env.winner == (3 - self.player_id):
            return -10000 - depth
        if env.game_over:
            return 0
        if depth == 0:
            return env.evaluate_position(self.player_id)
        
        available = env.get_available_actions()
        
        if is_maximizing:
            max_eval = -float('inf')
            for action in available:
                sim = self._simulate_move(env, action, self.player_id)
                eval_score = self._minimax(sim, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - self.player_id
            for action in available:
                sim = self._simulate_move(env, action, opponent)
                eval_score = self._minimax(sim, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _simulate_move(self, env, action, player):
        sim = SuperTicTacToe()
        sim.small_boards = [board.copy() for board in env.small_boards]
        sim.meta_board = env.meta_board.copy()
        sim.current_player = player
        sim.active_board = env.active_board
        sim.make_move(action)
        return sim
    
    def update_q_value(self, state, action, reward, next_state, next_actions):
        current_q = self.get_q_value(state, action)
        if next_actions:
            max_next_q = max([self.get_q_value(next_state, a) for a in next_actions])
        else:
            max_next_q = 0
        
        td_error = reward + self.gamma * max_next_q - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[(state, action)] = new_q
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0

# ============================================================================
# Training System
# ============================================================================

def play_game(env, agent1, agent2, training=True):
    env.reset()
    game_history = []
    agents = {1: agent1, 2: agent2}
    
    while not env.game_over:
        current_player = env.current_player
        current_agent = agents[current_player]
        
        state = env.get_state()
        action = current_agent.choose_action(env, training)
        
        if action is None:
            break
        
        game_history.append((state, action, current_player))
        next_state, reward, done = env.make_move(action)
        
        if training:
            next_actions = env.get_available_actions()
            current_agent.update_q_value(state, action, reward, next_state, next_actions)
        
        if done:
            if env.winner == 1:
                agent1.wins += 1
                agent2.losses += 1
                if training:
                    _update_outcome(agent1, game_history, 1, 100)
                    _update_outcome(agent2, game_history, 2, -50)
            elif env.winner == 2:
                agent2.wins += 1
                agent1.losses += 1
                if training:
                    _update_outcome(agent1, game_history, 1, -50)
                    _update_outcome(agent2, game_history, 2, 100)
            else:
                agent1.draws += 1
                agent2.draws += 1
                if training:
                    _update_outcome(agent1, game_history, 1, -10)
                    _update_outcome(agent2, game_history, 2, -10)
    
    return env.winner

def _update_outcome(agent, history, player_id, final_reward):
    agent_moves = [(s, a) for s, a, p in history if p == player_id]
    for i in range(len(agent_moves) - 1, -1, -1):
        state, action = agent_moves[i]
        discount = agent.gamma ** (len(agent_moves) - 1 - i)
        adjusted_reward = final_reward * discount
        current_q = agent.get_q_value(state, action)
        new_q = current_q + agent.lr * (adjusted_reward - current_q)
        agent.q_table[(state, action)] = new_q

# ============================================================================
# Visualization
# ============================================================================

def visualize_super_board(env, title="Super Tic-Tac-Toe"):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw meta-board grid (thick lines)
    for i in range(4):
        ax.plot([i*3, i*3], [0, 9], 'k-', linewidth=4)
        ax.plot([0, 9], [i*3, i*3], 'k-', linewidth=4)
    
    # Draw small board grids (thin lines)
    for i in range(10):
        if i % 3 != 0:
            ax.plot([i, i], [0, 9], 'gray', linewidth=1)
            ax.plot([0, 9], [i, i], 'gray', linewidth=1)
    
    # Draw pieces in each small board
    for board_idx in range(9):
        meta_row = board_idx // 3
        meta_col = board_idx % 3
        offset_row = meta_row * 3
        offset_col = meta_col * 3
        
        board = env.small_boards[board_idx]
        
        # If board is won, draw large marker
        if env.meta_board[board_idx] == 1:
            # Large X
            center_x = offset_col + 1.5
            center_y = 9 - (offset_row + 1.5)
            ax.plot([center_x - 1, center_x + 1], [center_y - 1, center_y + 1], 
                   'b-', linewidth=6, alpha=0.3)
            ax.plot([center_x - 1, center_x + 1], [center_y + 1, center_y - 1], 
                   'b-', linewidth=6, alpha=0.3)
        elif env.meta_board[board_idx] == 2:
            # Large O
            center_x = offset_col + 1.5
            center_y = 9 - (offset_row + 1.5)
            circle = plt.Circle((center_x, center_y), 1, 
                              color='r', fill=False, linewidth=6, alpha=0.3)
            ax.add_patch(circle)
        elif env.meta_board[board_idx] == -1:
            # Draw indicator
            rect = Rectangle((offset_col, 9 - offset_row - 3), 3, 3,
                           fill=True, color='gray', alpha=0.2)
            ax.add_patch(rect)
        
        # Draw individual pieces
        for r in range(3):
            for c in range(3):
                x = offset_col + c + 0.5
                y = 9 - (offset_row + r + 0.5)
                
                if board[r, c] == 1:
                    ax.plot([x - 0.3, x + 0.3], [y - 0.3, y + 0.3], 'b-', linewidth=3)
                    ax.plot([x - 0.3, x + 0.3], [y + 0.3, y - 0.3], 'b-', linewidth=3)
                elif board[r, c] == 2:
                    circle = plt.Circle((x, y), 0.3, color='r', fill=False, linewidth=3)
                    ax.add_patch(circle)
    
    # Highlight active board
    if env.active_board is not None:
        meta_row = env.active_board // 3
        meta_col = env.active_board % 3
        rect = Rectangle((meta_col * 3, 9 - (meta_row + 1) * 3), 3, 3,
                         fill=False, edgecolor='yellow', linewidth=3, linestyle='--')
        ax.add_patch(rect)
    
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    return fig

# ============================================================================
# Save/Load
# ============================================================================

def serialize_q_table(q_table):
    serialized = {}
    for (state, action), value in q_table.items():
        # Convert state and action to JSON-serializable format
        state_str = json.dumps([
            [list(map(int, board)) for board in state[0]],
            list(map(int, state[1])),
            state[2]
        ])
        action_str = json.dumps(list(map(int, action)))
        key = f"{state_str}||{action_str}"
        serialized[key] = float(value)
    return serialized

def deserialize_q_table(serialized):
    q_table = {}
    for key, value in serialized.items():
        state_str, action_str = key.split("||")
        state_data = json.loads(state_str)
        
        # Reconstruct state tuple
        small_boards = tuple(tuple(board) for board in state_data[0])
        meta_board = tuple(state_data[1])
        active_board = state_data[2]
        state = (small_boards, meta_board, active_board)
        
        action = tuple(json.loads(action_str))
        q_table[(state, action)] = value
    
    return q_table

def create_zip(agent1, agent2, config):
    agent1_data = {
        "q_table": serialize_q_table(agent1.q_table),
        "epsilon": agent1.epsilon,
        "lr": agent1.lr,
        "gamma": agent1.gamma,
        "minimax_depth": agent1.minimax_depth,
        "wins": agent1.wins,
        "losses": agent1.losses,
        "draws": agent1.draws
    }
    
    agent2_data = {
        "q_table": serialize_q_table(agent2.q_table),
        "epsilon": agent2.epsilon,
        "lr": agent2.lr,
        "gamma": agent2.gamma,
        "minimax_depth": agent2.minimax_depth,
        "wins": agent2.wins,
        "losses": agent2.losses,
        "draws": agent2.draws
    }
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent1.json", json.dumps(agent1_data, indent=2))
        zf.writestr("agent2.json", json.dumps(agent2_data, indent=2))
        zf.writestr("config.json", json.dumps(config, indent=2))
    
    buffer.seek(0)
    return buffer

def load_from_zip(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            agent1_data = json.loads(zf.read("agent1.json"))
            agent2_data = json.loads(zf.read("agent2.json"))
            config = json.loads(zf.read("config.json"))
            
            agent1 = SuperTTTAgent(1, agent1_data['lr'], agent1_data['gamma'])
            agent1.q_table = deserialize_q_table(agent1_data['q_table'])
            agent1.epsilon = agent1_data['epsilon']
            agent1.minimax_depth = agent1_data.get('minimax_depth', 2)
            agent1.wins = agent1_data.get('wins', 0)
            agent1.losses = agent1_data.get('losses', 0)
            agent1.draws = agent1_data.get('draws', 0)
            
            agent2 = SuperTTTAgent(2, agent2_data['lr'], agent2_data['gamma'])
            agent2.q_table = deserialize_q_table(agent2_data['q_table'])
            agent2.epsilon = agent2_data['epsilon']
            agent2.minimax_depth = agent2_data.get('minimax_depth', 2)
            agent2.wins = agent2_data.get('wins', 0)
            agent2.losses = agent2_data.get('losses', 0)
            agent2.draws = agent2_data.get('draws', 0)
            
            return agent1, agent2, config
    except Exception as e:
        st.error(f"Load failed: {e}")
        return None, None, None

# ============================================================================
# UI
# ============================================================================

st.sidebar.header("🎮 Controls")

with st.sidebar.expander("1️⃣ Agent Configuration", expanded=True):
    if 'lr1' not in st.session_state: st.session_state.lr1 = 0.1
    if 'gamma1' not in st.session_state: st.session_state.gamma1 = 0.95
    if 'minimax1' not in st.session_state: st.session_state.minimax1 = 2
    
    lr1 = st.slider("Learning Rate α₁", 0.01, 0.5, 0.1, 0.01)
    gamma1 = st.slider("Discount γ₁", 0.8, 0.99, 0.95, 0.01)
    minimax1 = st.slider("Minimax Depth₁", 1, 4, 2)

with st.sidebar.expander("2️⃣ Opponent Configuration", expanded=True):
    if 'lr2' not in st.session_state: st.session_state.lr2 = 0.1
    if 'gamma2' not in st.session_state: st.session_state.gamma2 = 0.95
    if 'minimax2' not in st.session_state: st.session_state.minimax2 = 2
    
    lr2 = st.slider("Learning Rate α₂", 0.01, 0.5, 0.1, 0.01)
    gamma2 = st.slider("Discount γ₂", 0.8, 0.99, 0.95, 0.01)
    minimax2 = st.slider("Minimax Depth₂", 1, 4, 2)

with st.sidebar.expander("3️⃣ Training Setup", expanded=True):
    episodes = st.number_input("Episodes", 100, 100000, 5000, 100)
    update_freq = st.number_input("Update Every N Games", 10, 1000, 100, 10)

with st.sidebar.expander("4️⃣ Storage", expanded=False):
    if 'agent1' in st.session_state and st.session_state.agent1 is not None:
        config = {
            "lr1": lr1, "gamma1": gamma1, "minimax1": minimax1,
            "lr2": lr2, "gamma2": gamma2, "minimax2": minimax2,
            "training_history": st.session_state.get('history', None)
        }
        
        zip_data = create_zip(st.session_state.agent1, st.session_state.agent2, config)
        st.download_button(
            "💾 Download Trained Agents",
            data=zip_data,
            file_name="super_ttt_agents.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.info("Train agents first")
    
    uploaded = st.file_uploader("Upload Agents (.zip)", type="zip")
    if uploaded and st.button("📥 Load Agents", use_container_width=True):
        a1, a2, cfg = load_from_zip(uploaded)
        if a1:
            st.session_state.agent1 = a1
            st.session_state.agent2 = a2
            st.session_state.history = cfg.get('training_history')
            st.toast("Agents loaded!", icon="💾")
            st.rerun()

train_btn = st.sidebar.button("🚀 Start Training", use_container_width=True, type="primary")

if st.sidebar.button("🧹 Reset All", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Initialize
if 'env' not in st.session_state:
    st.session_state.env = SuperTicTacToe()

if 'agent1' not in st.session_state:
    st.session_state.agent1 = SuperTTTAgent(1, lr1, gamma1)
    st.session_state.agent1.minimax_depth = minimax1
    st.session_state.agent2 = SuperTTTAgent(2, lr2, gamma2)
    st.session_state.agent2.minimax_depth = minimax2

agent1 = st.session_state.agent1
agent2 = st.session_state.agent2
agent1.minimax_depth = minimax1
agent2.minimax_depth = minimax2

# Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔵 Agent 1", f"Q-States: {len(agent1.q_table):,}", f"ε={agent1.epsilon:.4f}")
    st.metric("Wins", agent1.wins)
with col2:
    st.metric("🔴 Agent 2", f"Q-States: {len(agent2.q_table):,}", f"ε={agent2.epsilon:.4f}")
    st.metric("Wins", agent2.wins)
with col3:
    total = agent1.wins + agent1.losses + agent1.draws
    st.metric("Total Games", total)
    st.metric("Draws", agent1.draws)

st.markdown("---")

# Training
if train_btn:
    st.subheader("🎯 Training in Progress")
    
    agent1.reset_stats()
    agent2.reset_stats()
    
    progress = st.progress(0)
    status = st.empty()
    
    history = {
        'agent1_wins': [], 'agent2_wins': [], 'draws': [],
        'agent1_epsilon': [], 'agent2_epsilon': [],
        'agent1_q': [], 'agent2_q': [], 'episode': []
    }
    
    for ep in range(1, episodes + 1):
        play_game(st.session_state.env, agent1, agent2, training=True)
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if ep % update_freq == 0:
            history['agent1_wins'].append(agent1.wins)
            history['agent2_wins'].append(agent2.wins)
            history['draws'].append(agent1.draws)
            history['agent1_epsilon'].append(agent1.epsilon)
            history['agent2_epsilon'].append(agent2.epsilon)
            history['agent1_q'].append(len(agent1.q_table))
            history['agent2_q'].append(len(agent2.q_table))
            history['episode'].append(ep)
            
            progress.progress(ep / episodes)
            status.markdown(f"""
            | Metric | Agent 1 | Agent 2 |
            |:-------|:-------:|:-------:|
            | Wins | {agent1.wins} | {agent2.wins} |
            | ε | {agent1.epsilon:.4f} | {agent2.epsilon:.4f} |
            | Q-States | {len(agent1.q_table):,} | {len(agent2.q_table):,} |
            
            **Episode {ep}/{episodes}** ({ep/episodes*100:.1f}%) | Draws: {agent1.draws}
            """)
    
    progress.progress(1.0)
    st.toast("Training complete!", icon="🎉")
    st.session_state.history = history
    st.session_state.agent1 = agent1
    st.session_state.agent2 = agent2

# Charts
if 'history' in st.session_state and st.session_state.history:
    st.subheader("📊 Training Analytics")
    hist = st.session_state.history
    df = pd.DataFrame(hist)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### Performance")
        st.line_chart(df.set_index('episode')[['agent1_wins', 'agent2_wins', 'draws']])
    with c2:
        st.write("#### Exploration")
        st.line_chart(df.set_index('episode')[['agent1_epsilon', 'agent2_epsilon']])
    
    st.write("#### Knowledge Growth")
    st.line_chart(df.set_index('episode')[['agent1_q', 'agent2_q']])

# Demo battle
if 'agent1' in st.session_state and len(agent1.q_table) > 0:
    st.subheader("⚔️ Demo Battle")
    
    if st.button("🎬 Watch Trained Agents Battle!", use_container_width=True):
        demo_env = SuperTicTacToe()
        board_display = st.empty()
        agents = {1: agent1, 2: agent2}
        
        with st.spinner("Battle in progress..."):
            while not demo_env.game_over:
                action = agents[demo_env.current_player].choose_action(demo_env, False)
                if action is None:
                    break
                demo_env.make_move(action)
                
                fig = visualize_super_board(demo_env, 
                    f"Player {demo_env.current_player}'s Turn")
                board_display.pyplot(fig)
                plt.close(fig)
                time.sleep(0.8)
        
        if demo_env.winner == 1:
            st.success("🏆 Agent 1 (Blue) wins!")
        elif demo_env.winner == 2:
            st.error("🏆 Agent 2 (Red) wins!")
        else:
            st.warning("🤝 Draw!")

# Human vs AI
st.markdown("---")
st.header("🎮 Human vs AI")

if len(agent1.q_table) > 0:
    with st.container():
        h1, h2, h3 = st.columns([1, 1, 1])
        with h1:
            opponent = st.selectbox("Opponent", ["Agent 1 (Blue)", "Agent 2 (Red)"])
        with h2:
            starter = st.selectbox("First Move", ["Human", "AI"])
        with h3:
            st.write("")
            if st.button("🎯 New Game", use_container_width=True, type="primary"):
                st.session_state.human_env = SuperTicTacToe()
                st.session_state.game_active = True
                
                if "Agent 1" in opponent:
                    st.session_state.ai_agent = agent1
                    st.session_state.ai_id = 1
                    st.session_state.human_id = 2
                else:
                    st.session_state.ai_agent = agent2
                    st.session_state.ai_id = 2
                    st.session_state.human_id = 1
                
                if starter == "AI":
                    st.session_state.human_env.current_player = st.session_state.ai_id
                
                st.rerun()
    
    if 'human_env' in st.session_state and st.session_state.game_active:
        henv = st.session_state.human_env
        
        # AI turn
        if henv.current_player == st.session_state.ai_id and not henv.game_over:
            with st.spinner("🤖 AI thinking..."):
                time.sleep(0.7)
                action = st.session_state.ai_agent.choose_action(henv, False)
                if action:
                    henv.make_move(action)
                    st.rerun()
        
        # Status
        if henv.game_over:
            if henv.winner == st.session_state.human_id:
                st.success("🎉 YOU WIN!")
            elif henv.winner == st.session_state.ai_id:
                st.error("💀 AI WINS!")
            else:
                st.warning("🤝 DRAW!")
        else:
            turn = "Your Turn" if henv.current_player == st.session_state.human_id else "AI's Turn"
            st.caption(f"**{turn}**")
            if henv.active_board is not None:
                st.info(f"You must play in board {henv.active_board + 1}")
        
        # Board display
        fig = visualize_super_board(henv, "Super Tic-Tac-Toe")
        st.pyplot(fig)
        plt.close(fig)
        
        # Move selection
        if not henv.game_over and henv.current_player == st.session_state.human_id:
            st.write("**Select your move:**")
            available = henv.get_available_actions()
            
            move_cols = st.columns(3)
            for idx, (board_idx, row, col) in enumerate(available[:15]):  # Show first 15
                col_idx = idx % 3
                if move_cols[col_idx].button(
                    f"Board {board_idx+1}, Cell ({row+1},{col+1})",
                    key=f"move_{board_idx}_{row}_{col}",
                    use_container_width=True
                ):
                    henv.make_move((board_idx, row, col))
                    st.rerun()
else:
    st.info("Train or load agents to play!")
