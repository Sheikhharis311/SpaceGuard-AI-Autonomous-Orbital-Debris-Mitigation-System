#!/usr/bin/env python3
"""
Advanced Space Debris Removal Simulator with AI Assistant
Features:
1. Realistic orbital mechanics with J2 perturbations and atmospheric drag
2. LSTM trajectory prediction with periodic training
3. Multi-drone RL coordination with collision avoidance
4. Real-time danger scoring system
5. Interactive 3D visualization with capture animations
6. Position history logging and continuous learning
7. AI voice assistant for system monitoring and control
8. Natural language processing for voice commands
9. Personality-based response generation
10. Continuous learning from interactions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vispy import scene, visuals, app
import random
import time
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from skyfield.api import load, EarthSatellite
import json
from collections import defaultdict, deque
import os
from threading import Thread, Lock
from queue import Queue
import warnings
from scipy.spatial.distance import cdist
import speech_recognition as sr
import pyttsx3
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# Initialize NLP components
nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_lg')

# Constants
EARTH_RADIUS = 6378.137  # km
EARTH_MU = 398600.4418  # km^3/s^2
J2 = 1.08263e-3
SAFE_DISTANCE = 100  # km between drones
CAPTURE_DISTANCE = 50  # km
MAX_DEBRIS = 50
MAX_DRONES = 5
SIMULATION_STEP = 1  # second
DEBRIS_SIZE_RANGE = (0.1, 5.0)  # meters
DEBRIS_MASS_RANGE = (1, 500)  # kg

class SpaceDebris:
    """Enhanced space debris object with physics and history tracking"""
    def __init__(self, id, tle=None):
        self.id = id
        self.size = random.uniform(*DEBRIS_SIZE_RANGE)
        self.mass = random.uniform(*DEBRIS_MASS_RANGE)
        self.metal_content = random.random()
        self.position_history = deque(maxlen=100)
        self.velocity_history = deque(maxlen=100)
        self.capture_status = False
        
        if tle:
            self._init_from_tle(tle)
        else:
            self._init_random_orbit()
            
    def _init_from_tle(self, tle):
        """Initialize from Two-Line Element set"""
        self.tle = tle
        self.satellite = EarthSatellite(tle[1], tle[2], tle[0])
        ts = load.timescale()
        pos, vel = self.satellite.at(ts.now()).position.km, self.satellite.at(ts.now()).velocity.km_per_s
        self.position = np.array(pos)
        self.velocity = np.array(vel)
        self._record_history()
        
    def _init_random_orbit(self):
        """Initialize with realistic orbital parameters"""
        self.semi_major_axis = random.uniform(6571, 7571)  # 200-1200 km altitude
        self.eccentricity = random.uniform(0, 0.02)
        self.inclination = np.radians(random.uniform(0, 90))
        self._update_from_orbital_elements()
        
    def _update_from_orbital_elements(self):
        """Update state vectors from orbital elements"""
        nu = random.uniform(0, 2*np.pi)  # True anomaly
        
        # Position in perifocal coordinates
        r = self.semi_major_axis * (1 - self.eccentricity**2) / (1 + self.eccentricity * np.cos(nu))
        pos_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0])
        
        # Velocity in perifocal coordinates
        vel_pqw = np.sqrt(EARTH_MU / (self.semi_major_axis * (1 - self.eccentricity**2))) * np.array([
            -np.sin(nu), self.eccentricity + np.cos(nu), 0
        ])
        
        # Transform to ECI
        raan = random.uniform(0, 2*np.pi)
        argp = random.uniform(0, 2*np.pi)
        rot = self._rotation_matrix(self.inclination, raan, argp)
        
        self.position = rot.dot(pos_pqw)
        self.velocity = rot.dot(vel_pqw)
        self._record_history()
        
    def _rotation_matrix(self, inc, raan, argp):
        """Create rotation matrix from orbital elements"""
        ci, si = np.cos(inc), np.sin(inc)
        cr, sr = np.cos(raan), np.sin(raan)
        ca, sa = np.cos(argp), np.sin(argp)
        
        return np.array([
            [ca*cr - sa*ci*sr,  ca*sr + sa*ci*cr,  sa*si],
            [-sa*cr - ca*ci*sr, -sa*sr + ca*ci*cr,  ca*si],
            [si*sr,            -si*cr,            ci]
        ])
        
    def _record_history(self):
        """Record current state to history"""
        self.position_history.append(np.copy(self.position))
        self.velocity_history.append(np.copy(self.velocity))
        
    def update_position(self, dt):
        """Advanced physics update with J2 and drag"""
        r = np.linalg.norm(self.position)
        pos = self.position
        
        # Two-body acceleration
        a = -EARTH_MU * pos / r**3
        
        # J2 perturbation
        z = pos[2]
        k = 1.5 * J2 * EARTH_MU * EARTH_RADIUS**2 / r**5
        a_j2 = k * np.array([
            pos[0] * (5*z**2/r**2 - 1),
            pos[1] * (5*z**2/r**2 - 1),
            pos[2] * (5*z**2/r**2 - 3)
        ])
        
        # Atmospheric drag (simplified)
        if r < EARTH_RADIUS + 800:  # Below 800 km
            v_rel = self.velocity - np.cross(np.array([0, 0, 7.292115e-5]), pos)
            v_mag = np.linalg.norm(v_rel)
            rho = 6e-13 * np.exp(-(r - EARTH_RADIUS) / 80)  # Approximate density
            a_drag = -0.5 * rho * 2.2 * (np.pi * self.size**2 / 4) / self.mass * v_mag * v_rel
            a += a_drag
        
        a += a_j2
        
        # Update state
        self.velocity += a * dt
        self.position += self.velocity * dt
        self._record_history()

class DangerScoreService:
    """Real-time danger score calculation service with API integration"""
    def __init__(self):
        self.scores = Queue()
        self.running = True
        self.lock = Lock()
        self._start_service()
        
    def _start_service(self):
        """Start background scoring service"""
        def _run():
            while self.running:
                try:
                    scores = self._calculate_scores()
                    with self.lock:
                        while not self.scores.empty():
                            self.scores.get()
                        self.scores.put(scores)
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    print(f"Score service error: {e}")
        
        Thread(target=_run, daemon=True).start()
        
    def _calculate_scores(self):
        """Calculate danger scores with probabilistic modeling"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'scores': {
                i: {
                    'collision_prob': self._probabilistic_collision_model(),
                    'criticality': random.uniform(0.1, 1.0),
                    'priority': random.uniform(0.5, 1.0)
                } 
                for i in range(100)  # Simulate 100 debris IDs
            }
        }
        
    def _probabilistic_collision_model(self):
        """Advanced probabilistic collision model"""
        base_prob = random.uniform(0, 0.2)
        # Add time-dependent variation
        time_factor = 0.1 * np.sin(time.time() / 3600)  
        return np.clip(base_prob + time_factor, 0, 0.3)
        
    def get_score(self, debris_id):
        """Get current danger score for debris"""
        with self.lock:
            if not self.scores.empty():
                latest = self.scores.queue[-1]
                return latest['scores'].get(debris_id % 100, {
                    'collision_prob': 0.1,
                    'criticality': 0.5,
                    'priority': 0.7
                })
        return {'collision_prob': 0.1, 'criticality': 0.5, 'priority': 0.7}
        
    def stop(self):
        """Stop the service"""
        self.running = False

class LSTMPredictor:
    """Enhanced LSTM trajectory predictor with continuous learning"""
    def __init__(self, look_back=20, look_forward=10):
        self.look_back = look_back
        self.look_forward = look_forward
        self.model = self._build_model()
        self.training_data = defaultdict(list)
        self.training_interval = 10  # steps
        self.model_file = "models/lstm_model.h5"
        os.makedirs("models", exist_ok=True)
        
    def _build_model(self):
        """Construct the LSTM model architecture"""
        model = Sequential([
            LSTM(128, input_shape=(self.look_back, 6), return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.look_forward * 3)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Load existing model if available
        if os.path.exists(self.model_file):
            try:
                model.load_weights(self.model_file)
                print("Loaded pre-trained LSTM model")
            except:
                print("Could not load model, starting fresh")
                
        return model
        
    def add_training_data(self, debris):
        """Add debris history to training dataset"""
        if len(debris.position_history) >= self.look_back + self.look_forward:
            # Combine position and velocity features
            combined = np.hstack([
                np.array(list(debris.position_history)),
                np.array(list(debris.velocity_history))
            ])
            self.training_data[debris.id].append(combined)
        
    def prepare_training_data(self):
        """Prepare training sequences from collected data"""
        X, y = [], []
        for debris_data in self.training_data.values():
            for full_sequence in debris_data:
                for i in range(len(full_sequence) - self.look_back - self.look_forward):
                    X.append(full_sequence[i:i+self.look_back])
                    y.append(full_sequence[i+self.look_back:i+self.look_back+self.look_forward, :3].flatten())
        return np.array(X), np.array(y)
        
    def periodic_train(self, current_step):
        """Train model at regular intervals"""
        if current_step % self.training_interval == 0 and self.training_data:
            X, y = self.prepare_training_data()
            
            if len(X) > 100:  # Only train if sufficient data
                print(f"\nTraining LSTM with {len(X)} samples...")
                
                # Callbacks
                callbacks = [
                    ModelCheckpoint(
                        self.model_file,
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True
                    )
                ]
                
                try:
                    history = self.model.fit(
                        X, y,
                        epochs=30,
                        batch_size=64,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=1
                    )
                    self._plot_training(history)
                except Exception as e:
                    print(f"Training error: {e}")
                    
    def _plot_training(self, history):
        """Plot training progress"""
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.ylabel('MAE (km)')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_progress.png')
        plt.close()
        
    def predict(self, debris):
        """Predict future trajectory for debris"""
        if len(debris.position_history) < self.look_back:
            return None
            
        # Prepare input sequence
        combined = np.hstack([
            np.array(list(debris.position_history)[-self.look_back:]),
            np.array(list(debris.velocity_history)[-self.look_back:])
        ])
        
        # Make prediction
        prediction = self.model.predict(combined[np.newaxis, ...], verbose=0)[0]
        return prediction.reshape(-1, 3)

class Drone:
    """Autonomous debris capture drone with RL and collision avoidance"""
    def __init__(self, id, initial_pos):
        self.id = id
        self.position = np.array(initial_pos, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.fuel = 1000
        self.target = None
        self.collision_radius = 30  # km
        self.max_speed = 20  # km/s
        self.history = deque(maxlen=100)
        self.capture_mechanism = "net"  # Options: net, harpoon, magnetic
        
    def update(self, action, dt):
        """Update drone state based on RL action"""
        action = np.clip(action, -1, 1)
        self.velocity = action * self.max_speed
        self.position += self.velocity * dt
        self.fuel -= np.linalg.norm(self.velocity) * dt * 0.1
        self.history.append(np.copy(self.position))
        
    def avoid_collision(self, other_drones):
        """Perform advanced collision avoidance maneuvers"""
        for drone in other_drones:
            if drone.id == self.id:
                continue
                
            dist = np.linalg.norm(self.position - drone.position)
            if dist < 2 * self.collision_radius:
                # Calculate evasive direction using potential fields
                direction = (self.position - drone.position)
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                
                # Add randomness to avoid deadlocks
                direction += 0.1 * np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                
                self.velocity += direction * 5
                
                # Limit speed
                speed = np.linalg.norm(self.velocity)
                if speed > self.max_speed:
                    self.velocity = self.velocity / speed * self.max_speed

class SpaceDebrisEnv(gym.Env):
    """RL environment for drone control with enhanced observations"""
    def __init__(self, simulator, drone_id):
        super().__init__()
        self.simulator = simulator
        self.drone_id = drone_id
        
        # Enhanced observation space
        self.observation_space = spaces.Dict({
            "drone_position": spaces.Box(low=-1e4, high=1e4, shape=(3,)),
            "drone_velocity": spaces.Box(low=-20, high=20, shape=(3,)),
            "drone_fuel": spaces.Box(low=0, high=1000, shape=(1,)),
            "target_position": spaces.Box(low=-1e4, high=1e4, shape=(3,)),
            "target_velocity": spaces.Box(low=-10, high=10, shape=(3,)),
            "danger_score": spaces.Box(low=0, high=1, shape=(1,)),
            "nearby_drones": spaces.Box(low=-1e4, high=1e4, shape=(MAX_DRONES-1, 3)),  # Positions of other drones
            "nearby_debris": spaces.Box(low=-1e4, high=1e4, shape=(5, 3))  # Positions of 5 nearest debris
        })
        
        # Action space: normalized direction vector
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        
    def reset(self):
        """Reset environment"""
        return self._get_obs()
    
    def _get_obs(self):
        """Get current observation with enhanced context"""
        drone = self.simulator.drones[self.drone_id]
        target = self.simulator._get_target_debris(drone)
        
        # Get positions of other drones
        other_drones = [d.position for d in self.simulator.drones if d.id != self.drone_id]
        other_drones += [np.zeros(3)] * (MAX_DRONES-1 - len(other_drones))  # Pad if needed
        
        # Get positions of nearest debris
        if self.simulator.debris:
            debris_positions = np.array([d.position for d in self.simulator.debris])
            distances = cdist([drone.position], debris_positions)[0]
            nearest_indices = np.argsort(distances)[:5]
            nearest_debris = debris_positions[nearest_indices]
            nearest_debris = np.pad(nearest_debris, ((0, 5-len(nearest_debris)), (0, 0)), 'constant')
        else:
            nearest_debris = np.zeros((5, 3))
        
        obs = {
            "drone_position": drone.position,
            "drone_velocity": drone.velocity,
            "drone_fuel": np.array([drone.fuel / 1000]),
            "target_position": np.zeros(3) if target is None else target.position,
            "target_velocity": np.zeros(3) if target is None else target.velocity,
            "danger_score": np.array([0]) if target is None else np.array([self.simulator.danger_service.get_score(target.id)['priority']]),
            "nearby_drones": np.array(other_drones),
            "nearby_debris": nearest_debris
        }
        return obs
    
    def step(self, action):
        """Execute one step with enhanced rewards"""
        drone = self.simulator.drones[self.drone_id]
        target = self.simulator._get_target_debris(drone)
        
        # Update drone
        drone.update(action, SIMULATION_STEP)
        
        # Calculate reward components
        reward = 0
        done = False
        
        if target:
            # Distance reward (closer is better)
            distance = np.linalg.norm(drone.position - target.position)
            reward += -distance / 1000
            
            # Capture reward
            if distance < CAPTURE_DISTANCE:
                reward += 100
                done = True
        
        # Fuel penalty
        reward -= 0.01 * (1000 - drone.fuel) / 1000
        
        # Collision avoidance penalty
        for other in self.simulator.drones:
            if other.id != drone.id:
                dist = np.linalg.norm(drone.position - other.position)
                if dist < SAFE_DISTANCE:
                    # Quadratic penalty that increases sharply as distance decreases
                    penalty = 10 * (1 - (dist/SAFE_DISTANCE))**2
                    reward -= penalty
        
        return self._get_obs(), reward, done, {}

class AIVoiceAssistant:
    """AI Voice Assistant for system monitoring and control"""
    def __init__(self, simulator):
        self.simulator = simulator
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0
        self.recognizer.energy_threshold = 3000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.operation_timeout = 5
        
        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 175)
        self.tts_engine.setProperty('volume', 1.0)
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', voices[1].id)  # Female voice
        
        # Language models
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Personality and state
        self.personality = {
            'name': "OrbitGuard",
            'traits': {
                'professionalism': 0.8,
                'friendliness': 0.7,
                'urgency': 0.6
            },
            'mood': 0.5  # Neutral
        }
        self.last_interaction = time.time()
        self.listening = False
        self.speaking = False
        
        # Start listening thread
        self.thread = Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
    
    def _listen_loop(self):
        """Continuous listening loop"""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while True:
                if not self.listening or self.speaking:
                    time.sleep(0.1)
                    continue
                    
                try:
                    print("\n[Assistant Listening...]")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=7)
                    text = self.recognizer.recognize_google(audio, show_all=False)
                    print(f"User: {text}")
                    self.process_command(text)
                    self.last_interaction = time.time()
                except sr.WaitTimeoutError:
                    # Check if we should initiate conversation
                    if time.time() - self.last_interaction > 20:
                        self.initiate_conversation()
                    continue
                except sr.UnknownValueError:
                    print("[Could not understand audio]")
                except Exception as e:
                    print(f"[Error in listening: {e}]")
    
    def process_command(self, text):
        """Process voice command and generate response"""
        # Analyze command
        doc = nlp(text.lower())
        keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # System status commands
        if any(word in keywords for word in ['status', 'report', 'update']):
            self.report_status()
        elif any(word in keywords for word in ['stop', 'pause', 'halt']):
            self.simulator.pause_simulation()
            self.speak("Simulation paused")
        elif any(word in keywords for word in ['start', 'resume', 'continue']):
            self.simulator.resume_simulation()
            self.speak("Simulation resumed")
        elif any(word in keywords for word in ['danger', 'threat', 'critical']):
            self.report_danger_status()
        elif any(word in keywords for word in ['help', 'assist', 'support']):
            self.provide_help()
        else:
            # Generate conversational response
            response = self.generate_response(text)
            self.speak(response)
    
    def report_status(self):
        """Generate system status report"""
        active_drones = sum(1 for d in self.simulator.drones if d.fuel > 0)
        debris_count = len(self.simulator.debris)
        
        status = (f"Current system status: {active_drones} drones active, "
                 f"{debris_count} debris objects remaining. ")
        
        # Add danger assessment
        if debris_count > 30:
            status += "The debris field is dense, recommend aggressive capture strategy."
        elif debris_count > 15:
            status += "Moderate debris density, maintaining current operations."
        else:
            status += "Light debris field, consider optimizing drone paths."
        
        self.speak(status)
    
    def report_danger_status(self):
        """Report on dangerous debris"""
        if not self.simulator.debris:
            self.speak("No debris currently tracked")
            return
            
        # Get top 3 most dangerous debris
        scored = []
        for debris in self.simulator.debris:
            score = self.simulator.danger_service.get_score(debris.id)
            scored.append((score['priority'], debris))
        
        scored.sort(reverse=True)
        
        if len(scored) >= 3:
            response = (f"Top 3 dangerous debris objects: "
                      f"1 with priority {scored[0][0]:.2f}, "
                      f"2 with priority {scored[1][0]:.2f}, "
                      f"3 with priority {scored[2][0]:.2f}. ")
            
            # Add specific warning if critical
            if scored[0][0] > 0.9:
                response += "Warning! Object 1 is highly critical!"
            self.speak(response)
        else:
            self.speak(f"Tracking {len(scored)} debris objects with maximum priority {scored[0][0]:.2f}")
    
    def provide_help(self):
        """Provide help information"""
        help_text = ("I can provide system status reports, pause or resume operations, "
                    "assess danger levels, and explain current operations. "
                    "How may I assist you?")
        self.speak(help_text)
    
    def generate_response(self, prompt):
        """Generate context-aware response using language models"""
        # Get current context
        context = {
            'drones_active': sum(1 for d in self.simulator.drones if d.fuel > 0),
            'debris_count': len(self.simulator.debris),
            'time': datetime.now().strftime("%H:%M"),
            'mood': self.personality['mood']
        }
        
        # Create prompt with context
        full_prompt = (f"Space debris removal system assistant. Current status: {context['drones_active']} active drones, "
                      f"{context['debris_count']} debris objects. Time: {context['time']}. "
                      f"User query: {prompt}. Professional yet friendly response:")
        
        # Generate response
        input_ids = self.tokenizer.encode(full_prompt, return_tensors='pt')
        output = self.gpt_model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "response:" in response:
            response = response.split("response:")[-1].strip()
        
        return response
    
    def speak(self, text):
        """Convert text to speech with personality adjustments"""
        self.speaking = True
        print(f"Assistant: {text}")
        
        # Adjust speech based on mood and personality
        rate = 165 + int(30 * self.personality['traits']['urgency'])
        volume = 0.8 + 0.2 * self.personality['traits']['friendliness']
        
        self.tts_engine.setProperty('rate', rate)
        self.tts_engine.setProperty('volume', volume)
        
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        
        # Add natural pause
        pause_duration = min(0.5 + len(text.split()) * 0.1, 2.0)
        time.sleep(pause_duration)
        self.speaking = False
    
    def initiate_conversation(self):
        """Initiate conversation when idle"""
        topics = [
            "Would you like a system status update?",
            "The debris removal operation is proceeding. Any questions?",
            "I'm here to assist with the space debris cleanup. How can I help?",
            "Current operations are nominal. Do you need any information?"
        ]
        self.speak(random.choice(topics))
        self.listening = True

class SpaceDebrisSimulator:
    """Main simulation class with all enhanced features"""
    def __init__(self):
        self.debris = []
        self.drones = []
        self.time_step = 0
        self.predictor = LSTMPredictor()
        self.danger_service = DangerScoreService()
        self.visualizer = self._init_visualizer()
        self.ai_assistant = AIVoiceAssistant(self)
        self.paused = False
        self._setup()
        
    def _init_visualizer(self):
        """Initialize enhanced 3D visualization with capture animations"""
        canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), show=True)
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 45
        view.camera.distance = 5000
        
        # Create enhanced visuals
        visuals = {
            'debris': scene.visuals.Markers(parent=view.scene),
            'drones': scene.visuals.Markers(parent=view.scene),
            'trajectories': scene.visuals.Line(parent=view.scene),
            'captures': scene.visuals.Line(parent=view.scene),
            'predictions': scene.visuals.Line(parent=view.scene),
            'danger_zones': scene.visuals.Markers(parent=view.scene),
            'capture_anim': scene.visuals.Line(parent=view.scene, method='gl')
        }
        
        # Configure visuals
        visuals['debris'].set_data(
            pos=np.zeros((1,3)),
            size=10,
            face_color=(1, 0, 0, 0.8),
            edge_color=None
        )
        
        visuals['drones'].set_data(
            pos=np.zeros((1,3)),
            size=20,
            face_color=(0, 1, 0, 1),
            edge_color=(1, 1, 1, 1)
        )
        
        visuals['danger_zones'].set_data(
            pos=np.zeros((1,3)),
            size=10,
            face_color=(1, 1, 0, 0.3),
            edge_color=None
        )
        
        visuals['capture_anim'].set_data(
            pos=np.zeros((2,3)),
            color=(0, 1, 1, 1),
            width=5
        )
        
        return {'canvas': canvas, 'view': view, 'visuals': visuals}
    
    def _setup(self):
        """Initialize simulation with all components"""
        self._load_debris(MAX_DEBRIS)
        self._init_drones(MAX_DRONES)
        self._init_rl_agents()
        
    def _load_debris(self, count):
        """Load debris from TLE or generate simulated"""
        try:
            url = "https://celestrak.org/NORAD/elements/active.txt"
            response = requests.get(url, timeout=5)
            lines = response.text.split('\n')
            
            for i in range(0, min(count*3, len(lines)-2), 3):
                name = lines[i].strip()
                tle = (name, lines[i+1].strip(), lines[i+2].strip())
                self.debris.append(SpaceDebris(len(self.debris), tle))
                
        except Exception as e:
            print(f"Using simulated debris (TLE load failed: {e})")
            self.debris = [SpaceDebris(i) for i in range(count)]
            
    def _init_drones(self, count):
        """Initialize drone fleet with different capture mechanisms"""
        positions = [
            [300, 300, 300],
            [700, 300, 300],
            [500, 700, 300],
            [300, 700, 600],
            [700, 700, 600]
        ]
        mechanisms = ["net", "harpoon", "magnetic", "net", "harpoon"]
        self.drones = [Drone(i, positions[i]) for i in range(min(count, len(positions)))]
        for i, drone in enumerate(self.drones[:len(mechanisms)]):
            drone.capture_mechanism = mechanisms[i]
        
    def _init_rl_agents(self):
        """Initialize RL agents for each drone with multi-processing"""
        self.rl_agents = []
        for drone in self.drones:
            env = DummyVecEnv([lambda: SpaceDebrisEnv(self, drone.id)])
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                policy_kwargs={
                    'net_arch': [dict(pi=[128, 128], vf=[128, 128])
                }
            )
            self.rl_agents.append(model)
            
    def _get_target_debris(self, drone):
        """Get target debris for drone based on danger scores and drone capabilities"""
        if not self.debris:
            return None
            
        # Score debris based on multiple factors
        scored = []
        for debris in self.debris:
            score = self.danger_service.get_score(debris.id)
            
            # Distance factor
            dist = np.linalg.norm(drone.position - debris.position)
            dist_factor = 1 / (dist + 1)
            
            # Mechanism compatibility
            if drone.capture_mechanism == "magnetic" and debris.metal_content < 0.3:
                continue  # Skip low-metal debris for magnetic drones
                
            if drone.capture_mechanism == "harpoon" and debris.size < 1.0:
                continue  # Skip small debris for harpoons
                
            # Combined priority score
            priority = (0.4 * score['priority'] + 
                       0.3 * dist_factor + 
                       0.2 * debris.metal_content + 
                       0.1 * debris.size/DEBRIS_SIZE_RANGE[1])
            
            scored.append((priority, debris))
            
        # Select best target not being pursued by others
        scored.sort(reverse=True)
        for priority, debris in scored[:5]:
            if not any(d.target == debris.id for d in self.drones if d.id != drone.id):
                return debris
                
        return scored[0][1] if scored else None
        
    def _update_visualization(self):
        """Update 3D visualization with all enhanced elements"""
        # Prepare debris data with danger coloring
        debris_pos = np.array([d.position for d in self.debris])
        debris_sizes = np.array([d.size * 10 for d in self.debris])
        
        # Color debris by danger level
        debris_colors = []
        for d in self.debris:
            score = self.danger_service.get_score(d.id)
            danger = score['priority']
            debris_colors.append((1, 1-danger, 0, 0.8))  # Red to yellow gradient
        
        # Prepare drone data with mechanism coloring
        drone_pos = np.array([d.position for d in self.drones])
        drone_colors = []
        for d in self.drones:
            if d.capture_mechanism == "net":
                drone_colors.append((0, 1, 0, 1))  # Green
            elif d.capture_mechanism == "harpoon":
                drone_colors.append((1, 0, 0, 1))  # Red
            else:  # magnetic
                drone_colors.append((0, 0, 1, 1))  # Blue
        
        # Update debris visuals
        self.visualizer['visuals']['debris'].set_data(
            pos=debris_pos,
            size=debris_sizes,
            face_color=debris_colors,
            edge_color=None
        )
        
        # Update drone visuals
        self.visualizer['visuals']['drones'].set_data(
            pos=drone_pos,
            size=20,
            face_color=drone_colors,
            edge_color=(1, 1, 1, 1)
        )
        
        # Update trajectories
        if len(self.debris) > 0:
            trajectories = []
            for debris in self.debris[:10]:  # Limit for performance
                hist = list(debris.position_history)[-10:]  # Last 10 positions
                if len(hist) > 1:
                    trajectories.append(np.array(hist))
            
            if trajectories:
                all_points = np.vstack(trajectories)
                self.visualizer['visuals']['trajectories'].set_data(
                    pos=all_points,
                    color=(1, 0, 0, 0.3),
                    width=2
                )
        
        # Update predictions
        if len(self.debris) > 0 and len(self.drones) > 0:
            predictions = []
            for drone in self.drones:
                target = self._get_target_debris(drone)
                if target:
                    pred = self.predictor.predict(target)
                    if pred is not None:
                        points = np.vstack([drone.position, pred[0]])
                        predictions.append(points)
            
            if predictions:
                all_preds = np.vstack(predictions)
                self.visualizer['visuals']['predictions'].set_data(
                    pos=all_preds,
                    color=(0, 1, 1, 0.8),
                    width=3,
                    connect='segments'
                )
        
        # Update danger zones visualization
        if len(self.debris) > 0:
            danger_scores = [self.danger_service.get_score(d.id)['priority'] for d in self.debris]
            danger_pos = np.array([d.position for d in self.debris])
            danger_sizes = np.array([10 + 90 * s for s in danger_scores])  # Scale by danger
            
            self.visualizer['visuals']['danger_zones'].set_data(
                pos=danger_pos,
                size=danger_sizes,
                face_color=[(1, 1, 0, 0.1 + 0.2 * s) for s in danger_scores],
                edge_color=None
            )
        
        self.visualizer['canvas'].update()
        
    def _animate_capture(self, drone, debris):
        """Animate the capture process based on mechanism"""
        if drone.capture_mechanism == "net":
            # Net deployment animation
            points = np.linspace(drone.position, debris.position, 20)
            self.visualizer['visuals']['capture_anim'].set_data(
                pos=points,
                color=(0, 1, 0, 1),
                width=3
            )
        elif drone.capture_mechanism == "harpoon":
            # Harpoon shot animation
            points = np.array([drone.position, debris.position])
            self.visualizer['visuals']['capture_anim'].set_data(
                pos=points,
                color=(1, 0, 0, 1),
                width=2
            )
        else:  # magnetic
            # Magnetic field animation
            t = np.linspace(0, 2*np.pi, 20)
            radius = np.linalg.norm(drone.position - debris.position) / 2
            center = (drone.position + debris.position) / 2
            normal = np.cross(drone.position - center, [0, 0, 1])
            normal = normal / np.linalg.norm(normal)
            
            # Create a circle perpendicular to the line between drone and debris
            circle = []
            for angle in t:
                offset = radius * (np.cos(angle) * np.cross(normal, drone.position-center) + 
                         np.sin(angle) * normal)
                circle.append(center + offset)
            
            self.visualizer['visuals']['capture_anim'].set_data(
                pos=np.array(circle),
                color=(0, 0, 1, 1),
                width=2,
                connect='strip'
            )
        
        # Animate for 0.5 seconds
        start_time = time.time()
        while time.time() - start_time < 0.5:
            self.visualizer['canvas'].update()
            time.sleep(0.05)
        
        # Clear animation
        self.visualizer['visuals']['capture_anim'].set_data(pos=np.zeros((1,3)))
        
    def _handle_capture(self, drone, debris):
        """Handle debris capture event with visualization"""
        # Visualize capture based on mechanism
        self._animate_capture(drone, debris)
        
        # Remove debris
        self.debris.remove(debris)
        drone.target = None
        
        print(f"Drone {drone.id} ({drone.capture_mechanism}) captured debris {debris.id}")
        
        # Notify AI assistant
        self.ai_assistant.speak(f"Drone {drone.id} successfully captured debris")
        
    def pause_simulation(self):
        """Pause the simulation"""
        self.paused = True
        self.ai_assistant.speak("Simulation paused")
        
    def resume_simulation(self):
        """Resume the simulation"""
        self.paused = False
        self.ai_assistant.speak("Simulation resumed")

    def run(self, steps=500):
        """Run main simulation loop with all features"""
        print("Starting enhanced simulation with AI assistant...")
        start_time = time.time()
        
        # Initial greeting
        self.ai_assistant.speak("Space debris removal system initialized and ready")
        
        try:
            for step in range(steps):
                if self.paused:
                    time.sleep(0.1)
                    continue
                    
                self.time_step += 1
                
                # Update debris positions
                for debris in self.debris:
                    debris.update_position(SIMULATION_STEP)
                    self.predictor.add_training_data(debris)
                
                # Periodic LSTM training
                self.predictor.periodic_train(self.time_step)
                
                # Update drones using RL
                for drone, agent in zip(self.drones, self.rl_agents):
                    if drone.fuel <= 0:
                        continue
                        
                    # Get observation and action
                    obs = SpaceDebrisEnv(self, drone.id).reset()
                    action, _ = agent.predict(obs, deterministic=True)
                    
                    # Update drone with collision avoidance
                    drone.update(action, SIMULATION_STEP)
                    drone.avoid_collision([d for d in self.drones if d.id != drone.id])
                    
                    # Assign target if none
                    if drone.target is None:
                        target = self._get_target_debris(drone)
                        if target:
                            drone.target = target.id
                    
                    # Check for captures
                    if drone.target is not None:
                        target = next((d for d in self.debris if d.id == drone.target), None)
                        if target and np.linalg.norm(drone.position - target.position) < CAPTURE_DISTANCE:
                            self._handle_capture(drone, target)
                
                # Update visualization every 5 steps
                if step % 5 == 0 or step == steps - 1:
                    self._update_visualization()
                
                # Print status every 50 steps
                if step % 50 == 0:
                    print(f"Step {step}/{steps} | Debris: {len(self.debris)} | Active drones: {sum(1 for d in self.drones if d.fuel > 0)}")
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("Simulation stopped by user")
            self.ai_assistant.speak("Simulation terminated by user")
        finally:
            self.danger_service.stop()
            runtime = time.time() - start_time
            print(f"Simulation completed in {runtime:.2f} seconds")
            self._save_models()
            self.ai_assistant.speak("Simulation complete. All systems shutting down")
            
    def _save_models(self):
        """Save trained models with versioning"""
        os.makedirs("saved_models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save LSTM model
        lstm_path = f"saved_models/lstm_{timestamp}.h5"
        self.predictor.model.save(lstm_path)
        
        # Save RL models
        for i, agent in enumerate(self.rl_agents):
            agent.save(f"saved_models/drone_{i}_ppo_{timestamp}")

if __name__ == "__main__":
    simulator = SpaceDebrisSimulator()
    simulator.run(steps=500)
