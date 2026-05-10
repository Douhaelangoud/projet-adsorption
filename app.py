from flask import Flask, render_template, redirect, url_for, request, flash, session, jsonify, send_file
from forms import SignupForm
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from flask_wtf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from datetime import datetime
import numpy as np
import os
import time
import sys
import io
import base64
import socket
import json
import requests
import ollama
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import Config
from flask_migrate import Migrate
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from translations import TRANSLATIONS, get_text


# -------------------- CONFIGURATION -------------------- #
app = Flask(__name__, template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config['SECRET_KEY']

# Language support
SUPPORTED_LANGUAGES = ['en', 'fr', 'es']
DEFAULT_LANGUAGE = 'en'

# Get local IP address for external access
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()
SERVER_PORT = 5000

# Debug complet de la configuration email
print("\n=== CONFIGURATION EMAIL ===")
print(f"Serveur SMTP : {app.config['MAIL_SERVER']}")
print(f"Port : {app.config['MAIL_PORT']}")
print(f"TLS : {app.config['MAIL_USE_TLS']} | SSL : {app.config['MAIL_USE_SSL']}")
print(f"Expéditeur : {app.config['MAIL_DEFAULT_SENDER']}")
print(f"Username : {app.config['MAIL_USERNAME']}")
print(f"Password configurée : {'*'*len(app.config['MAIL_PASSWORD']) if app.config['MAIL_PASSWORD'] else 'AUCUN MOT DE PASSE TROUVÉ'}")
print(f"Sender : {app.config['MAIL_DEFAULT_SENDER']}\n")

# Initialisation des extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
mail = Mail(app)
csrf = CSRFProtect(app)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# ==================== LANGUAGE MANAGEMENT ====================
def get_current_language():
    """Get the current language from session or default"""
    if 'language' not in session:
        session['language'] = DEFAULT_LANGUAGE
    return session.get('language', DEFAULT_LANGUAGE)

def get_text_by_key(key):
    """Helper to get translated text in template"""
    lang = get_current_language()
    return get_text(key, lang)

# Session-based login_required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Context processor to pass language-related data to templates
@app.context_processor
def inject_language():
    lang = session.get('lang', 'fr')
    return {
        'current_language': lang,
        'supported_languages': [
            {'code': 'en', 'name': 'English', 'flag': '🇬🇧'},
            {'code': 'fr', 'name': 'Français', 'flag': '🇫🇷'},
            {'code': 'es', 'name': 'Español', 'flag': '🇪🇸'},
        ],
        'gettext': lambda key: get_text(key, lang),
        'app_name': get_text('app_name', lang),
        'app_description': get_text('app_description', lang),
    }

# -------------------- MODELE UTILISATEUR -------------------- #
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    firstname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    activation_token = db.Column(db.String(500), unique=True, nullable=True)
    failed_login_attempts = db.Column(db.Integer, default=0)
    last_failed_login = db.Column(db.Float, default=0)
    ip_address = db.Column(db.String(50), nullable=True)
    mac_address = db.Column(db.String(255), nullable=True)
    is_admin = db.Column(db.Boolean, default=False)
    is_banned = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True) 

    # New fields for dashboard and analytics
    domain = db.Column(db.String(50), nullable=True)  # Google, Facebook, WhatsApp, LinkedIn, GitHub, Other
    how_found_app = db.Column(db.String(100), nullable=True)  # How they discovered the app
    created_at = db.Column(db.DateTime, default=db.func.now(), nullable=False)
    calculations = db.relationship('Calculation', backref='user', lazy=True, cascade='all, delete-orphan')

# -------------------- MODELE CALCUL -------------------- #
class Calculation(db.Model):
    __tablename__ = 'calculation'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    calculation_type = db.Column(db.String(20), nullable=False)  # 'absorption' ou 'desorption'
    mode = db.Column(db.String(20), nullable=False)  # 'stages' ou 'target'
    timestamp = db.Column(db.DateTime, default=db.func.now(), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    
    # Paramètres entrée (JSON stringified)
    parameters = db.Column(db.Text, nullable=False)
    
    # Résultats (JSON stringified)
    results = db.Column(db.Text, nullable=False)
    
    # Graphique (base64 encoded)
    plot_url = db.Column(db.Text, nullable=False)

# -------------------- INITIALISATION BASE DE DONNEES -------------------- #
with app.app_context():
    try:
        db.create_all()
        print("✅ Base de données initialisée avec succès")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'initialisation de la BD: {e}")
# -------------------- CALCUL DE DÉSORPTION -------------------- #
def calcul_desorption(G_prime, m, L, x0, N_etages):
    """
    Calcul du facteur de désorption et des concentrations par étage
    Paramètres:
    - G_prime: Débit de gaz
    - m: Masse de sorbant
    - L: Débit du liquide/solide
    - x0: Concentration initiale
    - N_etages: Nombre d'étages
    """
    # Calcul du facteur de désorption
    S = (G_prime * m) / L
    
    # Pré-allocation
    x_entrant = np.zeros(N_etages)
    x_sortant = np.zeros(N_etages)
    x_entrant[0] = x0
    
    # Calcul par étage
    for i in range(N_etages):
        x_sortant[i] = x_entrant[i] / (1 + S)
        if i < N_etages - 1:
            x_entrant[i+1] = x_sortant[i]
    
    # Rendement
    rendement = (x0 - x_sortant[-1]) / x0 * 100
    
    return S, x_entrant, x_sortant, rendement

def calcul_etages_necessaires(G_prime, m, L, x0, x_objectif):
    """
    Calcul du nombre d'étages nécessaires pour atteindre un objectif de concentration
    Paramètres:
    - G_prime: Débit de gaz
    - m: Masse de sorbant
    - L: Débit du liquide/solide
    - x0: Concentration initiale
    - x_objectif: Concentration finale objectif
    """
    # Calcul du facteur de désorption
    S = (G_prime * m) / L
    
    # Si S = 0, impossible de désorbér
    if S <= 0:
        return None, None, None, None, "Le facteur de désorption doit être positif"
    
    # Vérifier que l'objectif est réalisable
    if x_objectif >= x0:
        return None, None, None, None, "L'objectif doit être inférieur à la concentration initiale"
    
    # Calcul du nombre d'étages requis
    # x_final = x0 / (1 + S)^n
    # log(x_final) = log(x0) - n * log(1 + S)
    # n = (log(x0) - log(x_final)) / log(1 + S)
    
    N_etages = int(np.ceil(np.log(x0 / x_objectif) / np.log(1 + S)))
    
    # Vérifier que N_etages est réaliste (max 100 étages)
    if N_etages > 100:
        return None, None, None, None, f"Nombre d'étages requis trop élevé ({N_etages}). Ajustez vos paramètres."
    
    # Recalculer avec le nombre d'étages exact pour avoir des résultats précis
    x_entrant = np.zeros(N_etages)
    x_sortant = np.zeros(N_etages)
    x_entrant[0] = x0
    
    for i in range(N_etages):
        x_sortant[i] = x_entrant[i] / (1 + S)
        if i < N_etages - 1:
            x_entrant[i+1] = x_sortant[i]
    
    rendement = (x0 - x_sortant[-1]) / x0 * 100
    
    return S, x_entrant, x_sortant, rendement, N_etages

# -------------------- ABSORPTION CALCULATION -------------------- #
def calcul_absorption(L, G, yo, num_stages=5):
    """
    Calcul de l'absorption graphique (Diagramme McCabe-Thiele)
    Paramètres:
    - L: Débit du liquide
    - G: Débit du gaz
    - yo: Concentration initiale du gaz (fraction décimale, ex: 0.06)
    - num_stages: Nombre d'étages à calculer
    """
    try:
        slope = L / G
        Y0 = yo / (1 - yo) * 100 if (1 - yo) != 0 else yo * 100
        
        # Fonction d'intersection pour le diagramme McCabe-Thiele
        def intersection(Y):
            X = Y / (0.5 + slope)
            Yeq = 0.5 * X
            return X, Yeq
        
        # Calcul des étages
        stages_data = []
        Y_current = Y0
        
        for i in range(num_stages):
            X, Y_eq = intersection(Y_current)
            stages_data.append({
                'stage': i + 1,
                'X': round(X, 4),
                'Y': round(Y_eq, 4),
                'Y_operational': round(Y_current, 4)
            })
            Y_current = Y_eq
        
        # Génération du graphique
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Courbe d'équilibre
        x = np.linspace(0, 8, 100)
        y_eq = 0.5 * x
        ax.plot(x, y_eq, label="Equilibrium Curve: Y = 0.5X", color='blue', linewidth=2)
        
        # Droites opératoires et points d'intersection
        colors = plt.cm.autumn(np.linspace(0, 1, num_stages))
        X_points = []
        Y_points = []
        
        Y_current = Y0
        for i, stage_info in enumerate(stages_data):
            X, Y_eq = stage_info['X'], stage_info['Y']
            X_points.append(X)
            Y_points.append(Y_eq)
            
            # Droite opératoire
            y_op = Y_current - slope * x
            ax.plot(x, y_op, color=colors[i], linewidth=1.5, alpha=0.7)
            
            # Point d'intersection
            ax.scatter([X], [Y_eq], color='red', s=100, zorder=5)
            
            # Projections avec tirets
            ax.plot([X, X], [0, Y_eq], 'k--', linewidth=0.8, alpha=0.5)
            ax.plot([0, X], [Y_eq, Y_eq], 'k--', linewidth=0.8, alpha=0.5)
            
            # Annotations
            ax.text(-0.5, Y_current, f'Y{i}', fontsize=9, ha='right')
            ax.text(X, -0.3, f'X{i}', fontsize=9, ha='center')
            
            Y_current = Y_eq
        
        # Point initial
        ax.scatter([0], [Y0], color='green', s=150, marker='s', zorder=5, label=f'Initial: Y0={Y0:.2f}%')
        
        # Configuration des axes
        ax.set_xlabel("X (%)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Y (%)", fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, 8)
        ax.set_ylim(-0.5, 7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper left')
        ax.set_title(f"Absorption Diagram - McCabe-Thiele Method (Stages: {num_stages})", 
                     fontsize=12, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        # Convert to base64
        img = io.BytesIO()
        fig.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
        # Calculate final efficiency
        final_Y = Y_points[-1] if Y_points else Y0
        efficiency = ((Y0 - final_Y) / Y0 * 100) if Y0 > 0 else 0
        
        return {
            'L': L,
            'G': G,
            'slope': round(slope, 4),
            'Y0': round(Y0, 4),
            'stages': stages_data,
            'final_Y': round(final_Y, 4),
            'efficiency': round(efficiency, 2),
            'plot_url': plot_url,
            'num_stages': num_stages
        }
    
    except Exception as e:
        return {
            'error': str(e)
        }

def calcul_etages_absorption_necessaires(L, G, yo, y_target):
    """
    Calcul du nombre d'étages nécessaires pour atteindre un objectif de concentration
    Paramètres:
    - L: Débit du liquide
    - G: Débit du gaz
    - yo: Concentration initiale du gaz (fraction décimale)
    - y_target: Concentration cible en Y (fraction décimale)
    """
    try:
        slope = L / G
        Y0 = yo / (1 - yo) * 100 if (1 - yo) != 0 else yo * 100
        Y_target = y_target / (1 - y_target) * 100 if (1 - y_target) != 0 else y_target * 100
        
        # Vérifications
        if slope <= 0:
            return {'error': 'Slope L/G must be positive'}
        
        if Y_target >= Y0:
            return {'error': 'Target concentration must be lower than initial concentration'}
        
        # Fonction d'intersection
        def intersection(Y):
            X = Y / (0.5 + slope)
            Yeq = 0.5 * X
            return X, Yeq
        
        # Calcul itératif du nombre d'étages
        num_stages = 0
        Y_current = Y0
        stages_data = []
        max_stages = 100
        
        while Y_current > Y_target and num_stages < max_stages:
            X, Y_eq = intersection(Y_current)
            num_stages += 1
            stages_data.append({
                'stage': num_stages,
                'X': round(X, 4),
                'Y': round(Y_eq, 4),
                'Y_operational': round(Y_current, 4)
            })
            Y_current = Y_eq
        
        if num_stages >= max_stages:
            return {'error': f'Too many stages required ({num_stages}). Adjust your parameters.'}
        
        # Génération du graphique
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Courbe d'équilibre
        x = np.linspace(0, 8, 100)
        y_eq = 0.5 * x
        ax.plot(x, y_eq, label="Equilibrium Curve: Y = 0.5X", color='blue', linewidth=2)
        
        # Droites opératoires et points d'intersection
        colors = plt.cm.autumn(np.linspace(0, 1, num_stages))
        
        Y_current = Y0
        for i, stage_info in enumerate(stages_data):
            X, Y_eq = stage_info['X'], stage_info['Y']
            
            # Droite opératoire
            y_op = Y_current - slope * x
            ax.plot(x, y_op, color=colors[i], linewidth=1.5, alpha=0.7)
            
            # Point d'intersection
            ax.scatter([X], [Y_eq], color='red', s=100, zorder=5)
            
            # Projections avec tirets
            ax.plot([X, X], [0, Y_eq], 'k--', linewidth=0.8, alpha=0.5)
            ax.plot([0, X], [Y_eq, Y_eq], 'k--', linewidth=0.8, alpha=0.5)
            
            # Annotations
            ax.text(-0.5, Y_current, f'Y{i}', fontsize=9, ha='right')
            ax.text(X, -0.3, f'X{i}', fontsize=9, ha='center')
            
            Y_current = Y_eq
        
        # Point initial et target
        ax.scatter([0], [Y0], color='green', s=150, marker='s', zorder=5, label=f'Initial: Y0={Y0:.2f}%')
        ax.axhline(y=Y_target, color='orange', linestyle='--', linewidth=2, label=f'Target: Y={Y_target:.2f}%')
        
        # Configuration des axes
        ax.set_xlabel("X (%)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Y (%)", fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, 8)
        ax.set_ylim(-0.5, max(Y0, 7) + 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper left')
        ax.set_title(f"Absorption Diagram - Target Achieved (Stages: {num_stages})", 
                     fontsize=12, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        # Convert to base64
        img = io.BytesIO()
        fig.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
        # Calculate efficiency
        final_Y = Y_current
        efficiency = ((Y0 - final_Y) / Y0 * 100) if Y0 > 0 else 0
        
        return {
            'L': L,
            'G': G,
            'slope': round(slope, 4),
            'Y0': round(Y0, 4),
            'Y_target': round(Y_target, 4),
            'final_Y': round(final_Y, 4),
            'stages': stages_data,
            'efficiency': round(efficiency, 2),
            'plot_url': plot_url,
            'num_stages': num_stages
        }
    
    except Exception as e:
        return {'error': str(e)}

# -------------------- AI SEARCH FUNCTION -------------------- #
def search_web(query):
    """
    Search the web using SerpAPI for additional context
    """
    try:
        import serpapi
        # You'll need to set your SerpAPI key in environment or config
        api_key = os.getenv('SERPAPI_KEY')  # Add this to your environment
        if not api_key:
            return "Search functionality not configured (missing SERPAPI_KEY)"
        
        client = serpapi.Client(api_key=api_key)
        results = client.search({
            "q": query,
            "num": 3  # Limit to 3 results for context
        })
        
        if 'organic_results' in results:
            context = "\n".join([
                f"- {result.get('title', '')}: {result.get('snippet', '')}"
                for result in results['organic_results'][:3]
            ])
            return f"Web search results for '{query}':\n{context}"
        else:
            return f"No search results found for '{query}'"
    except ImportError:
        return "Search functionality not available (serpapi not installed)"
    except Exception as e:
        return f"Search error: {str(e)}"

# -------------------- ROUTES -------------------- #

# Language switcher route
@app.route('/set-language/<language>')
def set_language(language):
    codes = ['en', 'fr', 'es']  # ✅ liste simple
    if language in SUPPORTED_LANGUAGES:
        session['lang'] = language
        flash(f'Language changed to {language.upper()}', 'info')
    return redirect(request.referrer or url_for('home'))

@app.route('/')
def home():
    lang = session.get('lang', 'fr')
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    """Admin dashboard with user statistics"""
    if not session.get('user_id'):
        flash('Please log in to access the dashboard', 'error')
        return redirect(url_for('login'))
    
    lang = session.get('lang', 'fr')

    # Get statistics
    total_users = User.query.count()
    
    # Users by domain
    domains = db.session.query(User.domain, db.func.count(User.id)).group_by(User.domain).all()
    domain_stats = {domain[0] or 'Not Specified': domain[1] for domain in domains}
    
    # Users by how they found us
    found_by = db.session.query(User.how_found_app, db.func.count(User.id)).group_by(User.how_found_app).all()
    found_by_stats = {source[0] or 'Direct': source[1] for source in found_by}
    

    # Calculate weekly and monthly stats
    from datetime import datetime, timedelta
    now = datetime.now()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
    weekly_users = User.query.filter(User.created_at >= week_ago).count()
    monthly_users = User.query.filter(User.created_at >= month_ago).count()
    
    return render_template('dashboard.html',
                         total_users=total_users,
                         domain_stats=domain_stats,
                         found_by_stats=found_by_stats,
                         weekly_users=weekly_users,
                         monthly_users=monthly_users,
                         lang=lang,
                         get_text=get_text)
                         

# -------------------- AI CHAT ROUTES -------------------- #
@app.route('/chat')
@login_required
def chat():
    """AI Chat Assistant - only for authenticated users"""
    lang = session.get('lang', 'fr')
    return render_template('chat.html', lang=lang, get_text=get_text)

@app.route('/chat-api', methods=['POST'])
@login_required
@csrf.exempt  # Exempt from CSRF since it's an API endpoint protected by session auth
def chat_api():
    """API endpoint for AI chat - only for authenticated users"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Optional: Add web search context for better responses
        search_context = ""
        if len(user_message) > 10:  # Only search for longer queries
            search_context = search_web(user_message)
        
        # Prepare prompt - AI can answer about anything
        system_prompt = """You are a helpful AI assistant for "Assiyaton" - an Absorption & Desorption Calculator application.

You can help users with:
1. Questions about the application (how to use calculators, history, PDF reports, features, login, signup, etc.)
2. Chemistry and chemical engineering concepts (absorption, desorption, McCabe-Thiele method, equilibrium, mass transfer, etc.)
3. General science and technology questions

CONFIDENTIAL - DO NOT SHARE:
- User account information (usernames, emails, passwords)
- Calculation results from other users
- Admin information or system internals
- Any personal or sensitive data

If asked about confidential topics, respond: "I cannot share that information. It's confidential."

GUIDELINES:
- Be helpful, friendly, and informative
- If you don't know something, admit it honestly
- You can search the web for additional information when needed
- Keep responses clear and concise
- Feel free to engage in conversation on any topic

ABOUT ASSIYATON (if asked):
- Chemical engineering calculator using McCabe-Thiele method
- Features: absorption calculator, desorption calculator, history, PDF reports
- Available in English, French, Spanish
- Requires email activation after signup
- Has admin panel for user management

Now respond helpfully to the user's question:"""
        
        if search_context and not search_context.startswith("Search"):
            system_prompt += f"\n\nAdditional context from web search:\n{search_context}"
        
        print(f"[CHAT API] Calling Ollama with message: {user_message[:50]}...")
        
        # Call Ollama
        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ]
        )
        
        ai_reply = response['message']['content']
        print(f"[CHAT API] Ollama response: {ai_reply[:50]}...")
        return jsonify({'reply': ai_reply})
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"[CHAT API ERROR] {error_msg}")
        print(f"[CHAT API TRACE]\n{error_trace}")
        return jsonify({'reply': f'AI is currently unavailable: {error_msg}'}), 500
        
        if search_context and not search_context.startswith("Search"):
            system_prompt += f"\n\nAdditional context from web search:\n{search_context}"
        
        print(f"[CHAT API] Calling Ollama with message: {user_message[:50]}...")
        
        # Call Ollama
        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ]
        )
        
        ai_reply = response['message']['content']
        print(f"[CHAT API] Ollama response: {ai_reply[:50]}...")
        return jsonify({'reply': ai_reply})
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"[CHAT API ERROR] {error_msg}")
        print(f"[CHAT API TRACE]\n{error_trace}")
        return jsonify({'reply': f'AI is currently unavailable: {error_msg}'}), 500

@app.route('/deactivate_account')
def deactivate_account():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    user.is_active = False
    db.session.commit()

    lang = session.get('lang', 'fr')  # 👈 save language

    session.clear()  # logout automatique

    session['lang'] = lang  # 👈 restore language

    flash('Your account has been deactivated', 'info')
    return redirect(url_for('login'))

@app.route('/api/dashboard-stats')
def api_dashboard_stats():
    """API endpoint for dashboard statistics"""
    if not session.get('user_id'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    lang = session.get('lang', 'fr')  # 👈 ICI tu ajoutes
    
    # Get statistics
    domains = db.session.query(User.domain, db.func.count(User.id)).group_by(User.domain).all()
    domain_stats = {domain[0] or 'Not Specified': domain[1] for domain in domains}
    
    found_by = db.session.query(User.how_found_app, db.func.count(User.id)).group_by(User.how_found_app).all()
    found_by_stats = {source[0] or 'Direct': source[1] for source in found_by}
    
    return jsonify({
        'domains': domain_stats,
        'found_by': found_by_stats,
        'total_users': User.query.count(),
        'lang': session.get('lang', 'fr')  # 👈 ICI tu ajoutes
    })

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    lang = session.get('lang', 'fr')   # 👈 AJOUT 1 (LANGUE)

    if request.method == 'POST':
        username = request.form.get('username')
        firstname = request.form.get('firstname')
        email = request.form.get('email')
        email_confirm = request.form.get('email_confirm')
        password = request.form.get('password')
        ip_address = request.remote_addr

        # 🔒 CAPTCHA GOOGLE
        captcha_response = request.form.get("g-recaptcha-response")
        
        # Validate reCAPTCHA
        if not captcha_response:
            flash("Please complete the reCAPTCHA verification.")
            return redirect(url_for('signup'))
        
        # Verify reCAPTCHA with Google
        recaptcha_secret = app.config['RECAPTCHA_PRIVATE_KEY']
        recaptcha_verify_url = 'https://www.google.com/recaptcha/api/siteverify'
        recaptcha_payload = {
            'secret': recaptcha_secret,
            'response': captcha_response
        }
        
        try:
            recaptcha_response = requests.post(recaptcha_verify_url, data=recaptcha_payload)
            recaptcha_result = recaptcha_response.json()
            
            if not recaptcha_result.get('success'):
                flash("reCAPTCHA verification failed. Please try again.")
                return redirect(url_for('signup'))
        except Exception as e:
            flash("Error verifying reCAPTCHA. Please try again.")
            return redirect(url_for('signup'))

        # Validate that both emails exist and match
        if not email or not email_confirm:
            flash("Both email fields are required.")
            return redirect(url_for('signup'))

        if email != email_confirm:
            flash("Emails do not match. Please verify.")
            return redirect(url_for('signup'))

        # Validate password with all requirements: 8+ chars, uppercase, lowercase, number, special char
        import re
        password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]).{8,}$'
        
        if not re.match(password_pattern, password):
            flash("Password must contain: 8+ characters, uppercase, lowercase, number, and special character.")
            return redirect(url_for('signup'))

        if not all([username, firstname, email, password]):
            flash("All fields are required.")
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash("Email already in use.")
            return redirect(url_for('signup'))

        # Generate activation token
        activation_token = s.dumps(email, salt='email-activation')
        activation_link = url_for('activate_account', token=activation_token, _external=True)
        
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(
            username=username, 
            firstname=firstname, 
            email=email, 
            password=hashed_pw,
            is_active=False,
            activation_token=activation_token,
            domain=request.form.get('domain', 'Other'),
            how_found_app=request.form.get('how_found_app', 'Direct'),
            ip_address=request.remote_addr,
            mac_address=request.headers.get('User-Agent')
        )
        db.session.add(new_user)
        db.session.commit()

        # Send activation email
        try:
            # Generate activation URL with local IP instead of localhost
            activation_url = f"http://{LOCAL_IP}:{SERVER_PORT}/activate/{activation_token}"
            
            msg = Message(
                'Activate Your Assiyaton Account - Account Activation Required',
                sender=app.config['MAIL_DEFAULT_SENDER'],
                recipients=[email]
            )
            
            msg.html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                    }}
                    .container {{
                        max-width: 600px;
                        margin: 0 auto;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 0;
                        border-radius: 15px;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 40px;
                        text-align: center;
                        border-radius: 15px 15px 0 0;
                    }}
                    .header h1 {{
                        margin: 0;
                        font-size: 32px;
                        font-weight: 300;
                        letter-spacing: 1px;
                    }}
                    .content {{
                        background: white;
                        padding: 40px;
                        border-radius: 0 0 15px 15px;
                    }}
                    .greeting {{
                        font-size: 18px;
                        color: #2c3e50;
                        margin-bottom: 25px;
                    }}
                    .message {{
                        color: #555;
                        font-size: 15px;
                        line-height: 1.8;
                        margin-bottom: 30px;
                    }}
                    .activation-button {{
                        display: inline-block;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 15px 40px;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        font-size: 16px;
                        transition: transform 0.3s ease, box-shadow 0.3s ease;
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                        margin: 30px 0;
                    }}
                    .activation-button:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
                    }}
                    .activation-link {{
                        color: #667eea;
                        text-decoration: none;
                        word-break: break-all;
                        font-size: 13px;
                        background: #f5f5f5;
                        padding: 15px;
                        border-radius: 8px;
                        display: block;
                        margin-top: 20px;
                        border-left: 4px solid #667eea;
                    }}
                    .footer {{
                        background: #f8f9fa;
                        padding: 25px;
                        text-align: center;
                        color: #666;
                        font-size: 12px;
                        border-radius: 0 0 15px 15px;
                        border-top: 1px solid #e0e0e0;
                    }}
                    .security-note {{
                        background: #fff3cd;
                        border-left: 4px solid #ffc107;
                        padding: 15px;
                        border-radius: 5px;
                        margin-top: 25px;
                        font-size: 13px;
                        color: #856404;
                    }}
                    .security-note strong {{
                        display: block;
                        margin-bottom: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎯 Assiyaton</h1>
                        <p style="margin: 10px 0 0 0; font-size: 14px; opacity: 0.9;">Account Activation</p>
                    </div>
                    <div class="content">
                        <div class="greeting">
                            👋 Welcome, <strong>{firstname}</strong>!
                        </div>
                        <div class="message">
                            Thank you for registering with <strong>Assiyaton - Absorption & Desorption Calculator</strong>. 
                            We're excited to have you on board! 🚀
                            <br><br>
                            To complete your registration and activate your account, please click the button below:
                        </div>
                        <center>
                            <a href="{activation_url}" class="activation-button">
                                ✓ Activate My Account
                            </a>
                        </center>
                        <div class="activation-link">
                            <strong>Or copy this link:</strong><br>
                            {activation_url}
                        </div>
                        <div class="security-note">
                            <strong>🔒 Security Notice:</strong>
                            This link will expire in 24 hours for your security. 
                            If you did not create this account, please ignore this email.
                        </div>
                    </div>
                    <div class="footer">
                        <p>© 2026 Assiyaton - All rights reserved</p>
                        <p>Questions? Reply to this email or contact our support team</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            msg.body = f"Click here to activate your account: {activation_link}"
            mail.send(msg)
            flash("Account created! Please check your email to activate your account.", 'success')
        except Exception as e:
            app.logger.error(f"Error sending activation email: {str(e)}")
            flash("Account created but email sending failed. Please contact support.", 'warning')
        
        return redirect(url_for('login'))
    return render_template('signup.html', lang=lang, get_text=get_text, form=form)
     
@app.route('/login', methods=['GET', 'POST'])
def login():
    lang = session.get('lang', 'fr')  # 👈 AJOUT

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if not email or not password:
            flash("Please fill in all fields.")
            return redirect(url_for('login'))

        if user:
            # Check if account is activated
            if not user.is_active:
                flash("Your account is not activated. Please check your email for the activation link.")
                return redirect(url_for('login'))

            current_time = time.time()
            if user.failed_login_attempts >= 3:
                time_since_last_attempt = current_time - user.last_failed_login
                if time_since_last_attempt < 60:
                    flash("Too many attempts. Please wait 1 minute before trying again.")
                    return redirect(url_for('login'))
                else:
                    user.failed_login_attempts = 0
                    db.session.commit()

            if check_password_hash(user.password, password):
                session['user_id'] = user.id
                session['username'] = user.firstname
                user.failed_login_attempts = 0
                db.session.commit()
                return redirect(url_for('dashboard'))
            else:
                user.failed_login_attempts += 1
                user.last_failed_login = current_time
                db.session.commit()
                remaining_attempts = 3 - user.failed_login_attempts
                if remaining_attempts > 0:
                    flash(f"Incorrect password. You have {remaining_attempts} attempts left.")
                else:
                    flash("Too many incorrect attempts. Please wait 1 minute before retrying.")
                return redirect(url_for('login'))
        else:
            flash("Incorrect email or password.")
            return redirect(url_for('login'))
    return render_template('login.html', lang=lang, get_text=get_text)

@app.route('/activate/<token>', methods=['GET'])
def activate_account(token):
    lang = session.get('lang', 'fr')  # 👈 AJOUT

    try:
        email = s.loads(token, salt='email-activation', max_age=86400)  # 24 hours
    except SignatureExpired:
        flash("Activation link has expired. Please sign up again.", 'error')
        return redirect(url_for('signup'))
    except Exception as e:
        app.logger.error(f"Activation token error: {str(e)}")
        flash("Invalid activation link.", 'error')
        return redirect(url_for('signup'))

    user = User.query.filter_by(email=email).first()
    if user:
        if user.is_active:
            flash("Your account is already activated! Please log in.", 'info')
        else:
            user.is_active = True
            user.activation_token = None
            db.session.commit()
            flash("🎉 Your account has been successfully activated! You can now log in.", 'success')
        return redirect(url_for('login'))
    else:
        flash("User not found.", 'error')
        return redirect(url_for('signup'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    lang = session.get('lang', 'fr')

    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            try:
                token = s.dumps(email, salt='email-reset')
                reset_url = url_for('reset_password', token=token, _external=True)
                
                msg = Message('Réinitialisation du mot de passe',
                            sender=app.config['MAIL_DEFAULT_SENDER'],
                            recipients=[email])
                msg.body = f'''Pour réinitialiser votre mot de passe, visitez le lien suivant:
{reset_url}

Si vous n'avez pas demandé de réinitialisation, ignorez simplement cet email.
'''
                mail.send(msg)
                flash('Un email avec les instructions de réinitialisation a été envoyé.', 'success')
            except Exception as e:
                app.logger.error(f"Erreur d'envoi d'email: {str(e)}")
                flash(f"Erreur lors de l'envoi: {str(e)}", 'error')
        else:
            flash("Cet email n'est associé à aucun compte.", 'error')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html', lang=lang, get_text=get_text)

@app.route('/reset/<token>', methods=['GET', 'POST'])
def reset_password(token):
    lang = session.get('lang', 'fr')

    try:
        email = s.loads(token, salt='email-reset', max_age=600)
    except SignatureExpired:
        flash("Le lien de réinitialisation a expiré.", 'error')
        return redirect(url_for('forgot_password'))
    except:
        flash("Lien invalide.", 'error')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form.get('password')
        confirm = request.form.get('confirm')

        if not new_password or not confirm:
            flash("Tous les champs sont requis.", 'error')
        elif new_password != confirm:
            flash("Les mots de passe ne correspondent pas.", 'error')
        elif len(new_password) < 8 or not any(c.isupper() for c in new_password) or not any(c.isdigit() for c in new_password):
            flash("Le mot de passe doit contenir au moins 8 caractères, une majuscule et un chiffre.", 'error')
        else:
            user = User.query.filter_by(email=email).first()
            if user:
                user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
                db.session.commit()
                flash("Mot de passe mis à jour avec succès!", 'success')
                return redirect(url_for('login'))
            else:
                flash("Utilisateur non trouvé.", 'error')
    
    return render_template('reset.html', lang=lang, get_text=get_text)

@app.route('/formulaire', methods=['GET'])
def formulaire():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    lang = session.get('language', 'fr')
    return render_template('formulaire.html', lang=lang, get_text=get_text)

@app.route('/absorption', methods=['GET'])
def absorption():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    lang = session.get('language', 'fr')
    return render_template('absorption.html', lang=lang, get_text=get_text)

@app.route('/resultat', methods=['POST'])
def resultat():
    try:
        lang = session.get('lang', 'fr')
        calculation_type = request.form.get('calculation_type', 'desorption')
        
        # Sauvegarder les paramètres d'entrée originaux
        input_params = {}
        
        if calculation_type == 'absorption':
            # ==================== ABSORPTION CALCULATION ====================
            L = float(request.form['L'])
            G = float(request.form['G'])
            yo = float(request.form['yo'])
            
            # Stocker les paramètres d'entrée
            input_params = {
                'L': L,
                'G': G,
                'yo': yo
            }
            
            # Déterminer le mode de calcul
            mode = request.form.get('absorption_mode', 'stages')
            input_params['mode'] = mode
            
            if mode == 'stages':
                # Mode 1 : Nombre d'étages donné
                num_stages = int(request.form['num_stages'])
                input_params['num_stages'] = num_stages
                resultats_data = calcul_absorption(L, G, yo, num_stages)
                
                if 'error' in resultats_data:
                    flash(f"Error: {resultats_data['error']}")
                    return redirect(url_for('absorption'))
                
                resultats = resultats_data
                resultats['calculation_type'] = 'absorption'
                resultats['mode'] = 'stages'
                resultats['mode_info'] = f"Known Stages: {num_stages}"
            
            else:
                # Mode 2 : Objectif de concentration donné (Y target)
                y_target = float(request.form['y_target'])
                input_params['y_target'] = y_target
                resultats_data = calcul_etages_absorption_necessaires(L, G, yo, y_target)
                
                if 'error' in resultats_data:
                    flash(f"Error: {resultats_data['error']}")
                    return redirect(url_for('absorption'))
                
                resultats = resultats_data
                resultats['calculation_type'] = 'absorption'
                resultats['mode'] = 'target'
                resultats['mode_info'] = f"Target Concentration: {y_target:.4f} | Calculated Stages: {resultats['num_stages']}"
            
        else:
            # ==================== DESORPTION CALCULATION ====================
            G_prime = float(request.form['G_prime'])
            m = float(request.form['m'])
            L = float(request.form['L'])
            x0 = float(request.form['x0'])
            
            # Stocker les paramètres d'entrée
            input_params = {
                'G_prime': G_prime,
                'm': m,
                'L': L,
                'x0': x0
            }
            
            # Déterminer le mode de calcul (normalize: etages → stages, objectif → target)
            mode_raw = request.form.get('mode', 'etages')
            mode = 'stages' if mode_raw == 'etages' else 'target'
            input_params['mode'] = mode
            
            if mode == 'stages':
                # Mode 1 : Nombre d'étages donné
                N_etages = int(request.form['N_etages'])
                input_params['N_etages'] = N_etages
                S, x_entrant, x_sortant, rendement = calcul_desorption(G_prime, m, L, x0, N_etages)
                mode_info = f"Number of stages: {N_etages}"
            else:
                # Mode 2 : Objectif de concentration donné
                x_objectif = float(request.form['x_objectif'])
                input_params['x_objectif'] = x_objectif
                result = calcul_etages_necessaires(G_prime, m, L, x0, x_objectif)
                
                if len(result) == 5 and isinstance(result[4], str):
                    # Erreur
                    flash(result[4])
                    return redirect(url_for('formulaire'))
                
                S, x_entrant, x_sortant, rendement, N_etages = result
                mode_info = f"Target concentration: {x_objectif} | Calculated stages: {N_etages}"
            
            # Génération du graphique
            etages = np.arange(1, N_etages+1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(etages, x_entrant, 'bo-', label="Inlet x", linewidth=2, markersize=8)
            plt.plot(etages, x_sortant, 'ro-', label="Outlet x", linewidth=2, markersize=8)
            
            plt.xlabel("Number of Stages", fontsize=12)
            plt.ylabel("Concentration x", fontsize=12)
            plt.title("Desorption Curves", fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Sauvegarder en image base64
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            # Préparer les données pour affichage
            resultats = {
                'S': round(S, 6),
                'rendement': round(rendement, 2),
                'x0': float(x0),
                'N_etages': int(N_etages),
                'x_final': round(float(x_sortant[-1]), 6),
                'mode': mode,
                'mode_info': mode_info,
                'etages': [int(x) for x in etages],
                'x_entrant': [round(float(x), 6) for x in x_entrant],
                'x_sortant': [round(float(x), 6) for x in x_sortant],
                'plot_url': plot_url,
                'calculation_type': 'desorption'
            }
        
        # ==================== SAVE CALCULATION ==================== 
        if 'user_id' in session:
            try:
                user_id = session['user_id']
                plot_data = resultats.get('plot_url', '')
                
                # Déterminer le titre du calcul
                calculation_type = resultats.get('calculation_type', 'unknown')
                mode = resultats.get('mode', '')
                timestamp_str = time.strftime('%d/%m/%Y %H:%M:%S')
                
                # Créer un titre descriptif
                calc_number = Calculation.query.filter_by(user_id=user_id, calculation_type=calculation_type).count() + 1
                title = f"{calculation_type.capitalize()} Calculation #{calc_number} - {timestamp_str}"
                
                # Préparer les résultats (sans le plot_url qui est trop gros)
                results_copy = dict(resultats)
                results_copy.pop('plot_url', None)
                
                # Sauvegarder dans la BD
                calculation = Calculation(
                    user_id=user_id,
                    calculation_type=calculation_type,
                    mode=mode,
                    title=title,
                    parameters=json.dumps(input_params),  # Paramètres d'entrée
                    results=json.dumps(results_copy),      # Résultats calculés
                    plot_url=plot_data
                )
                db.session.add(calculation)
                db.session.commit()
                
                # Ajouter l'ID dans les résultats pour affichage
                resultats['calculation_id'] = calculation.id
                resultats['saved'] = True
                
            except Exception as e:
                app.logger.error(f"Error saving calculation: {str(e)}")
                app.logger.error(f"Traceback: {e}", exc_info=True)
                resultats['saved'] = False
        
        return render_template('resultat.html', resultats=resultats, lang=lang)

    except ValueError as e:
        flash(f"Error: Check your parameters (invalid numbers)")
        return redirect(url_for('formulaire'))
    except Exception as e:
        flash(f"Erreur lors du traitement : {e}")
        return redirect(url_for('formulaire'))

# ==================== HISTORIQUE ROUTES ====================

@app.route('/historique-absorption', methods=['GET'])
def historique_absorption():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    calculations = Calculation.query.filter_by(
        user_id=user_id, 
        calculation_type='absorption'
    ).order_by(Calculation.timestamp.desc()).all()
    
    return render_template('historique.html', 
                         calculations=calculations, 
                         calc_type='absorption',
                         username=session.get('username'))

@app.route('/historique-desorption', methods=['GET'])
def historique_desorption():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    calculations = Calculation.query.filter_by(
        user_id=user_id,
        calculation_type='desorption'
    ).order_by(Calculation.timestamp.desc()).all()
    
    return render_template('historique.html',
                         calculations=calculations,
                         calc_type='desorption',
                         username=session.get('username'))

@app.route('/historique/view/<int:calc_id>', methods=['GET'])
def view_calculation(calc_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    calculation = Calculation.query.get(calc_id)
    if not calculation or calculation.user_id != session['user_id']:
        flash("Calculation not found", 'error')
        return redirect(url_for('dashboard'))
    
    resultats = json.loads(calculation.results)
    resultats['plot_url'] = calculation.plot_url
    resultats['calculation_id'] = calculation.id
    resultats['calculation_type'] = calculation.calculation_type
    resultats['mode'] = calculation.mode
    resultats['title'] = calculation.title
    
    return render_template('resultat.html', resultats=resultats)

@app.route('/historique/delete/<int:calc_id>', methods=['POST'])
def delete_calculation(calc_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    calculation = Calculation.query.get(calc_id)
    if not calculation or calculation.user_id != session['user_id']:
        flash("Calculation not found", 'error')
        return redirect(url_for('dashboard'))
    
    calc_type = calculation.calculation_type
    db.session.delete(calculation)
    db.session.commit()
    
    flash("Calculation deleted successfully", 'success')
    
    if calc_type == 'absorption':
        return redirect(url_for('historique_absorption'))
    else:
        return redirect(url_for('historique_desorption'))

@app.route('/historique/edit/<int:calc_id>', methods=['GET', 'POST'])
def edit_calculation(calc_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    calculation = Calculation.query.get(calc_id)
    if not calculation or calculation.user_id != session['user_id']:
        flash("Calculation not found", 'error')
        return redirect(url_for('dashboard'))
    
    params = json.loads(calculation.parameters)
    results = json.loads(calculation.results)
    
    if request.method == 'POST':
        # Recalculate with new parameters
        calc_type = calculation.calculation_type
        
        try:
            if calc_type == 'absorption':
                # Get new parameters from form
                L = float(request.form.get('L', params['L']))
                G = float(request.form.get('G', params['G']))
                yo = float(request.form.get('yo', params['yo']))
                mode = request.form.get('absorption_mode', params['mode'])
                
                # Update parameters
                new_params = {'L': L, 'G': G, 'yo': yo, 'mode': mode}
                
                if mode == 'stages':
                    num_stages = int(request.form.get('num_stages', params.get('num_stages', 5)))
                    new_params['num_stages'] = num_stages
                    resultats_data = calcul_absorption(L, G, yo, num_stages)
                else:
                    y_target = float(request.form.get('y_target', params.get('y_target', 0.01)))
                    new_params['y_target'] = y_target
                    resultats_data = calcul_etages_absorption_necessaires(L, G, yo, y_target)
                
                if 'error' in resultats_data:
                    flash(f"Error: {resultats_data['error']}")
                    return render_template('edit_calculation.html', 
                                         calculation=calculation,
                                         params=params,
                                         username=session.get('username'))
                
                # Update the calculation in DB
                calculation.parameters = json.dumps(new_params)
                calculation.results = json.dumps({k: v for k, v in resultats_data.items() if k != 'plot_url'})
                calculation.plot_url = resultats_data.get('plot_url', '')
                db.session.commit()
                
                flash("Calculation updated successfully!", 'success')
                return redirect(url_for('view_calculation', calc_id=calc_id))
                
            else:  # desorption
                # Similar logic for desorption
                G_prime = float(request.form.get('G_prime', params['G_prime']))
                m = float(request.form.get('m', params['m']))
                L = float(request.form.get('L', params['L']))
                x0 = float(request.form.get('x0', params['x0']))
                mode = request.form.get('mode', params.get('mode', 'stages'))
                
                new_params = {'G_prime': G_prime, 'm': m, 'L': L, 'x0': x0, 'mode': mode}
                
                if mode == 'stages':
                    N_etages = int(request.form.get('N_etages', params.get('N_etages', 3)))
                    new_params['N_etages'] = N_etages
                    S, x_entrant, x_sortant, rendement = calcul_desorption(G_prime, m, L, x0, N_etages)
                else:
                    x_objectif = float(request.form.get('x_objectif', params.get('x_objectif', 0.001)))
                    new_params['x_objectif'] = x_objectif
                    result = calcul_etages_necessaires(G_prime, m, L, x0, x_objectif)
                    
                    if len(result) == 5 and isinstance(result[4], str):
                        flash(result[4])
                        return render_template('edit_calculation.html',
                                             calculation=calculation,
                                             params=params,
                                             username=session.get('username'))
                    
                    S, x_entrant, x_sortant, rendement, N_etages = result
                
                # Generate chart
                etages = np.arange(1, N_etages+1)
                plt.figure(figsize=(10, 6))
                plt.plot(etages, x_entrant, 'bo-', label="Inlet x", linewidth=2, markersize=8)
                plt.plot(etages, x_sortant, 'ro-', label="Outlet x", linewidth=2, markersize=8)
                plt.xlabel("Number of Stages", fontsize=12)
                plt.ylabel("Concentration x", fontsize=12)
                plt.title("Desorption Curves", fontsize=14)
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                img = io.BytesIO()
                plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close()
                
                new_results = {
                    'S': round(S, 6),
                    'rendement': round(rendement, 2),
                    'x0': x0,
                    'N_etages': N_etages,
                    'x_final': round(float(x_sortant[-1]), 6),
                    'mode': mode,
                    'etages': list(etages),
                    'x_entrant': [round(float(x), 6) for x in x_entrant],
                    'x_sortant': [round(float(x), 6) for x in x_sortant],
                }
                
                # Update the calculation in DB
                calculation.parameters = json.dumps(new_params)
                calculation.results = json.dumps(new_results)
                calculation.plot_url = plot_url
                db.session.commit()
                
                flash("Calculation updated successfully!", 'success')
                return redirect(url_for('view_calculation', calc_id=calc_id))
        
        except Exception as e:
            app.logger.error(f"Error updating calculation: {str(e)}")
            flash(f"Error: {str(e)}", 'error')
            return render_template('edit_calculation.html',
                                 calculation=calculation,
                                 params=params,
                                 username=session.get('username'))
    
    return render_template('edit_calculation.html',
                         calculation=calculation,
                         params=params,
                         username=session.get('username'))

# ==================== REPORT GENERATION ====================

def generate_absorption_report(calculation):
    """Generate a PDF report for absorption calculation"""
    params = json.loads(calculation.parameters)
    results = json.loads(calculation.results)
    
    # Create PDF
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1abc9c'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Title
    story.append(Paragraph("ABSORPTION CALCULATION REPORT", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Date and information
    info_data = [
        ['Report Date:', datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
        ['Calculation Mode:', params.get('mode', 'Unknown').upper()],
        ['Timestamp:', calculation.timestamp.strftime('%d/%m/%Y %H:%M:%S')],
    ]
    info_table = Table(info_data, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Input Parameters
    story.append(Paragraph("INPUT PARAMETERS", heading_style))
    input_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Liquid Flow Rate (L)', str(params.get('L', 'N/A')), 'units'],
        ['Gas Flow Rate (G)', str(params.get('G', 'N/A')), 'units'],
        ['Initial Concentration (yo)', str(params.get('yo', 'N/A')), 'fraction'],
    ]
    if params.get('mode') == 'stages':
        input_data.append(['Number of Stages', str(params.get('num_stages', 'N/A')), 'stages'])
    else:
        input_data.append(['Target Concentration', str(params.get('y_target', 'N/A')), 'fraction'])
    
    input_table = Table(input_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1abc9c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Results
    story.append(Paragraph("CALCULATION RESULTS", heading_style))
    result_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Slope (L/G)', f"{results.get('slope', 'N/A')}", '-'],
        ['Initial Y (Y0)', f"{results.get('Y0', 'N/A')}%", '%'],
        ['Final Y', f"{results.get('final_Y', 'N/A')}%", '%'],
        ['Number of Stages', f"{results.get('num_stages', 'N/A')}", 'stages'],
        ['Efficiency', f"{results.get('efficiency', 'N/A')}%", '%'],
    ]
    
    result_table = Table(result_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Interpretation
    story.append(Paragraph("INTERPRETATION & ANALYSIS", heading_style))
    L = params.get('L', 1)
    G = params.get('G', 1)
    efficiency = results.get('efficiency', 0)
    
    interpretation = f"""
    <b>Performance Summary:</b><br/>
    The absorption process achieved an efficiency of <b>{efficiency}%</b> with a liquid to gas ratio (L/G) of <b>{L/G:.2f}</b>.<br/>
    The system required <b>{results.get('num_stages', 'N/A')} stages</b> to reduce the inlet concentration from <b>{results.get('Y0', 'N/A')}%</b> to <b>{results.get('final_Y', 'N/A')}%</b>.<br/>
    <br/>
    <b>Key Observations:</b><br/>
    • The McCabe-Thiele graphical method was used for stage-by-stage analysis<br/>
    • Higher L/G ratios improve absorption efficiency<br/>
    • The operational line intersects the equilibrium curve at each stage<br/>
    • This design ensures optimal mass transfer conditions<br/>
    """
    
    story.append(Paragraph(interpretation, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    story.append(Paragraph("RECOMMENDATIONS", heading_style))
    recommendations = """
    <b>Design Recommendations:</b><br/>
    • Consider increasing the number of stages if higher removal efficiency is required<br/>
    • Optimize liquid and gas flow rates for cost-effective operation<br/>
    • Monitor pressure drop across stages during operation<br/>
    • Implement regular maintenance of absorption equipment<br/>
    """
    story.append(Paragraph(recommendations, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

def generate_desorption_report(calculation):
    """Generate a PDF report for desorption calculation"""
    params = json.loads(calculation.parameters)
    results = json.loads(calculation.results)
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#e74c3c'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Title
    story.append(Paragraph("DESORPTION CALCULATION REPORT", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Info
    info_data = [
        ['Report Date:', datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
        ['Calculation Mode:', params.get('mode', 'Unknown').upper()],
        ['Timestamp:', calculation.timestamp.strftime('%d/%m/%Y %H:%M:%S')],
    ]
    info_table = Table(info_data, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Input Parameters
    story.append(Paragraph("INPUT PARAMETERS", heading_style))
    input_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Gas Flow Rate (G\')', str(params.get('G_prime', 'N/A')), 'm³/s'],
        ['Sorbent Mass (m)', str(params.get('m', 'N/A')), 'kg'],
        ['Liquid Flow Rate (L)', str(params.get('L', 'N/A')), 'm³/s'],
        ['Initial Concentration (x0)', str(params.get('x0', 'N/A')), 'fraction'],
    ]
    if params.get('mode') == 'etages':
        input_data.append(['Number of Stages', str(params.get('N_etages', 'N/A')), 'stages'])
    else:
        input_data.append(['Target Concentration', str(params.get('x_objectif', 'N/A')), 'fraction'])
    
    input_table = Table(input_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Results
    story.append(Paragraph("CALCULATION RESULTS", heading_style))
    S = results.get('S', 'N/A')
    result_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Desorption Factor (S)', f"{S}", '-'],
        ['Initial Concentration', f"{results.get('x0', 'N/A')}", 'fraction'],
        ['Final Concentration', f"{results.get('x_final', 'N/A')}", 'fraction'],
        ['Number of Stages', f"{results.get('N_etages', 'N/A')}", 'stages'],
        ['Efficiency', f"{results.get('rendement', 'N/A')}%", '%'],
    ]
    
    result_table = Table(result_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Interpretation
    story.append(Paragraph("INTERPRETATION & ANALYSIS", heading_style))
    efficiency = results.get('rendement', 0)
    
    interpretation = f"""
    <b>Performance Summary:</b><br/>
    The desorption process achieved an efficiency of <b>{efficiency}%</b> with a desorption factor (S) of <b>{S}</b>.<br/>
    The system required <b>{results.get('N_etages', 'N/A')} stages</b> to reduce the inlet concentration from <b>{results.get('x0', 'N/A')}</b> to <b>{results.get('x_final', 'N/A')}</b>.<br/>
    <br/>
    <b>Key Observations:</b><br/>
    • Higher S values indicate more effective desorption<br/>
    • The concentration decreases exponentially through each stage<br/>
    • Mass transfer efficiency improves with increased gas flow<br/>
    • Stage configuration significantly impacts final product quality<br/>
    """
    
    story.append(Paragraph(interpretation, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    story.append(Paragraph("RECOMMENDATIONS", heading_style))
    recommendations = """
    <b>Process Recommendations:</b><br/>
    • Adjust gas flow rate to optimize desorption efficiency<br/>
    • Consider temperature effects on desorption rates<br/>
    • Implement quality control measures for outlet streams<br/>
    • Regular monitoring of adsorbent saturation levels<br/>
    """
    story.append(Paragraph(recommendations, styles['Normal']))
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

@app.route('/rapport/<int:calc_id>')
def generate_rapport(calc_id):
    """Generate and download PDF report for a calculation"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    calculation = Calculation.query.get(calc_id)
    if not calculation or calculation.user_id != session['user_id']:
        flash("Calculation not found", 'error')
        return redirect(url_for('dashboard'))
    
    try:
        if calculation.calculation_type == 'absorption':
            pdf_buffer = generate_absorption_report(calculation)
        else:
            pdf_buffer = generate_desorption_report(calculation)
        
        filename = f"{calculation.calculation_type}_report_{calc_id}.pdf"
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        app.logger.error(f"Error generating report: {str(e)}")
        flash(f"Error generating report: {str(e)}", 'error')
        return redirect(url_for('view_calculation', calc_id=calc_id))

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/admin')
def admin_panel():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    if not user or not user.is_admin:
        return "Access denied"

    users = User.query.all()
    new_user = User.query.order_by(User.id.desc()).first()

    # 🔥 CORRECTION LOGIQUE
    active_users = User.query.filter(
        User.is_active == True,
        User.is_banned == False
    ).count()

    inactive_users = User.query.filter(
        User.is_active == False,
        User.is_banned == False
    ).count()

    banned_users = User.query.filter(
        User.is_banned == True
    ).count()

    return render_template(
        'admin.html',
        users=users,
        active_users=active_users,
        inactive_users=inactive_users,
        banned_users=banned_users,
        new_user=new_user
    )

@app.route('/send_status_mail/<int:id>')
def send_status_mail(id):
    user = User.query.get(id)

    if user.is_admin:
        subject = "Account Status Update - Admin Access Granted"
        body = f"""
Dear {user.firstname},

We are pleased to inform you that your account has been updated.

✔ Status: ADMIN ACCESS GRANTED
✔ You now have full administrative privileges.

You can now access all administrative features of the platform.

Best regards,  
Administration Team
"""

    elif user.is_banned:
        subject = "Account Status Update - Account Suspended"
        body = f"""
Dear {user.firstname},

We regret to inform you that your account has been suspended.

❌ Status: BANNED
❌ Access to the platform has been restricted.

If you believe this is a mistake, please contact support.

Best regards,  
Support Team
"""

    elif user.is_active:
        subject = "Account Status Update - Activation Successful"
        body = f"""
Dear {user.firstname},

We are pleased to inform you that your account has been successfully activated.

🎉 Status: ACTIVE
✔ You now have full access to the platform.

Thank you for joining us.

Best regards,  
Support Team
"""
        
        login_url = url_for('login', _external=True)
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
                    padding: 0;
                    border-radius: 15px;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
                }}
                .header {{
                    background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                    border-radius: 15px 15px 0 0;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 32px;
                    font-weight: 300;
                    letter-spacing: 1px;
                }}
                .content {{
                    background: white;
                    padding: 40px;
                    border-radius: 0 0 15px 15px;
                }}
                .greeting {{
                    font-size: 18px;
                    color: #2c3e50;
                    margin-bottom: 25px;
                }}
                .message {{
                    color: #555;
                    font-size: 15px;
                    line-height: 1.8;
                    margin-bottom: 30px;
                }}
                .status-badge {{
                    display: inline-block;
                    background: #d4edda;
                    color: #155724;
                    padding: 12px 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                    font-weight: 600;
                    border-left: 4px solid #28a745;
                }}
                .login-button {{
                    display: inline-block;
                    background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
                    color: white;
                    padding: 15px 40px;
                    text-decoration: none;
                    border-radius: 30px;
                    font-weight: 600;
                    font-size: 16px;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    box-shadow: 0 4px 15px rgba(26, 188, 156, 0.4);
                    margin: 30px 0;
                    text-align: center;
                }}
                .login-button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(26, 188, 156, 0.6);
                }}
                .footer {{
                    background: #f8f9fa;
                    padding: 25px;
                    text-align: center;
                    color: #666;
                    font-size: 12px;
                    border-radius: 0 0 15px 15px;
                    border-top: 1px solid #e0e0e0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Assiyaton</h1>
                    <p style="margin: 10px 0 0 0; font-size: 14px; opacity: 0.9;">Account Activated</p>
                </div>
                <div class="content">
                    <div class="greeting">
                        👋 Bienvenue, <strong>{user.firstname}</strong>!
                    </div>
                    <div class="message">
                        We are pleased to inform you that your account has been successfully activated.
                    </div>
                    <div class="status-badge">
                        🎉 Status: ACTIVE<br/>
                        ✔ You now have full access to the platform.
                    </div>
                    <div class="message">
                        Thank you for joining us. Click the button below to complete your connection:
                    </div>
                    <center>
                        <a href="{login_url}" class="login-button">
                            ✓ Complete Your Connection
                        </a>
                    </center>
                </div>
                <div class="footer">
                    <p>© 2026 Assiyaton - All rights reserved</p>
                    <p>Questions? Contact our support team</p>
                </div>
            </div>
        </body>
        </html>
        """

    else:
        subject = "Account Status Update - Pending Activation"
        body = f"""
Dear {user.firstname},

Your account is currently under review or inactive.

⏳ Status: INACTIVE
Please complete any required steps to activate your account.

We appreciate your patience.

Best regards,  
Support Team
"""

    msg = Message(
        subject=subject,
        sender=app.config['MAIL_DEFAULT_SENDER'],
        recipients=[user.email],
        body=body
    )
    
    # Add HTML version for activation emails
    if user.is_active and 'html_body' in locals():
        msg.html = html_body

    mail.send(msg)
    flash("Mail sent successfully to the user!", 'success')

    return redirect('/admin')

@app.route('/activate_user/<int:id>')
def activate_user(id):
    user = User.query.get(id)
    user.is_active = True
    user.is_banned = False
    db.session.commit()
    return redirect('/admin')


@app.route('/deactivate_user/<int:id>')
def deactivate_user(id):
    user = User.query.get(id)
    user.is_active = False
    user.is_banned = False
    db.session.commit()
    return redirect('/admin')


@app.route('/ban_user/<int:id>')
def ban_user(id):
    user = User.query.get(id)
    user.is_banned = True
    user.is_active = False
    db.session.commit()
    return redirect('/admin')


@app.route('/make_admin/<int:id>')
def make_admin(id):
    user = User.query.get(id)
    user.is_admin = True
    db.session.commit()
    return redirect('/admin')

# 🔥 👉 AJOUTE ICI
@app.route('/guide')
def guide():
    if not session.get('user_id'):
         return redirect(url_for('login'))
    return render_template('guide.html')


@app.route('/edit_user/<int:id>', methods=['GET', 'POST'])
def edit_user(id):
    user = User.query.get(id)

    if request.method == 'POST':
        user.username = request.form['username']
        user.email = request.form['email']
        user.first_name = request.form['first_name']
        user.last_name = request.form['last_name']

        db.session.commit()
        return redirect('/admin')

    return render_template('edit_user.html', user=user)

@app.route('/fix_users')
def fix_users():
    users = User.query.all()
    for u in users:
        if u.is_banned is None:
            u.is_banned = False
    db.session.commit()
    return "Users fixed"

@app.route('/reset_users_logic')
def reset_users_logic():
    users = User.query.all()

    for u in users:
        # règle propre métier
        if u.is_banned:
            u.is_active = False
        else:
            # si pas banni → garder actif ou non selon ton choix
            if u.is_active is None:
                u.is_active = False

        if u.is_banned is None:
            u.is_banned = False

    db.session.commit()
    return "Database logic fixed"

def get_contributors():
    return [
        {
            'slug': 'douha-elangoud',
            'name': 'Douha Elangoud',
            'role': 'Développeuse Back-end',
            'bio': 'Douha est une développeuse back-end spécialisée en Python et Flask. Elle a conçu et implémenté le système d\'authentification, la gestion des utilisateurs et la base de données de l\'application.',
            'skills': ['Python', 'Flask', 'SQLAlchemy', 'PostgreSQL', 'API REST'],
            'linkedin': 'https://www.linkedin.com/in/douha-elangoud-821908312/',
            'github': 'https://github.com/Douhaelangoud',
            'photo': 'douha-elangoud.jpg'
        },
        {
            'slug': 'assiya-elkhayati',
            'name': 'Assiya Elkhayati',
            'role': 'Chef de Projet & Fullstack',
            'bio': 'Assiya a dirigé l\'ensemble du projet, coordonné l\'équipe et assuré la qualité du produit final. Elle a également contribué au développement front-end et à l\'intégration des composants.',
            'skills': ['JavaScript', 'HTML/CSS', 'Bootstrap', 'Gestion de projet', 'UI/UX'],
            'linkedin': 'https://www.linkedin.com/in/assiya-elkhayati-318b4721a/',
            'github': 'https://github.com/Elkhayatiassiya',
            'photo': 'assiya-elkhayati.jpg'
        },
        {
            'slug': 'axelle-tankoano',
            'name': 'Axelle Tankoano',
            'role': 'UI/UX Designer',
            'bio': 'Axelle a conçu l\'interface utilisateur et l\'expérience utilisateur de l\'application. Elle a créé le design moderne et intuitif que vous voyez aujourd\'hui.',
            'skills': ['Figma', 'Adobe XD', 'Design System', 'Prototyping', 'User Research'],
            'linkedin': 'https://www.linkedin.com/in/annick-axelle-tankoano-6a46a8299/',
            'github': 'https://github.com/Tankoano-Annick-Axelle',
            'photo': 'axelle-tankoano.jpg'
        },
        {
            'slug': 'wissal-achehab',
            'name': 'Wissal Achehab',
            'role': 'Développeuse Front-end',
            'bio': 'Wissal a développé l\'interface utilisateur interactive, un logiciel pour le dimensionnment des colonnes et du cout energétique, implémenté les formulaires dynamiques et assuré la compatibilité cross-browser de l\'application.',
            'skills': ['JavaScript', 'React', 'HTML5', 'CSS3', 'Responsive Design'],
            'linkedin': 'https://www.linkedin.com/in/wissal-achehab-a8a229340/',
            'github': 'https://github.com/ACHEHAB-WISSAL',
            'photo': 'wissal-achehab.jpg'
        },
        {
            'slug': 'soumaya-afoussi',
            'name': 'Soumaya Afoussi',
            'role': 'Analyste Qualité',
            'bio': 'Soumaya a effectué les tests fonctionnels, l\'assurance qualité et la validation des fonctionnalités. Elle a contribué à l\'optimisation des performances et de l\'expérience utilisateur.',
            'skills': ['Testing', 'QA', 'Selenium', 'Performance Testing', 'User Acceptance Testing'],
            'linkedin': 'https://www.linkedin.com/in/soumya-afoussi-189347314/',
            'github': 'https://github.com/soumyaafoussi0224',
            'photo': 'soumaya-afoussi.jpg'
        }
    ]

@app.route('/contributors')
def contributors():
    return render_template('contributors.html', contributors=get_contributors())

@app.route('/contributors/github')
def contributors_github():
    return render_template('contributors_github.html', contributors=get_contributors())

@app.route('/contributors/<slug>')
def contributor_profile(slug):
    contributor = next((c for c in get_contributors() if c['slug'] == slug), None)
    if not contributor:
        return render_template('404.html'), 404
    return render_template('contributor_profile.html', contributor=contributor)

@app.route('/contributors/github/<slug>')
def contributor_github_profile(slug):
    contributor = next((c for c in get_contributors() if c['slug'] == slug), None)
    if not contributor:
        return render_template('404.html'), 404
    return render_template('contributor_github_profile.html', contributor=contributor)

if __name__ == '__main__':
    # Test SMTP au démarrage
    try:
        with app.app_context():
            test_msg = Message(
                subject="Test SMTP - Configuration réussie",
                sender=app.config['MAIL_DEFAULT_SENDER'],
                recipients=[app.config['MAIL_USERNAME']],  # Envoi à vous-même
                body="Votre configuration SMTP fonctionne correctement!"
            )
            mail.send(test_msg)
            print("\n=== TEST SMTP RÉUSSI ===")
            print("Un email de test a été envoyé avec succès!")
    except Exception as e:
        print("\n=== ERREUR SMTP ===")
        print(f"Erreur lors du test SMTP: {str(e)}")
        print("Vérifiez votre configuration dans .env")
    
    print("\n" + "="*60)
    print("🚀 ASSIYATON - Absorption & Desorption Calculator")
    print("="*60)
    print(f"✅ Serveur lancé sur : http://{LOCAL_IP}:{SERVER_PORT}")
    print(f"✅ Accès local : http://localhost:{SERVER_PORT}")
    print(f"✅ Accessible depuis d'autres appareils sur le même réseau")
    print("="*60 + "\n")
    
    # 🔥 AJOUT ADMIN ICI
    with app.app_context():
        user = User.query.first()
        if user:
            user.is_admin = True
            db.session.commit()

    
    # Run Flask server on 0.0.0.0 to be accessible from other devices
    app.run(host='0.0.0.0', port=SERVER_PORT, debug=True)