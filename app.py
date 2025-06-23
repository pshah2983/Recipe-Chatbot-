from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from chatbot import Chatbot
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
from flask_migrate import Migrate

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recipes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Instantiate chatbot (make sure your CSV now includes the image_file column)
chatbot = Chatbot()

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    comments = db.relationship('Comment', backref='author', lazy=True)
    favorites = db.relationship('Favorite', backref='user', lazy=True)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipe_id = db.Column(db.Integer, nullable=False)

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipe_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def chat_page():
    return render_template('index.html')

@app.route('/homepage')
def homepage():
    chat_history = []
    if current_user.is_authenticated:
        # Load chat history from session or database (for demo, use session)
        chat_history = session.get('chat_history', [])
    return render_template('home.html', chat_history=chat_history)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        user = User.query.filter_by(username=data['username']).first()
        if user and check_password_hash(user.password_hash, data['password']):
            login_user(user)
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Invalid username or password'})
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.json
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'success': False, 'message': 'Username already exists'})
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'message': 'Email already registered'})
        
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password'])
        )
        db.session.add(user)
        db.session.commit()
        return jsonify({'success': True})
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    user_name = current_user.username if current_user.is_authenticated else None
    response = chatbot.get_response(user_message, user_name=user_name)
    return jsonify({'response': response})

@app.route('/favorite', methods=['POST'])
@login_required
def favorite_recipe():
    recipe_id = request.json.get('recipe_id')
    if not recipe_id:
        return jsonify({'success': False, 'message': 'Recipe ID required'})
    
    existing = Favorite.query.filter_by(user_id=current_user.id, recipe_id=recipe_id).first()
    if existing:
        db.session.delete(existing)
        db.session.commit()
        return jsonify({'success': True, 'action': 'removed'})
    
    favorite = Favorite(user_id=current_user.id, recipe_id=recipe_id)
    db.session.add(favorite)
    db.session.commit()
    return jsonify({'success': True, 'action': 'added'})

@app.route('/comment', methods=['POST'])
@login_required
def add_comment():
    data = request.json
    if not data.get('recipe_id') or not data.get('text'):
        return jsonify({'success': False, 'message': 'Recipe ID and comment text required'})
    
    comment = Comment(
        text=data['text'],
        recipe_id=data['recipe_id'],
        user_id=current_user.id
    )
    db.session.add(comment)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'comment': {
            'id': comment.id,
            'text': comment.text,
            'author': comment.author.username,
            'timestamp': comment.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
    })

@app.route('/comments/<int:recipe_id>')
def get_comments(recipe_id):
    comments = Comment.query.filter_by(recipe_id=recipe_id).order_by(Comment.timestamp.desc()).all()
    return jsonify({
        'comments': [{
            'id': c.id,
            'text': c.text,
            'author': c.author.username,
            'timestamp': c.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        } for c in comments]
    })

# Route to serve static images from the image_for_cuisines/data folder.
@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('image_for_cuisines/data', filename)

@app.route('/save_chat', methods=['POST'])
def save_chat():
    if not current_user.is_authenticated:
        return ('', 204)
    chat_history = session.get('chat_history', [])
    data = request.get_json()
    chat_history.append({
        'message': data.get('message', ''),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
    })
    session['chat_history'] = chat_history[-20:]  # Keep only last 20 messages
    return ('', 204)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist.
    app.run(debug=True)
