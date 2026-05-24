from flask import Blueprint, render_template

# Create the home blueprint — 'home' is the name used to reference its
# routes elsewhere (e.g. url_for('home.index') in templates)
home_bp = Blueprint('home', __name__)


# Register this function as the handler for GET requests to "/"
@home_bp.route('/')
def index():
    # Render the home template and return it as the HTTP response
    return render_template('home.html')
