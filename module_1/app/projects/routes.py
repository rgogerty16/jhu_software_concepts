from flask import Blueprint, render_template

# Create the projects blueprint — named 'projects' for use in url_for() calls
projects_bp = Blueprint('projects', __name__)


# Handle GET requests to "/projects"
@projects_bp.route('/projects')
def projects():
    return render_template('projects.html')
