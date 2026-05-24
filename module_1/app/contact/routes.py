from flask import Blueprint, render_template

# Create the contact blueprint — named 'contact' for use in url_for() calls
contact_bp = Blueprint('contact', __name__)


# Handle GET requests to "/contact"
@contact_bp.route('/contact')
def contact():
    return render_template('contact.html')
