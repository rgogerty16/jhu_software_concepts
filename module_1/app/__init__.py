from flask import Flask


def create_app():
    app = Flask(__name__)

    # Register each page as its own blueprint
    from app.home.routes import home_bp
    from app.contact.routes import contact_bp
    from app.projects.routes import projects_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(contact_bp)
    app.register_blueprint(projects_bp)

    return app
